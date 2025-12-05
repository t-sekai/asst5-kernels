// flash_attention.cu (Step 1 variant)
// Matches the original submission layout but assigns one cooperative block per
// query token to parallelize dot products and keep accumulators in SRAM.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

// Constants -------------------------------------------------------------------
#define THREADS_PER_BLOCK 128

// ------------------------------------------------------------------------
// Reduction helpers (warp and block level)
// ------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ float block_allreduce_sum(float val) {
  __shared__ float shared[THREADS_PER_BLOCK / warpSize];

  float sum = warp_reduce_sum(val);
  int lane = threadIdx.x & (warpSize - 1);
  int warp_id = threadIdx.x / warpSize;

  if (lane == 0) {
    shared[warp_id] = sum;
  }
  __syncthreads();

  sum = (lane < (THREADS_PER_BLOCK / warpSize)) ? shared[lane] : 0.0f;
  sum = warp_reduce_sum(sum);

  __shared__ float total;
  if (threadIdx.x == 0) {
    total = sum;
  }
  __syncthreads();
  return total;
}

// ------------------------------------------------------------------------
// CUDA Kernel (Step 1)
// ------------------------------------------------------------------------

template <typename scalar_t>
__global__ void block_parallel_flash_attention_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {
  // 1. Block <-> Query mapping (same as submission.cu, now cooperative)
  int q_idx = blockIdx.x;
  if (q_idx >= batch_size * num_heads * seq_len) {
    return;
  }

  int token_idx = q_idx % seq_len;
  int head_idx = (q_idx / seq_len) % num_heads;
  int batch_idx = q_idx / (seq_len * num_heads);

  // 2. Base offsets under (B, H, S, D) layout
  int base_offset = (batch_idx * num_heads * seq_len * head_dim) +
                    (head_idx * seq_len * head_dim);

  const scalar_t* q_vec = Q + base_offset + (token_idx * head_dim);
  const scalar_t* k_head_ptr = K + base_offset;
  const scalar_t* v_head_ptr = V + base_offset;
  scalar_t* o_vec = O + base_offset + (token_idx * head_dim);

  extern __shared__ float shared_mem[];
  float* shared_out = shared_mem;          // head_dim floats
  float* shared_q = shared_out + head_dim; // head_dim floats

  // 3. Load Q row and zero the shared accumulator cooperatively
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    shared_q[d] = static_cast<float>(q_vec[d]);
    shared_out[d] = 0.0f;
  }
  __syncthreads();

  // Online softmax state lives in registers/shared scalars
  float m_i = -INFINITY;
  float l_i = 0.0f;

  __shared__ float shared_alpha;
  __shared__ float shared_beta;
  __shared__ float shared_norm;
  if (threadIdx.x == 0) {
    shared_norm = 1e-6f; // avoid division by zero
  }
  __syncthreads();

  // 4. Iterate across the full sequence (no shared-memory tiling yet,
  //    but dot-products/updates are now cooperative)
  for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
    const scalar_t* k_vec = k_head_ptr + k_idx * head_dim;
    const scalar_t* v_vec = v_head_ptr + k_idx * head_dim;

    // A. Cooperative dot product for Q_i . K_j
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      float k_val = static_cast<float>(k_vec[d]);
      partial += shared_q[d] * k_val;
    }

    float score = block_allreduce_sum(partial) * scale;

    // B. Online softmax update (single-thread update, broadcast via shared mem)
    if (threadIdx.x == 0) {
      float m_prev = m_i;
      m_i = fmaxf(m_i, score);
      float alpha = expf(m_prev - m_i);
      float beta = expf(score - m_i);
      l_i = l_i * alpha + beta;

      shared_alpha = alpha;
      shared_beta = beta;
      shared_norm = l_i;
    }
    __syncthreads();

    float alpha = shared_alpha;
    float beta = shared_beta;

    // C. Update the shared output accumulator cooperatively
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      float v_val = static_cast<float>(v_vec[d]);
      shared_out[d] = shared_out[d] * alpha + v_val * beta;
    }
    __syncthreads();
  }

  // 5. Final normalization and writeback
  float norm = shared_norm + 1e-6f;
  float inv_norm = 1.0f / norm;

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float out_val = shared_out[d] * inv_norm;
    o_vec[d] = static_cast<scalar_t>(out_val);
  }
}

// ------------------------------------------------------------------------
// C++ / Python Interface
// ------------------------------------------------------------------------

torch::Tensor flash_attention_forward_step1(torch::Tensor Q, torch::Tensor K,
                                            torch::Tensor V) {
  auto O = torch::empty_like(Q);

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  // Grid/block config mirrors submission.cu but with cooperative blocks
  int total_queries = batch_size * num_heads * seq_len;
  dim3 blocks(total_queries);
  dim3 threads(THREADS_PER_BLOCK);

  size_t shared_mem_size = 2 * head_dim * sizeof(float); // q row + output

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "block_parallel_flash_attention_kernel", ([&] {
        block_parallel_flash_attention_kernel<scalar_t>
            <<<blocks, threads, shared_mem_size>>>(
                Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(), batch_size,
                num_heads, seq_len, head_dim, scale);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  return O;
}
