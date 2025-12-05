// flash_attention.cu
// Template for FlashAttention CUDA Kernel Submission

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

// Constants
#define THREADS_PER_BLOCK 128
#define TILE_SEQ 32

// ------------------------------------------------------------------------
// CUDA Kernel Implementation
// ------------------------------------------------------------------------

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ float block_allreduce_sum(float val) {
  __shared__ float shared[THREADS_PER_BLOCK / 32];

  float sum = warp_reduce_sum(val);
  int lane = threadIdx.x & (warpSize - 1);
  int warp_id = threadIdx.x / warpSize;

  if (lane == 0) {
    shared[warp_id] = sum;
  }
  __syncthreads();

  sum = (lane < (THREADS_PER_BLOCK / 32)) ? shared[lane] : 0.0f;
  sum = warp_reduce_sum(sum);

  __shared__ float total;
  if (threadIdx.x == 0) {
    total = sum;
  }
  __syncthreads();
  return total;
}

template <typename scalar_t>
__global__ void flash_attention_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {
  extern __shared__ __align__(16) float shared_mem[];
  float* shared_q = shared_mem;
  float* shared_out = shared_q + head_dim;
  float* shared_k = shared_out + head_dim;
  float* shared_v = shared_k + 2 * TILE_SEQ * head_dim;

  __shared__ float shared_alpha;
  __shared__ float shared_beta;
  __shared__ float shared_norm;
  __shared__ int tile_rows[2];

  int total_queries = batch_size * num_heads * seq_len;
  bool can_vectorize = (head_dim % 4 == 0);
  int vec_elems = head_dim / 4;

  for (int q_idx = blockIdx.x; q_idx < total_queries; q_idx += gridDim.x) {
    int token_idx = q_idx % seq_len;
    int head_idx = (q_idx / seq_len) % num_heads;
    int batch_idx = q_idx / (seq_len * num_heads);

    int base_offset = (batch_idx * num_heads * seq_len * head_dim) +
                      (head_idx * seq_len * head_dim);

    const scalar_t* q_vec = Q + base_offset + (token_idx * head_dim);
    const scalar_t* k_head_ptr = K + base_offset;
    const scalar_t* v_head_ptr = V + base_offset;
    scalar_t* o_vec = O + base_offset + (token_idx * head_dim);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      shared_q[d] = static_cast<float>(q_vec[d]);
      shared_out[d] = 0.0f;
    }
    __syncthreads();

    float m_i = -INFINITY;
    float l_i = 0.0f;
    if (threadIdx.x == 0) {
      shared_norm = 1e-6f;
    }
    __syncthreads();

    int num_tiles = (seq_len + TILE_SEQ - 1) / TILE_SEQ;
    for (int stage = 0; stage < num_tiles + 1; ++stage) {
      int load_stage = stage;
      if (load_stage < num_tiles) {
        int tile_len = min(TILE_SEQ, seq_len - load_stage * TILE_SEQ);
        int buffer = load_stage & 1;
        if (threadIdx.x == 0) {
          tile_rows[buffer] = tile_len;
        }

        const scalar_t* k_ptr =
            k_head_ptr + (load_stage * TILE_SEQ) * head_dim;
        const scalar_t* v_ptr =
            v_head_ptr + (load_stage * TILE_SEQ) * head_dim;
        float* k_dest = shared_k + buffer * TILE_SEQ * head_dim;
        float* v_dest = shared_v + buffer * TILE_SEQ * head_dim;
        for (int idx = threadIdx.x; idx < tile_len * head_dim;
             idx += blockDim.x) {
          int row = idx / head_dim;
          int col = idx % head_dim;
          k_dest[row * head_dim + col] =
              static_cast<float>(k_ptr[row * head_dim + col]);
          v_dest[row * head_dim + col] =
              static_cast<float>(v_ptr[row * head_dim + col]);
        }
      }
      __syncthreads();

      int compute_stage = stage - 1;
      if (compute_stage >= 0) {
        int buffer = compute_stage & 1;
        int tile_len = tile_rows[buffer];
        const float* tile_k = shared_k + buffer * TILE_SEQ * head_dim;
        const float* tile_v = shared_v + buffer * TILE_SEQ * head_dim;

        for (int j = 0; j < tile_len; ++j) {
          const float* k_tile = tile_k + j * head_dim;
          const float* v_tile = tile_v + j * head_dim;

          float partial = 0.0f;
          if (can_vectorize) {
            for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
              float4 q4 = reinterpret_cast<const float4*>(shared_q)[i];
              float4 k4 = reinterpret_cast<const float4*>(k_tile)[i];
              partial += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z +
                         q4.w * k4.w;
            }
            int remainder = head_dim & 3;
            if (remainder && threadIdx.x < remainder) {
              int offset = vec_elems * 4 + threadIdx.x;
              partial += shared_q[offset] * k_tile[offset];
            }
          } else {
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
              partial += shared_q[d] * k_tile[d];
            }
          }

          float score = block_allreduce_sum(partial) * scale;

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

          if (can_vectorize) {
            for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
              float4 out4 = reinterpret_cast<float4*>(shared_out)[i];
              float4 v4 = reinterpret_cast<const float4*>(v_tile)[i];
              out4.x = out4.x * alpha + v4.x * beta;
              out4.y = out4.y * alpha + v4.y * beta;
              out4.z = out4.z * alpha + v4.z * beta;
              out4.w = out4.w * alpha + v4.w * beta;
              reinterpret_cast<float4*>(shared_out)[i] = out4;
            }
            int remainder = head_dim & 3;
            if (remainder && threadIdx.x < remainder) {
              int offset = vec_elems * 4 + threadIdx.x;
              shared_out[offset] =
                  shared_out[offset] * alpha + v_tile[offset] * beta;
            }
          } else {
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
              shared_out[d] = shared_out[d] * alpha + v_tile[d] * beta;
            }
          }
          __syncthreads();
        }
      }
    }

    float inv_norm = 1.0f / shared_norm;
    if (can_vectorize) {
      for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        float4 out4 = reinterpret_cast<const float4*>(shared_out)[i];
        scalar_t* dst = o_vec + i * 4;
        dst[0] = static_cast<scalar_t>(out4.x * inv_norm);
        dst[1] = static_cast<scalar_t>(out4.y * inv_norm);
        dst[2] = static_cast<scalar_t>(out4.z * inv_norm);
        dst[3] = static_cast<scalar_t>(out4.w * inv_norm);
      }
      int remainder = head_dim & 3;
      if (remainder && threadIdx.x < remainder) {
        int offset = vec_elems * 4 + threadIdx.x;
        o_vec[offset] =
            static_cast<scalar_t>(shared_out[offset] * inv_norm);
      }
    } else {
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        o_vec[d] = static_cast<scalar_t>(shared_out[d] * inv_norm);
      }
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------
// C++ / Python Interface
// ------------------------------------------------------------------------

// Required: Main function that will be called from Python
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  // 1. Setup Output Tensor
  auto O = torch::empty_like(Q);

  // 2. Extract Dimensions
  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  // 3. Configure Kernel Launch Parameters
  int total_queries = batch_size * num_heads * seq_len;
  int grid = total_queries;
  if (grid > 2048) grid = 2048;
  if (grid == 0) grid = 1;
  dim3 blocks(grid);
  dim3 threads(THREADS_PER_BLOCK);

  size_t shared_mem_size =
      (2 + 4 * TILE_SEQ) * head_dim * sizeof(float);

  // 4. Dispatch and Launch
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "flash_attention_kernel", ([&] {
        cudaFuncSetAttribute(
            flash_attention_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        flash_attention_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(), batch_size,
            num_heads, seq_len, head_dim, scale);
      }));

  // 5. Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  // 6. Synchronize to ensure kernel completion
  // (Optional for standard PyTorch usage as generic stream syncs automatically,
  // but good for strict benchmarking boundaries)
  cudaDeviceSynchronize();

  return O;
}
