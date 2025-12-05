// Progressive Improvement Step 2: Shared-Memory Tiling of K/V
// Builds on Step 1 by staging blocks of K/V rows into shared memory so that
// each query touches HBM once per tile instead of once per score.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#define THREADS_PER_ROW 128
#define TILE_SEQ 32

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ float block_allreduce_sum(float val) {
  __shared__ float shared[THREADS_PER_ROW / warpSize];
  float sum = warp_reduce_sum(val);
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  if (lane == 0) {
    shared[warp_id] = sum;
  }
  __syncthreads();

  sum = (lane < (THREADS_PER_ROW / warpSize)) ? shared[lane] : 0.0f;
  sum = warp_reduce_sum(sum);

  __shared__ float total;
  if (threadIdx.x == 0) {
    total = sum;
  }
  __syncthreads();
  return total;
}

template <typename scalar_t>
__global__ void shared_tile_flash_attention_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {
  int q_idx = blockIdx.x;
  if (q_idx >= batch_size * num_heads * seq_len) {
    return;
  }

  int token_idx = q_idx % seq_len;
  int head_idx = (q_idx / seq_len) % num_heads;
  int batch_idx = q_idx / (seq_len * num_heads);

  int base_offset = (batch_idx * num_heads * seq_len * head_dim) +
                    (head_idx * seq_len * head_dim);

  const scalar_t* q_vec = Q + base_offset + (token_idx * head_dim);
  const scalar_t* k_head_ptr = K + base_offset;
  const scalar_t* v_head_ptr = V + base_offset;
  scalar_t* o_vec = O + base_offset + (token_idx * head_dim);

  extern __shared__ float shared_mem[];
  float* shared_out = shared_mem;  // head_dim
  float* shared_q = shared_out + head_dim;  // head_dim
  float* shared_k = shared_q + head_dim;    // TILE_SEQ * head_dim
  float* shared_v = shared_k + TILE_SEQ * head_dim;  // TILE_SEQ * head_dim

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    shared_q[d] = static_cast<float>(q_vec[d]);
    shared_out[d] = 0.0f;
  }
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.0f;

  __shared__ float shared_alpha;
  __shared__ float shared_beta;
  __shared__ float shared_norm;
  if (threadIdx.x == 0) {
    shared_norm = 1e-6f;
  }
  __syncthreads();

  for (int j_block = 0; j_block < seq_len; j_block += TILE_SEQ) {
    int tile_len = min(TILE_SEQ, seq_len - j_block);

    for (int idx = threadIdx.x; idx < tile_len * head_dim; idx += blockDim.x) {
      int row = idx / head_dim;
      int col = idx % head_dim;
      int seq_index = j_block + row;
      const scalar_t* k_vec = k_head_ptr + seq_index * head_dim;
      const scalar_t* v_vec = v_head_ptr + seq_index * head_dim;
      shared_k[row * head_dim + col] = static_cast<float>(k_vec[col]);
      shared_v[row * head_dim + col] = static_cast<float>(v_vec[col]);
    }
    __syncthreads();

    for (int j = 0; j < tile_len; ++j) {
      const float* tile_k = shared_k + j * head_dim;
      const float* tile_v = shared_v + j * head_dim;

      float partial = 0.0f;
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += shared_q[d] * tile_k[d];
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

      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        shared_out[d] = shared_out[d] * alpha + tile_v[d] * beta;
      }
      __syncthreads();
    }
  }

  float norm = shared_norm + 1e-6f;
  float inv_norm = 1.0f / norm;

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    o_vec[d] = static_cast<scalar_t>(shared_out[d] * inv_norm);
  }
}

torch::Tensor flash_attention_forward_step2(torch::Tensor Q, torch::Tensor K,
                                            torch::Tensor V) {
  auto O = torch::empty_like(Q);

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  int total_queries = batch_size * num_heads * seq_len;
  dim3 blocks(total_queries);
  dim3 threads(THREADS_PER_ROW);

  size_t shared_mem_size =
      (2 + 2 * TILE_SEQ) * head_dim * sizeof(float);  // q, out, k tile, v tile

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "shared_tile_flash_attention_kernel", ([&] {
        shared_tile_flash_attention_kernel<scalar_t>
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
