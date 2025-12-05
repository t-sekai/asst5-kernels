// Progressive Improvement Step 3: Vectorized math in shared memory.
// Builds on Step 2 (shared-memory tiling) and adds float4 math on the SRAM
// resident tensors so that dot products and output updates process four
// elements per instruction.

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

__device__ inline float4 madd4(const float4& a, const float4& b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ inline float sum_float4(const float4& v) {
  return v.x + v.y + v.z + v.w;
}

template <typename scalar_t>
__global__ void vectorized_flash_attention_kernel(
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

  extern __shared__ __align__(16) float shared_mem[];
  float* shared_out = shared_mem;
  float* shared_q = shared_out + head_dim;
  float* shared_k = shared_q + head_dim;
  float* shared_v = shared_k + TILE_SEQ * head_dim;

  // Load Q row and initialise accumulator.
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    shared_q[d] = static_cast<float>(q_vec[d]);
    shared_out[d] = 0.0f;
  }
  __syncthreads();

  bool can_vectorize = (head_dim % 4 == 0);
  int vec_elems = head_dim / 4;

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
      if (can_vectorize) {
        for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
          float4 q_vec4 =
              reinterpret_cast<const float4*>(shared_q)[i];
          float4 k_vec4 =
              reinterpret_cast<const float4*>(tile_k)[i];
          float4 prod = madd4(q_vec4, k_vec4);
          partial += sum_float4(prod);
        }
        int remainder = head_dim & 3;
        if (remainder && threadIdx.x < remainder) {
          int offset = vec_elems * 4 + threadIdx.x;
          partial += shared_q[offset] * tile_k[offset];
        }
      } else {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
          partial += shared_q[d] * tile_k[d];
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
          float4 out_vec = reinterpret_cast<float4*>(shared_out)[i];
          float4 v_vec4 = reinterpret_cast<const float4*>(tile_v)[i];
          out_vec.x = out_vec.x * alpha + v_vec4.x * beta;
          out_vec.y = out_vec.y * alpha + v_vec4.y * beta;
          out_vec.z = out_vec.z * alpha + v_vec4.z * beta;
          out_vec.w = out_vec.w * alpha + v_vec4.w * beta;
          reinterpret_cast<float4*>(shared_out)[i] = out_vec;
        }
        int remainder = head_dim & 3;
        if (remainder && threadIdx.x < remainder) {
          int offset = vec_elems * 4 + threadIdx.x;
          shared_out[offset] = shared_out[offset] * alpha +
                               tile_v[offset] * beta;
        }
      } else {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
          shared_out[d] = shared_out[d] * alpha + tile_v[d] * beta;
        }
      }
      __syncthreads();
    }
  }

  float norm = shared_norm + 1e-6f;
  float inv_norm = 1.0f / norm;

  if (can_vectorize) {
    for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) {
      float4 out_vec = reinterpret_cast<const float4*>(shared_out)[i];
      scalar_t* dst = o_vec + i * 4;
      dst[0] = static_cast<scalar_t>(out_vec.x * inv_norm);
      dst[1] = static_cast<scalar_t>(out_vec.y * inv_norm);
      dst[2] = static_cast<scalar_t>(out_vec.z * inv_norm);
      dst[3] = static_cast<scalar_t>(out_vec.w * inv_norm);
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
}

torch::Tensor flash_attention_forward_step3(torch::Tensor Q, torch::Tensor K,
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
      (2 + 2 * TILE_SEQ) * head_dim * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "vectorized_flash_attention_kernel", ([&] {
        vectorized_flash_attention_kernel<scalar_t>
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
