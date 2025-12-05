#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define NUM_BINS 256
#define NUM_CHANNELS 512
#define TILE_CHANNELS 64
#define VEC_SIZE 4
#define GRID_X 128

// Phase 1: build per-block partial histograms in shared memory, flush to global partial buffer
template <int BINS, int CHANNELS, int GRID_X_VALUE>
__global__ void histogram_partial(
    const uint8_t* __restrict__ data,
    int* __restrict__ partial,
    int length
) {
    extern __shared__ int s_hist[];
    constexpr int s_stride = BINS + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int total_smem_size = TILE_CHANNELS * s_stride;

    for (int i = tid; i < total_smem_size; i += blockDim.x * blockDim.y) {
        s_hist[i] = 0;
    }
    __syncthreads();

    int vec_c_offset = threadIdx.x * VEC_SIZE;
    int* hist_ptr0 = &s_hist[(vec_c_offset + 0) * s_stride];
    int* hist_ptr1 = &s_hist[(vec_c_offset + 1) * s_stride];
    int* hist_ptr2 = &s_hist[(vec_c_offset + 2) * s_stride];
    int* hist_ptr3 = &s_hist[(vec_c_offset + 3) * s_stride];

    const uint32_t* input_vec = reinterpret_cast<const uint32_t*>(data);
    constexpr int VEC_STRIDE = CHANNELS / VEC_SIZE;

    int block_c_start = blockIdx.y * TILE_CHANNELS;
    int vec_col = (block_c_start / VEC_SIZE) + threadIdx.x;
    int step_y = gridDim.x * blockDim.y;
    int start_row = blockIdx.x * blockDim.y + threadIdx.y;
    int r = start_row;
    int limit = length - (step_y * 3);

    for (; r < limit; r += step_y * 4) {
        size_t idx0 = (size_t)(r + step_y * 0) * VEC_STRIDE + vec_col;
        size_t idx1 = (size_t)(r + step_y * 1) * VEC_STRIDE + vec_col;
        size_t idx2 = (size_t)(r + step_y * 2) * VEC_STRIDE + vec_col;
        size_t idx3 = (size_t)(r + step_y * 3) * VEC_STRIDE + vec_col;
        uint32_t p0 = input_vec[idx0];
        uint32_t p1 = input_vec[idx1];
        uint32_t p2 = input_vec[idx2];
        uint32_t p3 = input_vec[idx3];
        atomicAdd(hist_ptr0 + (p0 & 0xFF), 1);
        atomicAdd(hist_ptr1 + ((p0 >> 8) & 0xFF), 1);
        atomicAdd(hist_ptr2 + ((p0 >> 16) & 0xFF), 1);
        atomicAdd(hist_ptr3 + ((p0 >> 24) & 0xFF), 1);

        atomicAdd(hist_ptr0 + (p1 & 0xFF), 1);
        atomicAdd(hist_ptr1 + ((p1 >> 8) & 0xFF), 1);
        atomicAdd(hist_ptr2 + ((p1 >> 16) & 0xFF), 1);
        atomicAdd(hist_ptr3 + ((p1 >> 24) & 0xFF), 1);

        atomicAdd(hist_ptr0 + (p2 & 0xFF), 1);
        atomicAdd(hist_ptr1 + ((p2 >> 8) & 0xFF), 1);
        atomicAdd(hist_ptr2 + ((p2 >> 16) & 0xFF), 1);
        atomicAdd(hist_ptr3 + ((p2 >> 24) & 0xFF), 1);

        atomicAdd(hist_ptr0 + (p3 & 0xFF), 1);
        atomicAdd(hist_ptr1 + ((p3 >> 8) & 0xFF), 1);
        atomicAdd(hist_ptr2 + ((p3 >> 16) & 0xFF), 1);
        atomicAdd(hist_ptr3 + ((p3 >> 24) & 0xFF), 1);
    }

    for (; r < length; r += step_y) {
        size_t vec_idx = (size_t)r * VEC_STRIDE + vec_col;
        uint32_t p = input_vec[vec_idx];
        atomicAdd(hist_ptr0 + (p & 0xFF), 1);
        atomicAdd(hist_ptr1 + ((p >> 8) & 0xFF), 1);
        atomicAdd(hist_ptr2 + ((p >> 16) & 0xFF), 1);
        atomicAdd(hist_ptr3 + ((p >> 24) & 0xFF), 1);
    }
    __syncthreads();

    for (int i = tid; i < TILE_CHANNELS * BINS; i += blockDim.x * blockDim.y) {
        int local_c = i / BINS;
        int bin = i % BINS;
        int count = s_hist[local_c * s_stride + bin];
        int global_c = block_c_start + local_c;
        size_t out_idx = ((size_t)blockIdx.x * CHANNELS + global_c) * BINS + bin;
        partial[out_idx] = count;
    }
}

// Phase 2: reduce partial histograms across blockIdx.x dimension
template <int BINS, int CHANNELS, int GRID_X_VALUE>
__global__ void reduce_kernel(
    const int* __restrict__ partial,
    int* __restrict__ hist
) {
    int c = blockIdx.x;
    int bin = threadIdx.x;
    if (bin >= BINS || c >= CHANNELS) return;

    int sum = 0;
    size_t base = ((size_t)0 * CHANNELS + c) * BINS + bin;
    size_t stride = (size_t)CHANNELS * BINS;
#pragma unroll
    for (int bx = 0; bx < GRID_X_VALUE; ++bx) {
        sum += partial[base + bx * stride];
    }

    hist[(size_t)c * BINS + bin] = sum;
}

torch::Tensor histogram_kernel(
    torch::Tensor data,
    int num_bins
) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");
    TORCH_CHECK(data.scalar_type() == torch::kUInt8, "Tensor data must be uint8");
    TORCH_CHECK(num_bins == NUM_BINS, "num_bins must be 256");
    TORCH_CHECK(data.size(1) == NUM_CHANNELS, "num_channels must be 512");

    if (!data.is_contiguous()) {
        data = data.contiguous();
    }

    static torch::Tensor cached_output;
    static torch::Tensor partial_hist;
    static int* d_out_ptr = nullptr;
    static int* d_partial_ptr = nullptr;
    static dim3 grid;
    static dim3 block;
    static int smem_bytes;
    static bool is_initialized = false;
    void* d_in_ptr = data.data_ptr();
    int length = data.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (!is_initialized) {
        auto options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(data.device());

        cached_output = torch::empty({NUM_CHANNELS, NUM_BINS}, options);
        partial_hist = torch::empty({GRID_X, NUM_CHANNELS, NUM_BINS}, options);
        d_out_ptr = (int*)cached_output.data_ptr();
        d_partial_ptr = (int*)partial_hist.data_ptr();

        int threads_x = TILE_CHANNELS / VEC_SIZE;
        int threads_y = 32;
        block = dim3(threads_x, threads_y);
        grid = dim3(GRID_X, NUM_CHANNELS / TILE_CHANNELS);
        smem_bytes = TILE_CHANNELS * (NUM_BINS + 1) * sizeof(int);

        cudaFuncSetAttribute(
            histogram_partial<NUM_BINS, NUM_CHANNELS, GRID_X>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        is_initialized = true;
    }

    // Phase 1: build partial histograms
    histogram_partial<NUM_BINS, NUM_CHANNELS, GRID_X><<<grid, block, smem_bytes, stream>>>(
        (uint8_t*)d_in_ptr,
        d_partial_ptr,
        length
    );

    // Phase 2: reduce to final histogram
    dim3 block_reduce(NUM_BINS);
    dim3 grid_reduce(NUM_CHANNELS);
    reduce_kernel<NUM_BINS, NUM_CHANNELS, GRID_X><<<grid_reduce, block_reduce, 0, stream>>>(
        d_partial_ptr,
        d_out_ptr
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return cached_output;
}
