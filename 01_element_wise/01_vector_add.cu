#include <cuda_runtime.h>
#include "../include/common.cuh"
#include "../include/bench.cuh"

namespace v1 {

    template <typename T, const int BLOCK_SIZE> 
    __global__ void vector_add_kernel(const T* A, const T* B, T* C, int N) {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < N) {
            C[index] = A[index] + B[index];
        }
    }

    /// @brief BLOCK_SIZE 个 thread 算 BLOCK_SIZE 个
    template <typename T, const int BLOCK_SIZE>
    void vector_add(const T* A, const T* B, T* C, int N) {
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add_kernel<T, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(A, B, C, N);
    }

} 

namespace v2 {

    constexpr int MAX_NUM_BLOCKS = 1024;

    __device__ __forceinline__ float4 float4_add(float4 a, float4 b) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    template <typename T, const int BLOCK_SIZE> 
    __global__ void vector_add_kernel(const T* A, const T* B, T* C, int N) {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        for (int i = index; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }

    /// @brief BLOCK_SIZE 个 thread 算 BLOCK_SIZE 个，可能 grid 爆出，需要 grid stride
    template <typename T, const int BLOCK_SIZE>
    void vector_add(const T* A, const T* B, T* C, int N) {
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (num_blocks > MAX_NUM_BLOCKS) {
            num_blocks = MAX_NUM_BLOCKS;
        }
        vector_add_kernel<T, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(A, B, C, N);
    }

}

namespace v3 {

    constexpr int MAX_NUM_BLOCKS = 1024;

    __device__ __forceinline__ float4 float4_add(float4 a, float4 b) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    template <const int BLOCK_SIZE> 
    __global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
        int N4 = N / 4;
        const int index = blockDim.x * blockIdx.x + threadIdx.x;
        const int grid_stride = gridDim.x * blockDim.x;

        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* B4 = reinterpret_cast<const float4*>(B);
        float4* C4 = reinterpret_cast<float4*>(C);

        for (int i = index; i < N4; i += grid_stride) {
            C4[i] = float4_add(A4[i], B4[i]);
        }

        for (int i = N4 * 4 + index; i < N; i += grid_stride) {
            C[i] = A[i] + B[i];
        }
    }

    /// @brief BLOCK_SIZE 个 thread 算 BLOCK_SIZE 个，可能 grid 爆出，需要 grid stride，同时用向量加速
    template <const int BLOCK_SIZE>
    void vector_add(const float* A, const float* B, float* C, int N) {
        int N4 = N / 4;
        int num_blocks = (N4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (num_blocks > MAX_NUM_BLOCKS) {
            num_blocks = MAX_NUM_BLOCKS;
        }
        vector_add_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(A, B, C, N);
    }

}

int main() {
    const int N = 10007;
    auto A = std::vector<float>(N);
    auto B = std::vector<float>(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = 1.0f;
    }

    auto cpu = [&](const float* const* in, float* out) {
        for (int i = 0; i < N; i++) {
            out[i] = in[0][i] + in[1][i];
        }
    };

    auto gpu1 = [&](const float* const* in, float* out) { 
        v1::vector_add<float, 256>(in[0], in[1], out, N); 
    };
    bench("add", {&A, &B}, N, cpu, gpu1);
    
    auto gpu2 = [&](const float* const* in, float* out) { 
        v2::vector_add<float, 256>(in[0], in[1], out, N); 
    };
    bench("add", {&A, &B}, N, cpu, gpu2);

    auto gpu3 = [&](const float* const* in, float* out) { 
        v3::vector_add<256>(in[0], in[1], out, N); 
    };
    bench("add", {&A, &B}, N, cpu, gpu3);
}