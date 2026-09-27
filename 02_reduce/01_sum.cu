/*
    1. 记得加同步
    2. __shfl_down_sync 和 __shfl_xor_sync
*/
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "../include/common.cuh"
#include "../include/bench.cuh"

namespace v1 {

    template <const int BLOCK_SIZE> 
    __global__ void sum_kernel(const float* input, float* output, int N) {
        const int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < N) {
            atomicAdd(output, input[4]);
        }
    }

    /// @brief 历史最差情况，所有 thread 用 atomic add 累加全局  
    template <const int BLOCK_SIZE> 
    void sum(const float* input, float* output, int N) {
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sum_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(input, output, N);
    }

}

namespace v2 {

    template <const int BLOCK_SIZE> 
    __global__ void sum_kernel(const float* input, float* output, int N) {
        const int tid = threadIdx.x;
        const int index = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ float mem[BLOCK_SIZE];
        mem[tid] = index < N ? input[index] : 0.0;
        __syncthreads(); 

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                mem[tid] += mem[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(output, mem[0]);
        }
    }

    /// @brief 历史最差情况，Block 内部归，Block atomic add
    template <const int BLOCK_SIZE> 
    void sum(const float* input, float* output, int N) {
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sum_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(input, output, N);
    }

}

namespace v3 {

    __device__ __forceinline__ float warp_reduce_sum(float val) {
        for (int stride = 16; stride > 0; stride >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, stride);
        }
        return val;
    }

    template <const int BLOCK_SIZE> 
    __global__ void sum_kernel(const float* input, float* output, int N) {
        constexpr int WARP_COUNT = BLOCK_SIZE / 32;
        static_assert(BLOCK_SIZE % 32 == 0);

        const int tid = threadIdx.x;
        const int index = blockDim.x * blockIdx.x + threadIdx.x;
        const int lane = tid & 31;
        const int warp_idx = tid >> 5;

        float val = index < N ? input[index] : 0.0;

        // 1. 先一共划分为 WARP_COUNT 组，进行规约。
        val = warp_reduce_sum(val);

        // 2. 对每个 warp 中 index 为 0 的 thread 
        __shared__ float mem[WARP_COUNT];
        if (lane == 0) {
            mem[warp_idx] = val;
        } 
        __syncthreads();

        // 3. 让 warp 0 归约
        if (warp_idx == 0) {
            float val = 0.0f;
            // 将 mem 搬运到 warp 0 中的 32 个 thread
            for (int i = lane; i < WARP_COUNT; i += 32) {
                val += mem[i];
            }
            val = warp_reduce_sum(val);
            if (lane == 0) {
                atomicAdd(output, val);
            }
        }
    }

    /// @brief 历史最差情况，Block 内部归，Block atomic add，shfl 归约
    template <const int BLOCK_SIZE> 
    void sum(const float* input, float* output, int N) {
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sum_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(input, output, N);
    }

}

namespace v4 {

    constexpr int MAX_NUM_BLOCKS = 1024;

    __device__ __forceinline__ float warp_reduce_sum(float val) {
        for (int stride = 16; stride > 0; stride >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, stride);
        }
        return val;
    }

    __device__ __forceinline__ float4 float4_add(float4 a, float4 b) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    template <const int BLOCK_SIZE> 
    __global__ void sum_kernel(const float* input, float* output, int N) {
        constexpr int WARP_COUNT = BLOCK_SIZE / 32;
        static_assert(BLOCK_SIZE % 32 == 0);
        int N4 = N / 4;

        const int tid = threadIdx.x;
        const int grid_stride = blockDim.x * gridDim.x;
        const int lane = tid & 31;
        const int warp_idx = tid >> 5;

        // 0. 先 float4 规约下！
        auto local4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4* I4 = reinterpret_cast<const float4*>(input);
        for (int i = tid ; i < N4; i += grid_stride) {
            local4 = float4_add(local4, I4[i]);
        }
        auto local = 0.0f;
        for (int i = N4 * 4 + tid; i < N; i += grid_stride) {
            local += input[i];
        }
        local += (local4.x + local4.y + local4.z + local4.w);

        // 1. 先一共划分为 WARP_COUNT 组，进行规约。
        local = warp_reduce_sum(local);

        // 2. 对每个 warp 中 index 为 0 的 thread 
        __shared__ float mem[WARP_COUNT];
        if (lane == 0) {
            mem[warp_idx] = local;
        } 
        __syncthreads();

        // 3. 让 warp 0 归约
        if (warp_idx == 0) {
            float val = 0.0f;
            // 将 mem 搬运到 warp 0 中的 32 个 thread
            for (int i = lane; i < WARP_COUNT; i += 32) {
                val += mem[i];
            }
            val = warp_reduce_sum(val);
            if (lane == 0) {
                atomicAdd(output, val);
            }
        }
    }

    /// @brief 历史最差情况，Block 内部归，Block atomic add，shfl 归约，向量加速
    template <const int BLOCK_SIZE> 
    void sum(const float* input, float* output, int N) {
        int N4 = N / 4;
        int num_blocks = (N4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (num_blocks > MAX_NUM_BLOCKS) {
            num_blocks = MAX_NUM_BLOCKS;
        }
        sum_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(input, output, N);
    }

}

namespace test {

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

using SumFn = void(*)(const float*, float*, int);

struct TestConfig {
    int   N      = 1 << 20;   // 元素个数
    int   warmup = 3;         // warm-up 次数
    int   iters  = 20;        // 计时次数
    float rtol   = 1e-2f;     // 相对误差阈值
    bool  check  = true;      // 是否校验
};

struct TestResult {
    float gpu_val = 0.0f;
    float cpu_val = 0.0f;
    float rel_err = 0.0f;
    float ms      = 0.0f;
    bool  passed  = false;
};

static float cpu_sum(const std::vector<float>& data) {
    double acc = 0.0;
    for (float v : data) acc += v;
    return static_cast<float>(acc);
}

TestResult run_test(const std::string& name, SumFn fn, const TestConfig& cfg = TestConfig{}) {
    const int N = cfg.N;
    printf("\n===== [%s] N = %d =====\n", name.c_str(), N);

    // 1. host 数据（全 1，期望结果 = N）
    std::vector<float> h_input(N, 1.0f);

    // 2. device 内存
    float* d_input  = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));

    // 3. warm-up
    for (int i = 0; i < cfg.warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        fn(d_input, d_output, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // 4. 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < cfg.iters; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_output, 0, sizeof(float)));  // ★ 必须清零
        fn(d_input, d_output, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    TestResult r;
    CUDA_CHECK(cudaEventElapsedTime(&r.ms, start, stop));
    r.ms /= cfg.iters;

    // 5. 拷回 & 校验
    CUDA_CHECK(cudaMemcpy(&r.gpu_val, d_output,
                          sizeof(float), cudaMemcpyDeviceToHost));
    r.cpu_val = cpu_sum(h_input);

    float denom = std::max(1.0f, std::fabs(r.cpu_val));
    r.rel_err   = std::fabs(r.gpu_val - r.cpu_val) / denom;
    r.passed    = (!cfg.check) || (r.rel_err < cfg.rtol);

    // 6. 打印
    printf("  time   : %.4f ms\n", r.ms);
    printf("  GPU    : %.6f\n", r.gpu_val);
    printf("  CPU    : %.6f\n", r.cpu_val);
    printf("  rel_err: %.3e  (tol=%.1e)\n", r.rel_err, cfg.rtol);
    printf("  result : %s\n", r.passed ? "PASS" : "FAIL");

    // 7. 释放
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return r;
}

}

int main(int argc, char** argv) {
    test::TestConfig cfg;
    cfg.N = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);

    bool all_pass = true;

    test::run_test("v1", v1::sum<256>, cfg);
    test::run_test("v2", v2::sum<256>, cfg);
    test::run_test("v3", v3::sum<256>, cfg);
    test::run_test("v4", v4::sum<256>, cfg);

    return all_pass ? 0 : 1;
}