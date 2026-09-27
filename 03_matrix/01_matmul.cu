
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <functional>
#include <cmath>
#include <stdexcept>

namespace v1 {

    template <const int BLOCK_M, const int BLOCK_N>
    __global__ void matmul_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
        const int row = blockDim.y * blockIdx.y + threadIdx.y;
        const int col = blockDim.x * blockIdx.x + threadIdx.x;
        if (row < M && col < N) {
            float temp = 0.0f;
            for (int i = 0; i < K; ++i) {
                temp += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = temp;
        }
    }

    /// @brief 每个 thread 计算一个输出，历史最差性能
    template <const int BLOCK_M, const int BLOCK_N>
    void matmul(const float* A, const float* B, float* C, const int M, const int K, const int N) {
        unsigned int gx = (N + BLOCK_N - 1) / BLOCK_N;
        unsigned int gy = (M + BLOCK_M - 1) / BLOCK_M;
        auto grid_dims = dim3{ gx, gy };
        auto block_dims = dim3{ BLOCK_N, BLOCK_M };
        matmul_kernel<BLOCK_M, BLOCK_N><<<grid_dims, block_dims>>>(A, B, C, M, N, K);
    }

}

namespace v2 {

    template <const int BLOCK_M, const int BLOCK_N, const int TILE_K>
    __global__ void matmul_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
        constexpr int NUM_THREAD = BLOCK_M * BLOCK_N;

        const int tid = threadIdx.y * BLOCK_N + threadIdx.x;
        
        const int row = blockDim.y * blockIdx.y + threadIdx.y;
        const int col = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ float A_cache[BLOCK_M][TILE_K];
        __shared__ float B_cache[TILE_K][BLOCK_N];

        int num_tiles = (K + TILE_K - 1) / TILE_K;
        float acc = 0.0f;
        for (int k = 0; k < num_tiles; ++k) {
            const int k_base = k * TILE_K;
            // 加载 A/B 到 cache
            for (int i = tid; i < BLOCK_M * TILE_K; i += NUM_THREAD) {
                int row = i / TILE_K;
                int col = i % TILE_K;
                int g_row = blockDim.y * blockIdx.y + row;
                int g_col = k_base + col;
                A_cache[row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
            }
            for (int i = tid; i < TILE_K * BLOCK_N; i += NUM_THREAD) {
                int row = i / BLOCK_N;
                int col = i % BLOCK_N;
                int g_row = k_base + row;
                int g_col = blockDim.x * blockIdx.x + col;
                B_cache[row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
            }
            __syncthreads();

            // 计算 block 的矩阵乘法
            for (int i = 0; i < TILE_K; ++i) {
                acc += A_cache[threadIdx.y][i] * B_cache[i][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            C[row * N + col] = acc;
        }
    }

    /// @brief 每个 thread 计算一个输出，利用 share memory 一个 block 计算
    template <const int BLOCK_M, const int BLOCK_N, const int TILE_K>
    void matmul(const float* A, const float* B, float* C, const int M, const int N, const int K) {
        unsigned int gx = (N + BLOCK_N - 1) / BLOCK_N;
        unsigned int gy = (M + BLOCK_M - 1) / BLOCK_M;
        auto grid_dims = dim3{ gx, gy };
        auto block_dims = dim3{ BLOCK_N, BLOCK_M };
        matmul_kernel<BLOCK_M, BLOCK_N, TILE_K><<<grid_dims, block_dims>>>(A, B, C, M, N, K);
    }

}

namespace v3 {

    constexpr int THREAD_M = 4;
    constexpr int THREAD_N = 4;

    template <const int BLOCK_M, const int BLOCK_N, const int TILE_K>
    __global__ void matmul_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
        constexpr int NUM_THREAD_M = BLOCK_M / THREAD_M;
        constexpr int NUM_THREAD_N = BLOCK_N / THREAD_N;
        constexpr int NUM_THREAD = NUM_THREAD_M * NUM_THREAD_N;

        // 给每个 thread 负责那块输出的全局 index
        // 注意用的是 BLOCK_M
        const int row_local = BLOCK_M * blockIdx.y + threadIdx.y * THREAD_M;
        const int col_local = BLOCK_N * blockIdx.x + threadIdx.x * THREAD_N;

        const int tid = threadIdx.y * NUM_THREAD_N + threadIdx.x;

        __shared__ float A_cache[BLOCK_M][TILE_K];
        __shared__ float B_cache[TILE_K][BLOCK_N];

        float acc[THREAD_M][THREAD_N] = {0.0f};
        float A_loacl[THREAD_M] = {0.0f};
        float B_loacl[THREAD_N] = {0.0f};

        int num_tiles = (K + TILE_K - 1) / TILE_K;
        for (int k = 0; k < num_tiles; ++k) {
            const int k_base = k * TILE_K;
            // 加载 A/B 到 cache
            for (int i = tid; i < BLOCK_M * TILE_K; i += NUM_THREAD) {
                int row = i / TILE_K;
                int col = i % TILE_K;
                // 注意用的是 BLOCK_M
                int g_row = BLOCK_M * blockIdx.y + row;
                int g_col = k_base + col;
                A_cache[row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
            }
            for (int i = tid; i < TILE_K * BLOCK_N; i += NUM_THREAD) {
                int row = i / BLOCK_N;
                int col = i % BLOCK_N;
                int g_row = k_base + row;
                int g_col = BLOCK_N * blockIdx.x + col;
                B_cache[row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
            }
            __syncthreads();

            // 计算 block 的矩阵乘法
            for (int i = 0; i < TILE_K; ++i) {
                // 加载 A_cache 的一列，加载 B_cache 的一行
                // cache 保证不会 out of index
                for (int j = 0; j < THREAD_M; ++j) {
                    A_loacl[j] = A_cache[threadIdx.y * THREAD_M + j][i];
                }
                for (int j = 0; j < THREAD_N; ++j) {
                    B_loacl[j] = B_cache[i][threadIdx.x * THREAD_N + j];
                }
                // 计算外积
                for (int r = 0; r < THREAD_M; ++r) {
                    for (int c = 0; c < THREAD_N; ++c) {
                        acc[r][c] += A_loacl[r] * B_loacl[c];
                    }
                }
            }
            __syncthreads();
        }

        for (int r = 0; r < THREAD_M; ++r) {
            for (int c = 0; c < THREAD_N; ++c) {
                int g_r = row_local + r;
                int g_c = col_local + c;
                if (g_r < M && g_c < N) {
                    C[g_r * N + g_c] = acc[r][c];
                }
            }
        }
    }

    /// @brief 每个 thread 计算多个输出，利用 share memory 一个 block 计算
    template <const int BLOCK_M, const int BLOCK_N, const int TILE_K>
    void matmul(const float* A, const float* B, float* C, const int M, const int N, const int K) {
        constexpr int NUM_THREAD_M = BLOCK_M / THREAD_M;
        constexpr int NUM_THREAD_N = BLOCK_N / THREAD_N;

        // BLOCK 的大小还是规定一个 BLOCK 需要计算的量
        unsigned int gx = ((N + BLOCK_N - 1) / BLOCK_N); 
        unsigned int gy = ((M + BLOCK_M - 1) / BLOCK_M);
        auto grid_dims = dim3{ gx, gy };
        auto block_dims = dim3{ NUM_THREAD_N, NUM_THREAD_M };
        matmul_kernel<BLOCK_M, BLOCK_N, TILE_K><<<grid_dims, block_dims>>>(A, B, C, M, N, K);
    }

}

namespace v4 {

    template <const int BLOCK_M, const int BLOCK_N, const int TILE_K>
    __global__ void matmul_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K) {
        constexpr int NUM_THREADS = BLOCK_M * BLOCK_N;
        const int tid_x = threadIdx.x;
        const int tid_y = threadIdx.y;
        const int tid = tid_y * BLOCK_N + tid_x;

        const int row_block = blockDim.y * blockIdx.y;
        const int col_block = blockDim.x * blockIdx.x;
        const int row_thread = row_block + tid_y;
        const int col_thread = col_block + tid_x;

        __shared__ float A_cache[2][BLOCK_M][TILE_K];
        __shared__ float B_cache[2][TILE_K][BLOCK_N];

        float acc = 0.0f;
        int num_tiles = (K + TILE_K - 1) / TILE_K;

        auto load_tile = [&](int k, int buf) {
            const int k_base = k * TILE_K;
            for (int i = tid; i < BLOCK_M * TILE_K; i += NUM_THREADS) {
                int row = i / TILE_K;
                int col = i % TILE_K;
                int row_g = row_block + row;
                int col_g = k_base + col;
                A_cache[buf][row][col] = (row_g < M && col_g < K) ? A[row_g * K + col_g] : 0.0f;
            }
            for (int i = tid; i < TILE_K * BLOCK_N; i += NUM_THREADS) {
                int row = i / BLOCK_N;
                int col = i % BLOCK_N;
                int row_g = k_base + row;
                int col_g = col_block + col;
                B_cache[buf][row][col] = (row_g < K && col_g < N) ? B[row_g * N + col_g] : 0.0f;
            }
        };

        load_tile(0, 0);
        __syncthreads();

        for (int k = 0; k < num_tiles; ++k) {
            const int cur = k & 1;
            const int next = cur ^ 1;

            if (k + 1 < num_tiles) {
                load_tile(k + 1, next);
            }

            for (int i = 0; i < TILE_K; ++i) {
                acc += A_cache[cur][tid_y][i] * B_cache[cur][i][tid_x];
            }
            
            __syncthreads();
        }

        if (row_thread < M && col_thread < N) {
            C[row_thread * N + col_thread] = acc;
        }
    }

    // 一个 thread 计算一个输出，但双 buffer
    template <const int BLOCK_M, const int BLOCK_N, const int TILE_K>
    void matmul(const float* A, const float* B, float* C, const int M, const int N, const int K) {
        unsigned int gx = (N + BLOCK_N - 1) / BLOCK_N;
        unsigned int gy = (M + BLOCK_M - 1) / BLOCK_M;
        auto grid_dims = dim3{ gx, gy };
        auto block_dims = dim3{ BLOCK_N, BLOCK_M };
        matmul_kernel<BLOCK_M, BLOCK_N, TILE_K><<<grid_dims, block_dims>>>(A, B, C, M, N, K);
    }

}

namespace test {

// ---------------- 内部工具 ----------------

namespace detail {

#define TEST_CUDA_CHECK(call)                                                 \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                cudaGetErrorString(err) + " at " + __FILE__ + ":" +           \
                std::to_string(__LINE__));                                    \
        }                                                                     \
    } while (0)

inline void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// 标准 CPU 实现：C = A * B
// A: M x K, B: K x N, C: M x N  (row-major)
inline void matmul_cpu(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

using MatmulFn = std::function<void(const float*, const float*, float*,
                                    int, int, int)>;

struct TestResult {
    bool  passed = false;
    int   M = 0, N = 0, K = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    double elapsed_ms = 0.0;
    double gflops = 0.0;
};

// 单次测试：A,B 行主序，C = A(MxK) * B(KxN)
inline TestResult run_single(const MatmulFn& fn,
                             int M, int N, int K,
                             unsigned seed) {
    TestResult r;
    r.M = M; r.N = N; r.K = K;

    std::vector<float> hA(M * K), hB(K * N);
    std::vector<float> hC(M * N, 0.0f), hRef(M * N, 0.0f);
    fill_random(hA, seed);
    fill_random(hB, seed + 1);

    // CPU 参考
    matmul_cpu(hA.data(), hB.data(), hRef.data(), M, N, K);

    // GPU 侧
    size_t bytesA = sizeof(float) * M * K;
    size_t bytesB = sizeof(float) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    float *dA, *dB, *dC;
    TEST_CUDA_CHECK(cudaMalloc(&dA, bytesA));
    TEST_CUDA_CHECK(cudaMalloc(&dB, bytesB));
    TEST_CUDA_CHECK(cudaMalloc(&dC, bytesC));
    TEST_CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    // warmup
    fn(dA, dB, dC, M, N, K);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    TEST_CUDA_CHECK(cudaEventCreate(&start));
    TEST_CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 20;
    TEST_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn(dA, dB, dC, M, N, K);
    TEST_CUDA_CHECK(cudaEventRecord(stop));
    TEST_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    r.elapsed_ms = ms / iters;
    r.gflops = 2.0 * M * N * K / (r.elapsed_ms * 1e-3) / 1e9;

    // 取回结果
    TEST_CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // 误差
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float a = std::fabs(hC[i] - hRef[i]);
        float r_ = a / (std::fabs(hRef[i]) + 1e-6f);
        if (a > max_abs) max_abs = a;
        if (r_ > max_rel) max_rel = r_;
    }
    r.max_abs_err = max_abs;
    r.max_rel_err = max_rel;
    r.passed = (max_abs < 1e-3f) || (max_rel < 1e-3f);

    TEST_CUDA_CHECK(cudaFree(dA));
    TEST_CUDA_CHECK(cudaFree(dB));
    TEST_CUDA_CHECK(cudaFree(dC));
    TEST_CUDA_CHECK(cudaEventDestroy(start));
    TEST_CUDA_CHECK(cudaEventDestroy(stop));
    return r;
}

inline void print_result(const TestResult& r, bool ok) {
    std::cout << "  [" << std::setw(5) << r.M << " x " << std::setw(5) << r.N
              << " x " << std::setw(5) << r.K << "]  "
              << (ok ? "PASS" : "FAIL")
              << "  time=" << std::fixed << std::setprecision(3)
              << std::setw(9) << r.elapsed_ms << " ms"
              << "  " << std::setprecision(1) << std::setw(8) << r.gflops
              << " GFLOPS"
              << "  max_abs=" << std::scientific << std::setprecision(2)
              << r.max_abs_err
              << "  max_rel=" << r.max_rel_err
              << std::defaultfloat << "\n";
}

inline void run_case(const MatmulFn& fn, int M, int N, int K, unsigned seed, int& passed, int& total) {
    ++total;
    try {
        TestResult r = run_single(fn, M, N, K, seed);
        if (r.passed) ++passed;
        print_result(r, r.passed);
    } catch (const std::exception& e) {
        std::cout << "  [" << M << " x " << N << " x " << K
                  << "]  ERROR: " << e.what() << "\n";
    }
}

} // namespace detail

// ---------------- 对外唯一接口 ----------------

inline int run_all(const detail::MatmulFn& fn) {
    using namespace detail;

    int passed = 0, total = 0;
    unsigned seed = 42;

    std::cout << "========================================\n";
    std::cout << "        CUDA Matmul Test Suite\n";
    std::cout << "========================================\n";

    std::cout << "\n[1] Square sizes (2^n)\n";
    for (int s : {32, 64, 128, 256})
        run_case(fn, s, s, s, seed, passed, total);

    std::cout << "\n[2] Non-square / skinny shapes\n";
    for (auto [M, N, K] : std::vector<std::tuple<int,int,int>>{
            {1, 1, 1}, {1, 128, 128}, {128, 1, 128}, {128, 128, 1},
            {64, 128, 32}, {128, 64, 256}, {256, 256, 32}})
        run_case(fn, M, N, K, seed, passed, total);

    std::cout << "\n[3] Odd / non power-of-2 sizes\n";
    for (int s : {17, 33, 100, 255, 333})
        run_case(fn, s, s, s, seed, passed, total);

    std::cout << "\n[4] Tiles boundary (block size +/- 1)\n";
    for (int s : {15, 16, 17, 31, 32, 33, 63, 64, 65})
        run_case(fn, s, s, s, seed, passed, total);

    std::cout << "\n[5] Large K\n";
    for (auto [M, N, K] : std::vector<std::tuple<int,int,int>>{
            {64, 64, 1024}, {128, 128, 2048}, {256, 256, 1024}})
        run_case(fn, M, N, K, seed, passed, total);

    std::cout << "\n[6] Random seeds consistency (128x128x128)\n";
    for (unsigned s : {1u, 7u, 123u, 9999u})
        run_case(fn, 128, 128, 128, s, passed, total);

    std::cout << "\n========================================\n";
    std::cout << "Result: " << passed << " / " << total << " passed\n";
    std::cout << "========================================\n";
    return passed == total ? 0 : 1;
}

} 

int main() {
    test::run_all(v2::matmul<16, 16, 16>);
    test::run_all(v3::matmul<16, 16, 16>);
    test::run_all(v4::matmul<16, 16, 32>);
}