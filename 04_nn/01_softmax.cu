#include <cuda_runtime.h>

namespace v1 {

    template <const int BLOCK_ROW>
    __global__ void softmax_kernel(const float* x, float* y, const int ROWS, const int COLS) {
        const int row_block = blockDim.x * blockIdx.x;
        const int row = row_block + threadIdx.x;
        if (row >= ROWS) {
            return;
        }
        
        const float* x_ptr = x + row * COLS;
        float* y_ptr = y + row * COLS;

        // online softmax state
        float m_i = -INFINITY;
        float l_i = 0.0f;

        // 第一遍：计算 max 和 sum(exp(x - max))
        for (int col = 0; col < COLS; ++col) {
            float val = x_ptr[col];
            float m_new = fmaxf(m_i, val);
            float alpha = __expf(m_i - m_new);
            float val_exp = __expf(val - m_new);
            l_i = alpha * l_i + val_exp;
            m_i = m_new;
        }

        // 第二遍：计算 softmax
        for (int col = 0; col < COLS; ++col) {
            float val = x_ptr[col];
            float val_exp = __expf(val - m_i);
            y_ptr[col] = val_exp / l_i;
        }
    }

    template <const int BLOCK_ROW>
    void softmax(const float* x, float* y, const int ROWS, const int COLS) {
        // 每个 block 负责 BLOCK_ROW 行
        int num_blocks = (ROWS + BLOCK_ROW - 1) / BLOCK_ROW;
        softmax_kernel<BLOCK_ROW><<<num_blocks, BLOCK_ROW>>>(x, y, ROWS, COLS);
    }

}

namespace v2 {

    __device__ __forceinline__ float warp_max(float val) {
        for (int stride = 16; stride > 0; stride >>= 1) {
            val = std::max(val, __shfl_down_sync(0xffffffff, val, stride));
        }
        return val;
    }

    __device__ __forceinline__ float warp_sum(float val) {
        for (int stride = 16; stride > 0; stride >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, stride);
        }
        return val;
    }

    template <const int THREAD_SIZE>
    __global__ void softmax_kernel(const float* x, float* y, const int ROWS, const int COLS) {
        const int row = blockIdx.x;
        const float* x_ptr = x + row * COLS;
        float* y_ptr = y + row * COLS;
        const int tid = threadIdx.x;
        const int lane = tid & 31;
        const int warp = tid >> 5;
        static_assert(THREAD_SIZE % 32 == 0);
        constexpr int WARP_SIZE = THREAD_SIZE / 32;
        __shared__ float mem[WARP_SIZE];
        __shared__ float x_max;
        __shared__ float x_exp_sum;

        // 成为一个 block 内归约问题，先最一般的算法，三次循环 1、获得最大值；2、计算 exp 求和；3、计算输出

        // 1. max 归约
        // THREAD_SIZE 个 thread 先搬 COLS 个数据！
        float local = -INFINITY;
        for (int i = tid; i < COLS; i += THREAD_SIZE) {
            local = std::max(local, x_ptr[i]);
        }

        // 只需要对 THREAD_SIZE 个元素进行归约
        local = warp_max(local);

        // 交给 lane == 0 的 thread 来搬运
        if (lane == 0) {
            mem[warp] = local;
        }
        __syncthreads();
        // 再对这些归约，但需要注意我们只能用 32 个处理 WARP_SIZE 个，必须先全部放在 warp 0 上
        if (warp == 0) {
            // 这里又是 32 个元素搬运 WARP_SIZE 个值
            float local = -INFINITY;
            for (int i = lane; i < WARP_SIZE; i += 32) {
                local = std::max(local, mem[i]);
            }
            local = warp_max(local);
            // 由 lane 0 写入共享内存
            if (lane == 0) {
                x_max = local;
            }
        }
        __syncthreads();

        // 2. 计算 exp 求和
        local = 0.0f;
        for (int i = tid; i < COLS; i += THREAD_SIZE) {
            local += __expf(x_ptr[i] - x_max); 
        }
        local = warp_sum(local);
        if (lane == 0) {
            mem[warp] = local;
        }
        __syncthreads();
        if (warp == 0) {
            float local = 0.0f;
            for (int i = lane; i < WARP_SIZE; i += 32) {
                local += mem[i];
            }
            local = warp_sum(local);
            if (lane == 0) {
                x_exp_sum = local;
            }
        }
        __syncthreads();

        // 3. 计算输出
        for (int i = tid; i < COLS; i += THREAD_SIZE) {
            y_ptr[i] = __expf(x_ptr[i] - x_max) / x_exp_sum;
        }
    }

    template <const int THREAD_SIZE>
    void softmax(const float* x, float* y, const int ROWS, const int COLS) {
        // 每个 block 负责 1 行，使用 THREAD_SIZE 个 thread 一起处理 COLS 个元素
        softmax_kernel<THREAD_SIZE><<<ROWS, THREAD_SIZE>>>(x, y, ROWS, COLS);
    }

}

namespace v3 {

    __device__ __forceinline__ float warp_max(float val) {
        for (int stride = 16; stride > 0; stride >>= 1) {
            val = std::max(val, __shfl_down_sync(0xffffffff, val, stride));
        }
        return val;
    }

    __device__ __forceinline__ float warp_sum(float val) {
        for (int stride = 16; stride > 0; stride >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, stride);
        }
        return val;
    }

    template <const int THREAD_SIZE>
    __global__ void softmax_kernel(const float* x, float* y, const int ROWS, const int COLS) {
        const int row = blockIdx.x;
        const float* x_ptr = x + row * COLS;
        float* y_ptr = y + row * COLS;

        const int tid = threadIdx.x;
        __shared__ float x_cache1[THREAD_SIZE];
        __shared__ float x_cache2[THREAD_SIZE];
        __shared__ float m_i;
        __shared__ float l_i;

        // 成为一个 block 内归约问题，online softmax
        // 合并原来的 1/2 循环

        if (tid == 0) {
            m_i = -INFINITY;
            l_i = 0.0f;
        }
        __syncthreads();

        int num_tiles = (COLS + THREAD_SIZE - 1) / THREAD_SIZE;
        for (int c = 0; c < num_tiles; ++c) {
            // 这个循环，所有 thread 一起处理 [col_base, col_base + THREAD_SIZE) 列 
            // 加载一个块的 x
            int col_base = c * THREAD_SIZE;
            int col = col_base + tid;
            x_cache1[tid] = col < COLS ? x_ptr[col] : -INFINITY;
            x_cache2[tid] = col < COLS ? x_ptr[col] : -INFINITY;

            // 对这块 x 计算最大值
            for (int offset = THREAD_SIZE / 2; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    x_cache1[tid] = std::max(x_cache1[tid], x_cache1[tid+offset]);
                }
                __syncthreads();
            } // x_cache1[0] 是最大值！

            // 得到当前块的最大值
            float m_new = std::max(x_cache1[0], m_i);
            // 计算当前 block 的 exp
            x_cache2[tid] = __expf(x_cache2[tid] - m_new);
            __syncthreads();

            // 对这块 x 求和
            for (int offset = THREAD_SIZE / 2; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    x_cache2[tid] += x_cache2[tid+offset];
                }
                __syncthreads();
            } // x_cache2[0] 是求和！

            if (tid == 0) {
                float alpha = __expf(m_i - m_new);
                m_i = m_new;
                l_i = l_i * alpha + x_cache2[0];
            }
            __syncthreads();
        }

        // 2. 计算输出
        for (int i = tid; i < COLS; i += THREAD_SIZE) {
            y_ptr[i] = __expf(x_ptr[i] - m_i) / l_i;
        }        
    }

    template <const int THREAD_SIZE>
    void softmax(const float* x, float* y, const int ROWS, const int COLS) {
        // 每个 block 负责 1 行，使用 THREAD_SIZE 个 thread 一起处理 COLS 个元素，利用 online softmax
        softmax_kernel<THREAD_SIZE><<<ROWS, THREAD_SIZE>>>(x, y, ROWS, COLS);
    }

}