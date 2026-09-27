#pragma once

#include <cuda_runtime.h>
#include <math.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// =============================================================================
//                    Sum
// =============================================================================

/// 一个 warp（32 个线程）内做全归约求和
/// 输入每个线程本地的 v，返回该 warp 内所有 32 个线程 v 的总和。
__device__ __forceinline__ float wrap_reduce_sum(float v) {
    #pragma unroll
    /*
        __shfl_xor_sync: 
        - 0xffffffff: 参与交换的线程掩码，表示全部 32 个线程都参与
        - var: 要交换的值（这里是 v）
        - laneMask: 按位异或的 lane 掩码。当前线程会拿到 lane_id ^ laneMask 那个线程的 v 值。
    */
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

template <size_t BLOCK>
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[BLOCK / 32];
    const size_t lane = threadIdx.x & 31;
    const size_t warp = threadIdx.x >> 5;

    // 每 32 个 thread 内部归约
    v = wrap_reduce_sum(v);
    // 每组第一个 thread 写入共享内存
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();

    // 由第一组 warp 的 thread 读取
    if (warp == 0) {
        // 这个 warp 内部的 thread 选一个对应位置的求和值
        // BLOCK / 32 表示 warp 的数量，但这里要求数量不多于 32
        v = (lane < BLOCK / 32) ? warp_sums[lane] : 0.f;
        v = wrap_reduce_sum(v);
    }

    return v;
}

// =============================================================================
//                    Max
// =============================================================================

__device__ __forceinline__ float wrap_reduce_max(float v) {
    #pragma unroll
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return 0;
}

template <size_t BLOCK>
__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float warp_maxs[BLOCK / 32];
    const size_t lane = threadIdx.x & 31;
    const size_t warp = threadIdx.x >> 5;

    v = wrap_reduce_max(v);
    if (lane == 0) {
        warp_maxs[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        v = (lane < BLOCK / 32) ? warp_maxs[lane] : -INFINITY;
        v = wrap_reduce_max(v);
    }

    return v;
}

// =============================================================================
//                    Scan
// =============================================================================

__device__ __forceinline__ float warp_scan_inclusive(float v) {
    const size_t lane = threadIdx.x & 31;
    #pragma unroll
    /*
        offset: 1, 2, 4, 8, 16
        - 1: [a0, a0+a1, a1+a2, a2+a3, a3+a4, a4+a5, a5+a6, a6+a7]
        - 2: [a0, a0+a1, a0+a1+a2, a1+a2+a3, a2+a3+a4, a3+a4+a5, a4+a5+a6, a5+a6+a7]
        - 4: [a0, a0+a1, a0+a1+a2, a1+a2+a3, a0+a2+a3+a4, a1+a3+a4+a5, a2+a4+a5+a6, a3+a5+a6+a7]
        - 8: [a0, a0+a1, a0+a1+a2, a1+a2+a3, a0+a2+a3+a4, a1+a3+a4+a5, a2+a4+a5+a6, a3+a5+a6+a7]
    */
    for (size_t offset = 1; offset < 32; offset <<= 1) {
        // 当前线程读取 lane 编号比自己小 offset 的那个线程的 var。
        float other = __shfl_up_sync(0xffffffff, v, offset);
        // 如果 i - offset < 0（越界），返回的是自己的值（未定义但实际常是自己），所以必须用 if (lane >= offset) 把它挡掉。
        v += (lane >= offset) ? other : 0.f;
    }
    return v;
}

template <size_t BLOCK>
__device__ __forceinline__ float block_scan_inclusive(float v) {
    __shared__ float arrays[BLOCK / 32];
    const size_t lane = threadIdx.x & 31;
    const size_t warp = threadIdx.x >> 5;

    v = warp_scan_inclusive(v);
    if (lane == 31) {
        arrays[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        float w = (lane < BLOCK / 32) ? arrays[lane] : 0.f;
        w = warp_scan_inclusive(w);
        if (lane < BLOCK / 32) {
            arrays[lane] = w;
        }
    }
    __syncthreads();

    if (warp > 0) {
        v += arrays[warp - 1];
    }
    return v;
}

template <int BLOCK>
__device__ __forceinline__ float block_scan_exclusive(float v) {
    return block_scan_inclusive<BLOCK>(v) - v;
}

// =============================================================================
//                       Online Softmax
// =============================================================================

template <int BLOCK>
__device__ __forceinline__ void block_online_softmax(float& m, float& l) {
    __shared__ float warp_m[BLOCK / 32], warp_l[BLOCK / 32];
    __shared__ float block_m, block_l;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    
    // 计算全局最大值
    float tm = wrap_reduce_max(m);
    if (lane == 0) {
        warp_m[warp] = tm;
    }
    __syncthreads();
    if (warp == 0) {
        tm = (lane < BLOCK / 32) ? warp_m[lane] : -INFINITY;
        tm = wrap_reduce_max(tm);
        if (lane == 0) {
            block_m = tm;
        }
    }
    __syncthreads();
    const float m_g = block_m;

    // 对每个 m，计算 exp(m - max_m)
    float tl = l * __expf(m - m_g);
    tl = wrap_reduce_sum(tl);
    if (lane == 0) {
        warp_l[warp] = tl;
    }
    __syncthreads();
    if (warp == 0) {
        tl = (lane < BLOCK / 32) ? warp_l[lane] : 0.f;
        tl = wrap_reduce_sum(tl);
        if (lane == 0)
            block_l = tl;
    }
    __syncthreads();

    m = m_g;
    l = block_l;
}

// =============================================================================
//                         Vector load / store helpers
// =============================================================================

/*
    const float* p,   // 矩阵基址（行主序）
    int ld,           // leading dimension：两行之间的跨度（ld >= nc）
    int r, int c,     // 当前要加载的起始行、列
    int nr, int nc    // 矩阵实际的行数、列数（用于边界判断）
*/
__device__ __forceinline__ float4 load4(const float* p, int ld, int r, int c, int nr, int nc) {
    const int i = r * ld + c;
    // 行有效（不越界）、4 个元素都在列范围内、线性下标 i 是 4 的倍数，内存对齐到 16 字节。
    if (r < nr && c + 3 < nc && (i & 3) == 0) {
        return __ldg(reinterpret_cast<const float4*>(p + i));
    } 
    return make_float4(
        (r < nr && c < nc) ? p[i] : 0.f, 
        (r < nr && c + 1 < nc) ? p[i + 1] : 0.f,
        (r < nr && c + 2 < nc) ? p[i + 2] : 0.f, 
        (r < nr && c + 3 < nc) ? p[i + 3] : 0.f
    );
}

__device__ __forceinline__ void store4(float* p, int ld, int r, int c, float4 v, int nr, int nc) {
    const int i = r * ld + c;
    if (r < nr && c + 3 < nc && (i & 3) == 0) {
        *reinterpret_cast<float4*>(p + i) = v;
        return;
    }
    if (r < nr && c + 0 < nc) p[i + 0] = v.x;
    if (r < nr && c + 1 < nc) p[i + 1] = v.y;
    if (r < nr && c + 2 < nc) p[i + 2] = v.z;
    if (r < nr && c + 3 < nc) p[i + 3] = v.w;
}

/// 向量加载
__device__ __forceinline__ float4 load4(const float* p, int i, int n) {
    return load4(p, n, 0, i, 1, n);
}

__device__ __forceinline__ void store4(float* p, int i, int n, float4 v) {
    store4(p, n, 0, i, v, 1, n);
}
