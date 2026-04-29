#pragma once
#include <torch/torch.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "utils/utils.h"

// ============================================================================
// fp8_paged_mqa_logits — decode-phase indexer kernel (GLM-5-FP8, CDNA3)
//
// Tunable parameters:
//   NUM_WARPS  — template: number of warps per block (affects occupancy)
//   CHUNK_K    — template: K positions processed per outer loop (affects SMEM)
//   SplitKV    — runtime:  parallelism over context length (affects grid size)
//
// Uses V_MFMA_F32_16X16X32_FP8_FP8.
// ============================================================================

using fp8 = __hip_fp8_storage_t;

constexpr int WARP_SIZE  = 64;
constexpr int NUM_HEADS  = 32;
constexpr int HEAD_SIZE  = 128;

#define CDIV(a, b) ((a) + (b) - 1) / b 

// ============================================================================
// Kernel
// ============================================================================
namespace v1 {
    template <int NUM_WARPS, int CHUNK_K>
    __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
    void fp8_paged_mqa_logits_kernel(
        const fp8*   __restrict__ Q_ptr,           // [batch, next_n, 32, 128]     fp8
        const fp8*   __restrict__ kv_cache_ptr,    // [num_phys_blocks, index_dim] raw
        const float* __restrict__ weights_ptr,     // [batch*next_n, 32]           fp32
        const int*   __restrict__ context_lens,    // [batch]                      int32
        const int*   __restrict__ block_tables,    // [batch, max_blocks_per_seq]  int32
        float*       __restrict__ logits_ptr,      // [batch*next_n, max_model_len] fp32
        int batch_size, int next_n,
        int max_blocks_per_seq, int max_model_len, int index_dim,
        int SplitKV                                // runtime tunable
    ) {
        constexpr int MFMA_MN    = 16;
        constexpr int MFMA_K     = 32;
        constexpr int GPRs_AB    = 2;
        constexpr int GPRs_C     = 4;
        constexpr int numInputElementMFMA  = GPRs_AB * sizeof(float) / sizeof(fp8);  // 8
        constexpr int numOutputElementMFMA = GPRs_C;                                 // 4

        using VecInMFMA  = __attribute__((__vector_size__(GPRs_AB * sizeof(float)))) fp8;
        using VecOutMFMA = __attribute__((__vector_size__(GPRs_C  * sizeof(float)))) float;

        constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;
        constexpr int NUM_MFMA_ACC  = HEAD_SIZE / MFMA_K;              // 4
        constexpr int TILES_PER_CHUNK = CHUNK_K / MFMA_MN;             // e.g. 256/16 = 16
        static_assert(CHUNK_K % MFMA_MN == 0, "CHUNK_K must be multiple of MFMA_MN=16");

        // ---- block → (batch, next_n, split_kv) ----
        const int pid          = blockIdx.x;
        const int pid_split_kv = pid % SplitKV;
        const int batch_next   = pid / SplitKV;
        const int pid_batch    = batch_next / next_n;
        const int pid_next_n   = batch_next % next_n;
        if (pid_batch >= batch_size) return;

        const int tid    = threadIdx.x;
        const int warpId = tid / WARP_SIZE;
        const int laneId = tid % WARP_SIZE;

        const int mfmaInRow  = laneId % MFMA_MN;
        const int mfmaInCol  = numInputElementMFMA * (laneId / MFMA_MN);
        const int mfmaOutRow = numOutputElementMFMA * (laneId / MFMA_MN);
        const int mfmaOutCol = laneId % MFMA_MN;

        // ---- context range for this split ----
        const int ctx_len      = context_lens[pid_batch];
        const int ctx_chunks   = (ctx_len + CHUNK_K - 1) / CHUNK_K;
        const int split_chunks = (ctx_chunks + SplitKV - 1) / SplitKV;
        const int split_start  = pid_split_kv * split_chunks * CHUNK_K;
        const int split_end    = min(ctx_len, split_start + split_chunks * CHUNK_K);
        if (split_start >= ctx_len) return;

        // ================================================================
        // Shared memory (compile-time sizes from template params)
        // ================================================================
        __shared__ fp8   smem_Q    [NUM_HEADS * HEAD_SIZE];      // 4096 B
        __shared__ float smem_W    [NUM_HEADS];                  // 128 B
        __shared__ fp8   smem_KV   [CHUNK_K   * HEAD_SIZE];      // CHUNK_K×128 B
        __shared__ float smem_scale[CHUNK_K];                    // CHUNK_K×4 B

        // ================================================================
        // 1. Cooperatively load Q + W → SMEM
        // ================================================================
        constexpr int VEC_LEN = sizeof(float4) / sizeof(fp8);    // 16

        const int q_base = (pid_batch * next_n + pid_next_n) * NUM_HEADS * HEAD_SIZE;
        for (int i = tid * VEC_LEN; i < NUM_HEADS * HEAD_SIZE; i += BLOCK_THREADS * VEC_LEN) {
            *reinterpret_cast<float4*>(&smem_Q[i]) =
                *reinterpret_cast<const float4*>(&Q_ptr[q_base + i]);
        }

        const int w_base = (pid_batch * next_n + pid_next_n) * NUM_HEADS;
        if (tid < 8) {  // 32 floats / 4 per float4 = 8 loads
            *reinterpret_cast<float4*>(&smem_W[tid * 4]) =
                *reinterpret_cast<const float4*>(&weights_ptr[w_base + tid * 4]);
        }

        __syncthreads();

        // ---- pointers ----
        const int* bt       = block_tables + (int64_t)pid_batch * max_blocks_per_seq;
        float*     out_base = logits_ptr + (int64_t)(pid_batch * next_n + pid_next_n) * max_model_len;
        const int  causal_limit = ctx_len - next_n + pid_next_n;

        // ================================================================
        // 2. Main loop over KV tiles
        // ================================================================
        for (int kv_start = split_start; kv_start < split_end; kv_start += CHUNK_K) {
            const int kv_valid = min(CHUNK_K, split_end - kv_start);

            // ---- 2a. Cooperative load K data → smem_KV ----
            for (int i = tid * VEC_LEN; i < CHUNK_K * HEAD_SIZE; i += BLOCK_THREADS * VEC_LEN) {
                const int k = i / HEAD_SIZE;
                const int d = i % HEAD_SIZE;
                if (k < kv_valid) {
                    const int phys = bt[kv_start + k];
                    *reinterpret_cast<float4*>(&smem_KV[k * HEAD_SIZE + d]) =
                        *reinterpret_cast<const float4*>(kv_cache_ptr + (int64_t)phys * index_dim + d);
                } else {
                    *reinterpret_cast<float4*>(&smem_KV[k * HEAD_SIZE + d]) = make_float4(0, 0, 0, 0);
                }
            }

            // ---- 2b. Cooperative load K scales → smem_scale ----
            for (int i = tid; i < CHUNK_K; i += BLOCK_THREADS) {
                if (i < kv_valid) {
                    const int phys = bt[kv_start + i];
                    smem_scale[i] = *reinterpret_cast<const float*>(
                        kv_cache_ptr + (int64_t)phys * index_dim + HEAD_SIZE);
                } else {
                    smem_scale[i] = 0.0f;
                }
            }

            __syncthreads();

            // ---- 2c. Compute: each warp handles 16-position tiles ----
            for (int bk = warpId; bk < TILES_PER_CHUNK; bk += NUM_WARPS) {
                const int k = bk * MFMA_MN;

                float kv_scale = (k + mfmaOutCol < kv_valid) ? smem_scale[k + mfmaOutCol] : 0.0f;

                VecInMFMA  vA[NUM_HEADS / MFMA_MN];
                VecInMFMA  vB;
                VecOutMFMA vC[NUM_HEADS / MFMA_MN] = {};

                #pragma unroll
                for (int d = 0; d < HEAD_SIZE; d += MFMA_K) {
                    #pragma unroll
                    for (int h = 0; h < NUM_HEADS; h += MFMA_MN) {
                        vA[h / MFMA_MN] = *reinterpret_cast<const VecInMFMA*>(
                            &smem_Q[(h + mfmaInRow) * HEAD_SIZE + d + mfmaInCol]);
                    }
                    vB = *reinterpret_cast<const VecInMFMA*>(
                        &smem_KV[(k + mfmaInRow) * HEAD_SIZE + d + mfmaInCol]);
                    #pragma unroll
                    for (int i = 0; i < NUM_HEADS / MFMA_MN; ++i) {
                        vC[i] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                            (long)vA[i], (long)vB, vC[i], 0, 0, 0);
                    }
                }

                float total_score = 0.0f;
                #pragma unroll
                for (int i = 0; i < NUM_HEADS / MFMA_MN; ++i) {
                    #pragma unroll
                    for (int j = 0; j < numOutputElementMFMA; ++j) {
                        vC[i][j] *= kv_scale;
                        vC[i][j] = fmaxf(vC[i][j], 0.0f);
                        vC[i][j] *= smem_W[i * MFMA_MN + mfmaOutRow + j];
                        total_score += vC[i][j];
                    }
                }

                total_score += __shfl_down(total_score, 32);
                total_score += __shfl_down(total_score, 16);

                if (laneId < 16) {
                    const int abs_pos = kv_start + k + laneId;
                    if (abs_pos < ctx_len && abs_pos <= causal_limit && abs_pos < max_model_len) {
                        out_base[abs_pos] = total_score;
                    }
                }
            }

            __syncthreads();
        }
    }
}

namespace v2 {
    template <int NUM_WARPS, int CHUNK_K>
    __global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
    void fp8_paged_mqa_logits_kernel(
        const fp8*   __restrict__ Q_ptr,           // [batch, next_n, 32, 128]     fp8
        const fp8*   __restrict__ kv_cache_ptr,    // [num_phys_blocks, index_dim] raw
        const float* __restrict__ weights_ptr,     // [batch*next_n, 32]           fp32
        const int*   __restrict__ context_lens,    // [batch]                      int32
        const int*   __restrict__ block_tables,    // [batch, max_blocks_per_seq]  int32
        float*       __restrict__ logits_ptr,      // [batch*next_n, max_model_len] fp32
        int batch_size, int next_n,
        int max_blocks_per_seq, int max_model_len, int index_dim,
        int SplitKV                                // runtime tunable
    ){
        // ---- MFMA 16×16×32 constants ----
        constexpr int MFMA_MN = 16;
        constexpr int MFMA_K  = 32;
        constexpr int GPRs_AB = 2;
        constexpr int GPRs_C  = 4;
        constexpr int numInputElementMFMA  = GPRs_AB * sizeof(float) / sizeof(fp8);  // 8
        constexpr int numOutputElementMFMA = GPRs_C;                                  // 4

        using VecInMFMA  = __attribute__((__vector_size__(GPRs_AB * sizeof(float)))) fp8;
        using VecOutMFMA = __attribute__((__vector_size__(GPRs_C  * sizeof(float)))) float;

        // ---- derived constants ----
        constexpr int BLOCK_THREADS   = NUM_WARPS * WARP_SIZE;
        constexpr int H_LOOPS         = NUM_HEADS / MFMA_MN;                          // 2
        constexpr int D_LOOPS         = HEAD_SIZE / MFMA_K;                           // 4
        constexpr int TILES_PER_CHUNK = CHUNK_K / MFMA_MN;
        constexpr int PAD             = 8;
        constexpr int KV_ROW          = HEAD_SIZE + PAD;                              // 136
        constexpr int VEC_LEN         = sizeof(float4) / sizeof(fp8);                 // 16
        constexpr int NB_LOAD_KV      = CHUNK_K * HEAD_SIZE / (BLOCK_THREADS * VEC_LEN);
        constexpr int NB_LOAD_SCALE   = CDIV(CHUNK_K, BLOCK_THREADS);

        static_assert(CHUNK_K * HEAD_SIZE % (BLOCK_THREADS * VEC_LEN) == 0, "CHUNK_K * HEAD_SIZE must be divisible by BLOCK_THREADS * VEC_LEN");
        static_assert(CHUNK_K % MFMA_MN == 0, "CHUNK_K must be multiple of MFMA_MN=16");

        // ---- block → (batch, next_n, split_kv) ----
        const int pid_batch    = blockIdx.x;
        const int pid_next_n   = blockIdx.y;
        const int pid_split_kv = blockIdx.z;
        if (pid_batch >= batch_size) return;

        // ---- context range for this split ----
        const int ctx_len      = context_lens[pid_batch];
        const int ctx_chunks   = CDIV(ctx_len, CHUNK_K);
        const int split_chunks = CDIV(ctx_chunks, SplitKV);
        const int split_start  = pid_split_kv * split_chunks * CHUNK_K;
        const int split_end    = min(ctx_len, split_start + split_chunks * CHUNK_K);
        if (split_start >= ctx_len) return;

        // ---- thread IDs ----
        const int tid        = threadIdx.x;
        const int warpId     = tid / WARP_SIZE;
        const int laneId     = tid % WARP_SIZE;
        const int mfmaInRow  = laneId % MFMA_MN;                            // [0...15]
        const int mfmaInCol  = numInputElementMFMA * (laneId / MFMA_MN);    // [0, 8, 16, 24]
        const int mfmaOutRow = numOutputElementMFMA * (laneId / MFMA_MN);   // [0, 4, 8, 12]
        const int mfmaOutCol = laneId % MFMA_MN;                            // [0...15]

        // ---- shared memory ----
        __shared__ fp8   smem_Q    [NUM_HEADS * HEAD_SIZE];     // 4 KB
        __shared__ float smem_W    [NUM_HEADS];                 // 128 B
        __shared__ fp8   smem_KV   [CHUNK_K][KV_ROW];           // CHUNK_K × 136 B
        __shared__ float smem_scale[CHUNK_K];                   // CHUNK_K × 4 B

        // ---- persistent register arrays (loaded once) ----
        VecInMFMA q_reg[D_LOOPS][H_LOOPS];
        float     w_reg[H_LOOPS][numOutputElementMFMA];

        // ---- prefetch register buffers (double-buffer) ----
        float4 pf_kv[NB_LOAD_KV];
        float  pf_scale[NB_LOAD_SCALE];

        // ---- derived pointers ----
        const int  q_row        = pid_batch * next_n + pid_next_n;
        const int* bt           = block_tables + (int64_t)pid_batch * max_blocks_per_seq;
        float*     out_base     = logits_ptr + (int64_t)q_row * max_model_len;
        const int  causal_limit = ctx_len - next_n + pid_next_n;

        // ================================================================
        //  Lambdas
        // ================================================================

        // ---- cooperative load Q[32,128] + W[32] → SMEM ----
        auto load_qw_global = [&]() {
            const int q_base = q_row * NUM_HEADS * HEAD_SIZE;
            for (int i = tid * VEC_LEN; i < NUM_HEADS * HEAD_SIZE; i += BLOCK_THREADS * VEC_LEN) {
                *reinterpret_cast<float4*>(&smem_Q[i]) =
                    *reinterpret_cast<const float4*>(&Q_ptr[q_base + i]);
            }
            const int w_base = q_row * NUM_HEADS;
            if (tid < 8) {
                *reinterpret_cast<float4*>(&smem_W[tid * 4]) =
                    *reinterpret_cast<const float4*>(&weights_ptr[w_base + tid * 4]);
            }
        };

        // ---- Q / W from SMEM → register arrays (called once) ----
        auto load_qw_to_regs = [&]() {
            #pragma unroll
            for (int d = 0; d < D_LOOPS; ++d) {
                #pragma unroll
                for (int h = 0; h < H_LOOPS; ++h) {
                    q_reg[d][h] = *reinterpret_cast<const VecInMFMA*>(
                        &smem_Q[(h * MFMA_MN + mfmaInRow) * HEAD_SIZE + d * MFMA_K + mfmaInCol]);
                }
            }
            #pragma unroll
            for (int h = 0; h < H_LOOPS; ++h) {
                #pragma unroll
                for (int j = 0; j < numOutputElementMFMA; ++j) {
                    w_reg[h][j] = smem_W[h * MFMA_MN + mfmaOutRow + j];
                }
            }
        };

        // ---- cooperative paged K + scale → SMEM (initial tile) ----
        auto load_kv_to_smem = [&](int kv_start, int kv_valid) {
            for (int i = tid * VEC_LEN; i < CHUNK_K * HEAD_SIZE; i += BLOCK_THREADS * VEC_LEN) {
                int k = i / HEAD_SIZE;
                int d = i % HEAD_SIZE;
                if (k < kv_valid) {
                    int phys = bt[kv_start + k];
                    *reinterpret_cast<float4*>(&smem_KV[k][d]) =
                        *reinterpret_cast<const float4*>(kv_cache_ptr + (int64_t)phys * index_dim + d);
                } else {
                    *reinterpret_cast<float4*>(&smem_KV[k][d]) = make_float4(0, 0, 0, 0);
                }
            }
            for (int i = tid; i < CHUNK_K; i += BLOCK_THREADS) {
                if (i < kv_valid) {
                    int phys = bt[kv_start + i];
                    smem_scale[i] = *reinterpret_cast<const float*>(
                        kv_cache_ptr + (int64_t)phys * index_dim + HEAD_SIZE);
                } else {
                    smem_scale[i] = 0.0f;
                }
            }
        };

        // ---- paged K + scale → register buffers (prefetch next tile) ----
        auto prefetch_kv = [&](int kv_start, int kv_valid) {
            #pragma unroll
            for (int i = 0; i < NB_LOAD_KV; ++i) {
                int idx = tid * VEC_LEN + i * BLOCK_THREADS * VEC_LEN;
                int k = idx / HEAD_SIZE;
                int d = idx % HEAD_SIZE;
                if (k < kv_valid) {
                    int phys = bt[kv_start + k];
                    pf_kv[i] = *reinterpret_cast<const float4*>(
                        kv_cache_ptr + (int64_t)phys * index_dim + d);
                } else {
                    pf_kv[i] = make_float4(0, 0, 0, 0);
                }
            }
            #pragma unroll
            for (int i = 0; i < NB_LOAD_SCALE; ++i) {
                int idx = tid + i * BLOCK_THREADS;
                if (idx < kv_valid) {
                    int phys = bt[kv_start + idx];
                    pf_scale[i] = *reinterpret_cast<const float*>(
                        kv_cache_ptr + (int64_t)phys * index_dim + HEAD_SIZE);
                } else {
                    pf_scale[i] = 0.0f;
                }
            }
        };

        // ---- register buffers → SMEM (flush after compute) ----
        auto flush_kv_prefetch = [&]() {
            #pragma unroll
            for (int i = 0; i < NB_LOAD_KV; ++i) {
                int idx = tid * VEC_LEN + i * BLOCK_THREADS * VEC_LEN;
                int k = idx / HEAD_SIZE;
                int d = idx % HEAD_SIZE;
                *reinterpret_cast<float4*>(&smem_KV[k][d]) = pf_kv[i];
            }
            #pragma unroll
            for (int i = 0; i < NB_LOAD_SCALE; ++i) {
                int idx = tid + i * BLOCK_THREADS;
                if (idx < CHUNK_K) smem_scale[idx] = pf_scale[i];
            }
        };

        // ---- MFMA compute + post-process + store ----
        auto compute_and_store = [&](int kv_start, int kv_valid) {
            for (int bk = warpId; bk < TILES_PER_CHUNK; bk += NUM_WARPS) {
                const int k = bk * MFMA_MN;
                const float kv_scale = (k + mfmaOutCol < kv_valid) ? smem_scale[k + mfmaOutCol] : 0.0f;

                VecOutMFMA vC[H_LOOPS] = {};

                #pragma unroll
                for (int d = 0; d < D_LOOPS; ++d) {
                    VecInMFMA vB = *reinterpret_cast<const VecInMFMA*>(
                        &smem_KV[k + mfmaInRow][d * MFMA_K + mfmaInCol]);
                    #pragma unroll
                    for (int h = 0; h < H_LOOPS; ++h) {
                        vC[h] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                            (long)q_reg[d][h], (long)vB, vC[h], 0, 0, 0);
                    }
                }

                float total_score = 0.0f;
                #pragma unroll
                for (int h = 0; h < H_LOOPS; ++h) {
                    #pragma unroll
                    for (int j = 0; j < numOutputElementMFMA; ++j) {
                        total_score += fmaxf(vC[h][j], 0.0f) * kv_scale * w_reg[h][j];
                    }
                }

                total_score += __shfl_down(total_score, 32);
                total_score += __shfl_down(total_score, 16);

                if (laneId < 16) {
                    const int abs_pos = kv_start + k + laneId;
                    if (abs_pos < ctx_len && abs_pos <= causal_limit && abs_pos < max_model_len) {
                        out_base[abs_pos] = total_score;
                    }
                }
            }
        };

        // ================================================================
        //  Execution
        // ================================================================
        // Phase 1: Q, W → SMEM → registers
        load_qw_global();
        __syncthreads();
        load_qw_to_regs();

        // Phase 2: first KV tile → SMEM
        const int first_valid = min(CHUNK_K, split_end - split_start);
        load_kv_to_smem(split_start, first_valid);
        __syncthreads();

        // Phase 3: main loop with double-buffered KV
        for (int kv_start = split_start; kv_start < split_end; kv_start += CHUNK_K) {
            const int kv_valid = min(CHUNK_K, split_end - kv_start);
            const bool has_next = (kv_start + CHUNK_K < split_end);

            // Issue HBM loads for next tile → register buffers
            if (has_next) {
                const int next_valid = min(CHUNK_K, split_end - kv_start - CHUNK_K);
                prefetch_kv(kv_start + CHUNK_K, next_valid);
            }

            // MFMA compute on current tile (reads from SMEM + Q/W registers)
            compute_and_store(kv_start, kv_valid);

            __syncthreads();  // all warps done reading SMEM

            // Flush register buffers → SMEM for next iteration
            if (has_next) {
                flush_kv_prefetch();
            }

            __syncthreads();  // SMEM ready for next iteration
        }

        // double-buffering: 2*smem
        // load-smem chunk-i
        // for chunk tiếp theo:
        //     -> load vào smem chunk-i+1
        //     -> tính toán trên chunk-i
        //     -> syncthreads()

        // double-buffering: smem+reg
        // load-reg chunk-i
        // for chunk:
        //.    -> load chunk-i+1 vào smem (-> reg, reg->smem)
        //         -> global->reg
        //         -> wait_cnt()
        //         -> reg->smem
        //     -> tính toán trên chunk-i/reg
        //     -> đưa từ smem-i+1 -> reg-i

        // double-buffering: smem+reg
        // load-smem chunk-i
        // for chunk:
        //     -> load chunk-i+1 -> reg
        //     -> tính toán dựa trên chunk-i/smem (smem->reg)
        //      -> wait_cnt()
        //     -> flush: đưa từ reg/i+1 -> smem/chunk-i
        //     -> syncthreads()
    }
}

// ============================================================================
// Host dispatch
// ============================================================================
void launch_fp8_paged_mqa_logits(
    const fp8* d_q, const fp8* d_kv, const float* d_w,
    const int* d_ctx, const int* d_bt, float* d_out,
    int batch_size, int next_n,
    int max_blocks_per_seq, int max_model_len, int index_dim,
    int ChunkK, int SplitKV, int num_warps, cudaStream_t stream
) {
    const dim3 grid = dim3(batch_size, next_n, SplitKV);

    #define LAUNCH(NW, CK)                                                              \
        v2::fp8_paged_mqa_logits_kernel<NW, CK><<<grid, NW * WARP_SIZE, 0, stream>>>(         \
            d_q, d_kv, d_w, d_ctx, d_bt, d_out,                                        \
            batch_size, next_n, max_blocks_per_seq, max_model_len, index_dim, SplitKV);

    #define LAUNCH_NW(NW)                                                               \
        switch (ChunkK) {                                                               \
            case 64:  { LAUNCH(NW, 64)  break; }                                       \
            case 128: { LAUNCH(NW, 128) break; }                                       \
            case 256: { LAUNCH(NW, 256) break; }                                       \
            default:                                                                    \
                throw std::runtime_error("Unsupported ChunkK=" + std::to_string(ChunkK)\
                    + ". Supported: 64, 128, 256");                                     \
        }

    switch (num_warps) {
        case 2: { LAUNCH_NW(2) break; }
        case 4: { LAUNCH_NW(4) break; }
        case 8: { LAUNCH_NW(8) break; }
        default:
            throw std::runtime_error("Unsupported num_warps=" + std::to_string(num_warps)
                + ". Supported: 2, 4, 8");
    }
    #undef LAUNCH_NW
    #undef LAUNCH
}

// ============================================================================
// Python-facing entry point
// ============================================================================
torch::Tensor fp8_paged_mqa_logits(
    torch::Tensor q_fp8,            // [batch, next_n, 32, 128]
    torch::Tensor kv_cache_fp8,     // [num_blocks, 1, 1, index_dim]
    torch::Tensor weights,          // [batch*next_n, 32]
    torch::Tensor context_lens,     // [batch]
    torch::Tensor block_tables,     // [batch, max_blocks_per_seq]
    int max_model_len,
    int ChunkK    = 256,
    int SplitKV   = -1,             // -1 = auto
    int num_warps = 4,
    int TotalCuCount = 304
) {
    const int batch_size         = q_fp8.size(0);
    const int next_n             = q_fp8.size(1);
    const int n_heads            = q_fp8.size(2);
    const int head_dim           = q_fp8.size(3);
    const int index_dim          = kv_cache_fp8.size(3);
    const int max_blocks_per_seq = block_tables.size(1);

    TORCH_CHECK(n_heads == 32 && head_dim == 128, "Only n_heads=32, head_dim=128 supported");

    // ---- auto SplitKV ----
    if (SplitKV <= 0) {
        constexpr int WavePerEU    = 2;
        const int tiles = batch_size * next_n;
        SplitKV = std::max(1, ((std::max(1, TotalCuCount / tiles) + 4) / 5) * 5 * WavePerEU);
    }

    torch::Tensor out_logits = torch::full(
        {batch_size * next_n, max_model_len},
        -std::numeric_limits<float>::infinity(),
        torch::dtype(torch::kFloat32).device(q_fp8.device()));

    const at::cuda::OptionalCUDAGuard guard(device_of(out_logits));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_fp8_paged_mqa_logits(
        static_cast<fp8*>(q_fp8.data_ptr()),
        static_cast<fp8*>(kv_cache_fp8.data_ptr()),
        static_cast<float*>(weights.data_ptr()),
        static_cast<int*>(context_lens.data_ptr()),
        static_cast<int*>(block_tables.data_ptr()),
        out_logits.data_ptr<float>(),
        batch_size, next_n, max_blocks_per_seq, max_model_len, index_dim,
        ChunkK, SplitKV, num_warps, stream);

    return out_logits;
}