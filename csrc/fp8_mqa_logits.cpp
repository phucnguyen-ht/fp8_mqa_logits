#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <limits>

#include "fp8_mqa_logits.h"

template <int NUM_HEADS=32, int HEAD_SIZE=128, int BLOCK_KV>
__global__ void FP8MQALogits(
  const fp8* Q_ptr,  // fp8e4m3 [seq_len, H, D]
  const fp8* KV_ptr,  // fp8e4m3 [seq_len_kv, D]
  const float* kv_scales_ptr,  // fp32 [seq_len_kv]
  const float* weights_ptr,  // fp32 [seq_len, H]
  const int* cu_start_ptr,  // int32 [seq_len]
  const int* cu_end_ptr,  // int32 [seq_len]
  float* logits_ptr,  // fp32 [seq_len, seq_len_kv]
  int seq_len,
  int seq_len_kv,
  int stride_q_s,
  int stride_q_h,
  int stride_q_d,
  int stride_kv_s,
  int stride_kv_d,
  int stride_w_s,
  int stride_w_h,
  int stride_logits_s,
  int stride_logits_k
) {
  constexpr int MFMA_MN = 16;
  constexpr int MFMA_K = 32;

  constexpr int GPRs_AB = 2;
  constexpr int GPRs_C = 4;

  constexpr int numInputElementMFMA = GPRs_AB * sizeof(float) / sizeof(fp8);
  constexpr int numOutputElementMFMA = GPRs_C;

  using VecInMFMA = __attribute__( (__vector_size__(GPRs_AB * sizeof(float)) )) fp8;
  using VecOutMFMA = __attribute__( (__vector_size__(GPRs_C * sizeof(float)) )) float;

  // Triton: row_id = tl.num_programs(0) - row_id - 1
  int row_id = gridDim.x - blockIdx.x - 1;
  
  if (row_id < 0 || row_id >= seq_len) return;

  int tid = threadIdx.x;
  int bdim = blockDim.x;

  const int warpId = tid / WARP_SIZE;
  const int laneId = tid % WARP_SIZE;

  const int mfmaInRow = laneId % MFMA_MN;
  const int mfmaInCol = numInputElementMFMA * (laneId / MFMA_MN);

  const int mfmaOutRow = numOutputElementMFMA * (laneId / MFMA_MN);
  const int mfmaOutCol = laneId % MFMA_MN;

  // Start and end bounds for this sequence row
  int start_ind = max(0, cu_start_ptr[row_id]);
  int end_ind = min(seq_len_kv, cu_end_ptr[row_id]);
  if (start_ind >= end_ind) return;

  __shared__ fp8 smem_Q[NUM_HEADS * HEAD_SIZE];
  __shared__ fp8 smem_KV[BLOCK_KV * HEAD_SIZE];
  __shared__ float smem_W[NUM_HEADS];

  // 1. Cooperatively load Q[NUM_HEADS, HEAD_SIZE] into shared memory
  constexpr int vecLoadLength = sizeof(float4) / sizeof(fp8);

  for (int i = tid * vecLoadLength; i < NUM_HEADS * HEAD_SIZE; i += bdim * vecLoadLength) {
    int h = i / HEAD_SIZE;
    int d = i % HEAD_SIZE;
    int q_offset = row_id * stride_q_s + h * stride_q_h + d * stride_q_d;
    // smem_Q[i] = Q_ptr[q_offset]; 
    *reinterpret_cast<float4 *>(&smem_Q[i]) = 
      *reinterpret_cast<const float4 *>(&Q_ptr[q_offset]);
  }

  // 2. Cooperatively load weights[NUM_HEADS] into shared memory
  for (int i = tid * 4; i < NUM_HEADS; i += bdim * 4) {
    int w_offset = row_id * stride_w_s + i * stride_w_h;
    // smem_W[i] = weights_ptr[w_offset];
    *reinterpret_cast<float4 *>(&smem_W[i]) = 
      *reinterpret_cast<const float4 *>(&weights_ptr[w_offset]);
  }

  __syncthreads();

  // 3. Loop over KV tiles
  for (int kv_block_start = start_ind; kv_block_start < end_ind; kv_block_start += BLOCK_KV) {
    int kv_cols_valid = min(BLOCK_KV, end_ind - kv_block_start);

    // Cooperatively load KV block mapped as [BLOCK_KV, HEAD_SIZE]
    // This ensures contiguous memory access (coalescing) along the D dimension.
    for (int i = tid * vecLoadLength; i < BLOCK_KV * HEAD_SIZE; i += bdim * vecLoadLength) {
      int k = i / HEAD_SIZE;
      int d = i % HEAD_SIZE;
      
      if (k < kv_cols_valid) {
        int kv_idx = kv_block_start + k;
        int kv_offset = kv_idx * stride_kv_s + d * stride_kv_d;
        // smem_KV[k * HEAD_SIZE + d] = KV_ptr[kv_offset];
        *reinterpret_cast<float4 *>(&smem_KV[k * HEAD_SIZE + d]) = 
          *reinterpret_cast<const float4 *>(&KV_ptr[kv_offset]);
      } else {
        // smem_KV[k * HEAD_SIZE + d] = 0.0f; 
        *reinterpret_cast<float4 *>(&smem_KV[k * HEAD_SIZE + d]) = 
          make_float4(0, 0, 0, 0);
      }
    }
    
    __syncthreads();

    // Compute 
    for (int bk = warpId; bk < BLOCK_KV / MFMA_MN; bk += bdim / WARP_SIZE) {
      int k = bk * MFMA_MN;
      float kv_scale = 0.0f;
      if (k + mfmaOutCol < kv_cols_valid) {
        int kv_idx = kv_block_start + k + mfmaOutCol;
        kv_scale = kv_scales_ptr[kv_idx];
      }
      
      VecInMFMA vA[NUM_HEADS / MFMA_MN];
      VecInMFMA vB;
      VecOutMFMA vC[NUM_HEADS / MFMA_MN] = {0};

      #pragma unroll
      for (int d = 0; d < HEAD_SIZE; d += MFMA_K) {
        #pragma unroll
        for (int h = 0; h < NUM_HEADS; h += MFMA_MN) {
          vA[h / MFMA_MN] = *reinterpret_cast<VecInMFMA *>(&smem_Q[(h + mfmaInRow) * HEAD_SIZE + d + mfmaInCol]);
        }
        vB = *reinterpret_cast<VecInMFMA *>(&smem_KV[(k + mfmaInRow) * HEAD_SIZE + d + mfmaInCol]);
        #pragma unroll
        for (int i = 0; i < NUM_HEADS / MFMA_MN; ++i) {
          vC[i] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8((long)vA[i], (long)vB, vC[i], 0, 0, 0);
        }
      }

      float total_score = 0;
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

      // Assume that the mfma instruction is mfma_f32_16x16x32
      // (This reduction code should be altered if mfma instruction is changed)
      total_score += __shfl_down(total_score, 32);
      total_score += __shfl_down(total_score, 16);

      if (laneId < 16 && k + laneId < kv_cols_valid) {
        int kv_idx = kv_block_start + k + laneId;
        int logit_idx = row_id * stride_logits_s + kv_idx * stride_logits_k;
        logits_ptr[logit_idx] = total_score;
      }
    }

    __syncthreads(); 
  }
}

torch::Tensor launch_FP8MQALogits(
  torch::Tensor Q_ptr,  // fp8e4m3 [seq_len, H, D]
  torch::Tensor KV_ptr,  // fp8e4m3 [seq_len_kv, D]
  torch::Tensor kv_scales_ptr,  // fp32 [seq_len_kv]
  torch::Tensor weights_ptr,  // fp32 [seq_len, H]
  torch::Tensor cu_start_ptr,  // int32 [seq_len]
  torch::Tensor cu_end_ptr  // int32 [seq_len]
) {
  auto d_Q_ptr = static_cast<fp8 *>(Q_ptr.data_ptr());
  auto d_KV_ptr = static_cast<fp8 *>(KV_ptr.data_ptr());
  auto d_kv_scales_ptr = static_cast<float *>(kv_scales_ptr.data_ptr());
  auto d_weights_ptr = static_cast<float *>(weights_ptr.data_ptr());
  auto d_cu_start_ptr = static_cast<int *>(cu_start_ptr.data_ptr());
  auto d_cu_end_ptr = static_cast<int *>(cu_end_ptr.data_ptr());

  constexpr int BLOCK_KV = 256;
  auto seq_len = Q_ptr.size(0);
  auto num_heads = Q_ptr.size(1);
  auto head_size = Q_ptr.size(2);
  auto seq_len_kv = KV_ptr.size(0);

  auto NUM_HEADS = num_heads;
  auto HEAD_SIZE = head_size;

  auto stride_q_s = Q_ptr.stride(0);
  auto stride_q_h = Q_ptr.stride(1);
  auto stride_q_d = Q_ptr.stride(2);

  auto stride_kv_s = KV_ptr.stride(0);
  auto stride_kv_d = KV_ptr.stride(1);

  auto stride_w_s = weights_ptr.stride(0);
  auto stride_w_h = weights_ptr.stride(1);

  torch::Tensor logits = torch::full(
    {seq_len, seq_len_kv},                                 // Shape as an initializer list
    -std::numeric_limits<float>::infinity(),               // Fill value (-inf)
    torch::TensorOptions().dtype(torch::kFloat32).device(Q_ptr.device()) // Options
  );

  auto stride_logits_s = logits.stride(0);
  auto stride_logits_k = logits.stride(1);

  auto d_logits_ptr = static_cast<float *>(logits.data_ptr());

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weights_ptr));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(seq_len);
  dim3 block(BLOCK_KV);

  FP8MQALogits<32, 128, BLOCK_KV><<<grid, block, 0, stream>>>(
    d_Q_ptr,
    d_KV_ptr,
    d_kv_scales_ptr,
    d_weights_ptr,
    d_cu_start_ptr,
    d_cu_end_ptr,
    d_logits_ptr,
    seq_len,
    seq_len_kv,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_kv_s,
    stride_kv_d,
    stride_w_s,
    stride_w_h,
    stride_logits_s,
    stride_logits_k
  );

  return logits;
}