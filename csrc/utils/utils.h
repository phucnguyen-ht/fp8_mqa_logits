#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>

#include <iostream>
#include <math.h>
#include <vector>
#include <set>
#include <random>

using fp8 = __hip_fp8_storage_t;
using fp8x2 = __hip_fp8x2_storage_t;
using fp8x4 = __hip_fp8x4_storage_t;

using bfp16 = __hip_bfloat16;
using bfp16x2 = __hip_bfloat162;

using fp16 = __half;
using fp16x2 = __half2;

using fp32x2 = float2;
using uint8x2_t = uint8_t __attribute__((ext_vector_type(2)));
using uint8x8_t = uint8_t __attribute__((ext_vector_type(8)));
using uint8x16_t = uint8_t __attribute__((ext_vector_type(16)));
using intx2_t = int __attribute__((ext_vector_type(2)));
using longx2_t = long __attribute__((ext_vector_type(2)));
using longx4_t = long __attribute__((ext_vector_type(4)));
using floatx2_t = float __attribute__((ext_vector_type(2)));
using floatx4_t = float __attribute__((ext_vector_type(4)));
using floatx8_t = float __attribute__((ext_vector_type(8)));
using floatx16_t = float __attribute__((ext_vector_type(16)));

#define CHECK_HIP(call)                                                        \
{                                                                              \
  auto hip_res = call;                                                         \
  if (hip_res != hipSuccess) {                                                 \
    std::cerr << "Failed in HIP call: " << #call                               \
              << " at " << __FILE__ << ":" << __LINE__                         \
              << " with error: " << hipGetErrorString(hip_res) << std::endl;   \
    std::abort();                                                              \
  }                                                                            \
}

template <typename T>
__host__ __device__ inline T clamp(T v, T lo, T hi) {
    // return v < lo ? lo : (v > hi ? hi : v);
    return v > hi ? hi : (v < lo ? lo : v);
}

__host__ __device__ inline bfp16 silu(bfp16 x) {
  return x /
         (__float2bfloat16(1) + __float2bfloat16(expf(__bfloat162float(-x))));
}

__host__ __device__ inline float silu(float x)
{
  return x / (1.0f + expf(-x));
}

__host__ __device__ inline float sigmoid(float x) 
{
  return 1.0f / (1.0f + expf(-x));
}

__host__ __device__ inline bfp16 gelu(bfp16 x) {
  const bfp16 sqrt_2_over_pi = __float2bfloat16(0.7978845608028654);
  bfp16 z = sqrt_2_over_pi * (x + __float2bfloat16(0.044715) * x * x * x);
  return x * __float2bfloat16(0.5) *
         (__float2bfloat16(1) + __float2bfloat16(tanhf(__bfloat162float(z))));
}

__host__ __device__ inline float gelu(float x)
{
  return x * 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__host__ __device__ inline float gelu_erf(float x)
{
  constexpr float INV_SQRT2 = 0.70710678118654752440f; // 1 / sqrt(2)
  return x * 0.5f * (1.0f + erff(x * INV_SQRT2));
}

__host__ __device__ inline float
convert_fp8_to_float(fp8 x, __hip_fp8_interpretation_t interpret) {
  fp16 half_val = __hip_cvt_fp8_to_halfraw(x, interpret);
  return static_cast<float>(half_val);
}

__host__ __device__ inline float2 convert_fp8x2_to_float2(fp8x2 x,
    __hip_fp8_interpretation_t interpret) {
  __half2 half_val = __hip_cvt_fp8x2_to_halfraw2(x, interpret);
  return make_float2(static_cast<float>(half_val.x), static_cast<float>(half_val.y));
}

__host__ __device__ inline void unpack_fp8x2(const ushort packed,
    float& w0, float& w1)
{
  unsigned char fp8_0 = packed & 0xFF;
  unsigned char fp8_1 = (packed >> 8) & 0xFF;

  w0 = convert_fp8_to_float(fp8_0, __HIP_E4M3_FNUZ);
  w1 = convert_fp8_to_float(fp8_1, __HIP_E4M3_FNUZ);
}

__host__ __device__ inline void unpack_fp8x4(const unsigned int packed,
    float& w0, float& w1, float& w2, float& w3)
{
  unsigned char fp8_0 = packed & 0xFF;
  unsigned char fp8_1 = (packed >> 8) & 0xFF;
  unsigned char fp8_2 = (packed >> 16) & 0xFF;
  unsigned char fp8_3 = (packed >> 24) & 0xFF;

  w0 = convert_fp8_to_float(fp8_0, __HIP_E4M3_FNUZ);
  w1 = convert_fp8_to_float(fp8_1, __HIP_E4M3_FNUZ);
  w2 = convert_fp8_to_float(fp8_2, __HIP_E4M3_FNUZ);
  w3 = convert_fp8_to_float(fp8_3, __HIP_E4M3_FNUZ);
}

__host__ __device__ inline void unpack_fp8x4(const unsigned int packed,
  bfp16& w0, bfp16& w1, bfp16& w2, bfp16& w3)
{
  unsigned char fp8_0 = packed & 0xFF;
  unsigned char fp8_1 = (packed >> 8) & 0xFF;
  unsigned char fp8_2 = (packed >> 16) & 0xFF;
  unsigned char fp8_3 = (packed >> 24) & 0xFF;

  w0 = __float2bfloat16(convert_fp8_to_float(fp8_0, __HIP_E4M3_FNUZ));
  w1 = __float2bfloat16(convert_fp8_to_float(fp8_1, __HIP_E4M3_FNUZ));
  w2 = __float2bfloat16(convert_fp8_to_float(fp8_2, __HIP_E4M3_FNUZ));
  w3 = __float2bfloat16(convert_fp8_to_float(fp8_3, __HIP_E4M3_FNUZ));
}

// Helper function to perform atomic add on hip_bfloat16
__device__ inline void atomic_add_g(ushort* addr, const float val)
{
    size_t offset    = reinterpret_cast<size_t>(addr) & 0x2;
    bool is_32_align = offset;
    uint32_t* addr_as_uint32_t =
        reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(addr) - offset);
    uint32_t current = *addr_as_uint32_t;

    uint32_t expected;

    do
    {
        expected              = current;
        ushort current_ushort = is_32_align ? current >> 16 : current & 0xffff;

        float next_float   = __uint_as_float(static_cast<uint32_t>(current_ushort) << 16) + val;
        ushort next_ushort = static_cast<ushort>(__float_as_uint(next_float) >> 16);
        uint32_t next      = is_32_align ? (current & 0xffff) | (next_ushort << 16)
                                         : (current & 0xffff0000) | next_ushort;

        current = atomicCAS(addr_as_uint32_t, expected, next);
    } while(current != expected);
}

// Copy from /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h
__device__ inline unsigned short float_2_bfloatraw(float f) {
#if HIP_BF16_AVX512_OP
  union {
    __bf16 bf16;
    unsigned short us;
  } u = {_mm_cvtness_sbh(f)};
  return u.us;
#else
  union {
    float fp32;
    unsigned int u32;
  } u = {f};
  if (~u.u32 & 0x7f800000) {
    // When the exponent bits are not all 1s, then the value is zero, normal,
    // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
    // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
    // This causes the bfloat16's mantissa to be incremented by 1 if the 16
    // least significant bits of the float mantissa are greater than 0x8000,
    // or if they are equal to 0x8000 and the least significant bit of the
    // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
    // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
    // has the value 0x7f, then incrementing it causes it to become 0x00 and
    // the exponent is incremented by one, which is the next higher FP value
    // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
    // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
    // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
    // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
    // incrementing it causes it to become an exponent of 0xFF and a mantissa
    // of 0x00, which is Inf, the next higher value to the unrounded value.
    u.u32 += 0x7fff + ((u.u32 >> 16) & 1); // Round to nearest, round to even
  } else if (u.u32 & 0xffff) {
    // When all of the exponent bits are 1, the value is Inf or NaN.
    // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
    // mantissa bit. Quiet NaN is indicated by the most significant mantissa
    // bit being 1. Signaling NaN is indicated by the most significant
    // mantissa bit being 0 but some other bit(s) being 1. If any of the
    // lower 16 bits of the mantissa are 1, we set the least significant bit
    // of the bfloat16 mantissa, in order to preserve signaling NaN in case
    // the bloat16's mantissa bits are all 0.
    u.u32 |= 0x10000; // Preserve signaling NaN
  }
  return static_cast<unsigned short>(u.u32 >> 16);
#endif
}

// Copy from gcnasm/bandwidth_memread
template<bool USE_NTLOAD, typename T>
__device__ __forceinline__ T nt_load(const T& ref)
{
  if (USE_NTLOAD) {
    return __builtin_nontemporal_load(&ref);
  } else {
    return ref;
  }
}

template <typename T>
__device__ void atomic_add_pair(void* addr, float val1, float val2) 
{
    if constexpr (std::is_same<T, bfp16>::value) {
      bfp16x2 packed(val1, val2);
      __builtin_amdgcn_global_atomic_fadd_v2bf16(reinterpret_cast<bfp16x2*>(addr), packed);
    } else if constexpr (std::is_same<T, fp16>::value) {
      // TODO(anhduong): Test fp16 atomic add
      // fp16x2 packed = __halves2half2(__float2half(val1), __float2half(val2));
      // __builtin_amdgcn_global_atomic_fadd_v2f16(reinterpret_cast<fp16x2*>(addr), packed);

      // NOTE(anhduong): Temporary fix using union
      typedef _Float16 __attribute__((ext_vector_type(2))) vec_fp162;
      static_assert(sizeof(vec_fp162) == sizeof(__half2_raw));
      union {
        __half2_raw h2r;
        vec_fp162 fp16;
      } u{static_cast<__half2_raw>(__floats2half2_rn(val1, val2))};
      __builtin_amdgcn_flat_atomic_fadd_v2f16((vec_fp162*)addr, u.fp16);
    } else if constexpr (std::is_same<T, float>::value) {
      atomicAdd(reinterpret_cast<float *>(addr), val1);
      atomicAdd(reinterpret_cast<float *>(addr) + 1, val2);
    } 
}
