// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c4-avx512vnni.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>


void xnn_qs8_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __mmask16 vinput_mask = _cvtu32_mask16(0xF);
  const __m512i vsaturate_min = _mm512_set1_epi32(-128);
  const __m512i vsaturate_max = _mm512_set1_epi32(127);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i v128 = _mm512_set1_epi8(128);
  const __m512i voutput_zero_point = _mm512_cvtepi16_epi32(_mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point));
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_load_epi32(w);
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc2x0123 = vacc0x0123;
    __m512i vacc3x0123 = vacc0x0123;
    w = (const int32_t*) w + 16;

    size_t k = 0;
    while (k < kc) {
      const __m128i va0slo = _mm_maskz_loadu_epi8(vinput_mask, a0);
      const __m128i va1slo = _mm_maskz_loadu_epi8(vinput_mask, a1);
      const __m128i va2slo = _mm_maskz_loadu_epi8(vinput_mask, a2);
      const __m128i va3slo = _mm_maskz_loadu_epi8(vinput_mask, a3);

      const __m128i va0s = _mm_shuffle_epi32(va0slo, _MM_SHUFFLE(0,0,0,0));
      const __m128i va1s = _mm_shuffle_epi32(va1slo, _MM_SHUFFLE(0,0,0,0));
      const __m128i va2s = _mm_shuffle_epi32(va2slo, _MM_SHUFFLE(0,0,0,0));
      const __m128i va3s = _mm_shuffle_epi32(va3slo, _MM_SHUFFLE(0,0,0,0));

      const __m512i va0x0123s = _mm512_broadcast_i32x4(va0s);
      const __m512i va1x0123s = _mm512_broadcast_i32x4(va1s);
      const __m512i va2x0123s = _mm512_broadcast_i32x4(va2s);
      const __m512i va3x0123s = _mm512_broadcast_i32x4(va3s);

      const __m512i va0x0123 = _mm512_xor_epi32(va0x0123s, v128);
      const __m512i va1x0123 = _mm512_xor_epi32(va1x0123s, v128);
      const __m512i va2x0123 = _mm512_xor_epi32(va2x0123s, v128);
      const __m512i va3x0123 = _mm512_xor_epi32(va3x0123s, v128);

      a0 += 4;
      a1 += 4;
      a2 += 4;
      a3 += 4;

      const __m512i vb0123 = _mm512_load_si512(w);

      vacc0x0123 = _mm512_dpbusd_epi32(vacc0x0123, va0x0123, vb0123);
      vacc1x0123 = _mm512_dpbusd_epi32(vacc1x0123, va1x0123, vb0123);
      vacc2x0123 = _mm512_dpbusd_epi32(vacc2x0123, va2x0123, vb0123);
      vacc3x0123 = _mm512_dpbusd_epi32(vacc3x0123, va3x0123, vb0123);

      w = (const int8_t*) w + 64;
      k += 4 * sizeof(int8_t);
    }

    __m512 vscaled0x0123 = _mm512_cvtepi32_ps(vacc0x0123);
    __m512 vscaled1x0123 = _mm512_cvtepi32_ps(vacc1x0123);
    __m512 vscaled2x0123 = _mm512_cvtepi32_ps(vacc2x0123);
    __m512 vscaled3x0123 = _mm512_cvtepi32_ps(vacc3x0123);

    vscaled0x0123 = _mm512_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm512_mul_ps(vscaled1x0123, vscale);
    vscaled2x0123 = _mm512_mul_ps(vscaled2x0123, vscale);
    vscaled3x0123 = _mm512_mul_ps(vscaled3x0123, vscale);

    vscaled0x0123 = _mm512_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm512_min_ps(vscaled1x0123, voutput_max_less_zero_point);
    vscaled2x0123 = _mm512_min_ps(vscaled2x0123, voutput_max_less_zero_point);
    vscaled3x0123 = _mm512_min_ps(vscaled3x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm512_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm512_cvtps_epi32(vscaled1x0123);
    vacc2x0123 = _mm512_cvtps_epi32(vscaled2x0123);
    vacc3x0123 = _mm512_cvtps_epi32(vscaled3x0123);

    vacc0x0123 = _mm512_add_epi32(vacc0x0123, voutput_zero_point);
    vacc1x0123 = _mm512_add_epi32(vacc1x0123, voutput_zero_point);
    vacc2x0123 = _mm512_add_epi32(vacc2x0123, voutput_zero_point);
    vacc3x0123 = _mm512_add_epi32(vacc3x0123, voutput_zero_point);

    vacc0x0123 = _mm512_max_epi32(vacc0x0123, vsaturate_min);
    vacc1x0123 = _mm512_max_epi32(vacc1x0123, vsaturate_min);
    vacc2x0123 = _mm512_max_epi32(vacc2x0123, vsaturate_min);
    vacc3x0123 = _mm512_max_epi32(vacc3x0123, vsaturate_min);

    vacc0x0123 = _mm512_min_epi32(vacc0x0123, vsaturate_max);
    vacc1x0123 = _mm512_min_epi32(vacc1x0123, vsaturate_max);
    vacc2x0123 = _mm512_min_epi32(vacc2x0123, vsaturate_max);
    vacc3x0123 = _mm512_min_epi32(vacc3x0123, vsaturate_max);

    __m128i vout0x0123 = _mm512_cvtepi32_epi8(vacc0x0123);
    __m128i vout1x0123 = _mm512_cvtepi32_epi8(vacc1x0123);
    __m128i vout2x0123 = _mm512_cvtepi32_epi8(vacc2x0123);
    __m128i vout3x0123 = _mm512_cvtepi32_epi8(vacc3x0123);

    vout0x0123 = _mm_max_epi8(vout0x0123, voutput_min);
    vout1x0123 = _mm_max_epi8(vout1x0123, voutput_min);
    vout2x0123 = _mm_max_epi8(vout2x0123, voutput_min);
    vout3x0123 = _mm_max_epi8(vout3x0123, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123);
      _mm_storeu_si128((__m128i*) c1, vout1x0123);
      _mm_storeu_si128((__m128i*) c2, vout2x0123);
      _mm_storeu_si128((__m128i*) c3, vout3x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - k);
      a1 = (const int8_t*) ((uintptr_t) a1 - k);
      a2 = (const int8_t*) ((uintptr_t) a2 - k);
      a3 = (const int8_t*) ((uintptr_t) a3 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123);
      _mm_mask_storeu_epi8(c1, vmask, vout1x0123);
      _mm_mask_storeu_epi8(c2, vmask, vout2x0123);
      _mm_mask_storeu_epi8(c3, vmask, vout3x0123);

      nc = 0;
    }
  } while (nc != 0);
}
