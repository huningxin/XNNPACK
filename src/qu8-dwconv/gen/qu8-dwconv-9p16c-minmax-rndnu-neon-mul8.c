// Auto-generated file. Do not edit!
//   Template: src/qu8-dwconv/unipass-neon-mul8.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>


void xnn_qu8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const uint16x8_t vkernel_zero_point = vmovl_u8(vld1_dup_u8(params->rndnu_neon.kernel_zero_point));
  const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
  const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
  const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->rndnu_neon.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->rndnu_neon.output_max);
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);


    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vacc4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vacc89AB = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vaccCDEF = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);


      const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vk0x01234567 = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);
      const uint8x8_t vi0x89ABCDEF = vld1_u8(i0); i0 += 8;
      const uint8x8_t vk0x89ABCDEF = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);

      uint16x8_t vprod01234567 = vmull_u8(vi0x01234567, vk0x01234567);
      uint16x8_t vprod89ABCDEF = vmull_u8(vi0x89ABCDEF, vk0x89ABCDEF);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vk1x01234567 = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);
      const uint8x8_t vi1x89ABCDEF = vld1_u8(i1); i1 += 8;
      const uint8x8_t vk1x89ABCDEF = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);

      vprod01234567 = vmull_u8(vi1x01234567, vk1x01234567);
      uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);
      vprod89ABCDEF = vmull_u8(vi1x89ABCDEF, vk1x89ABCDEF);
      uint16x8_t vsum89ABCDEF = vaddl_u8(vi0x89ABCDEF, vi1x89ABCDEF);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
      const uint8x8_t vk2x01234567 = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);
      const uint8x8_t vi2x89ABCDEF = vld1_u8(i2); i2 += 8;
      const uint8x8_t vk2x89ABCDEF = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);

      vprod01234567 = vmull_u8(vi2x01234567, vk2x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
      vprod89ABCDEF = vmull_u8(vi2x89ABCDEF, vk2x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi2x89ABCDEF);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
      const uint8x8_t vk3x01234567 = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);
      const uint8x8_t vi3x89ABCDEF = vld1_u8(i3); i3 += 8;
      const uint8x8_t vk3x89ABCDEF = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);

      vprod01234567 = vmull_u8(vi3x01234567, vk3x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
      vprod89ABCDEF = vmull_u8(vi3x89ABCDEF, vk3x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi3x89ABCDEF);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
      const uint8x8_t vk4x01234567 = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);
      const uint8x8_t vi4x89ABCDEF = vld1_u8(i4); i4 += 8;
      const uint8x8_t vk4x89ABCDEF = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);

      vprod01234567 = vmull_u8(vi4x01234567, vk4x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
      vprod89ABCDEF = vmull_u8(vi4x89ABCDEF, vk4x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi4x89ABCDEF);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
      const uint8x8_t vk5x01234567 = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);
      const uint8x8_t vi5x89ABCDEF = vld1_u8(i5); i5 += 8;
      const uint8x8_t vk5x89ABCDEF = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);

      vprod01234567 = vmull_u8(vi5x01234567, vk5x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
      vprod89ABCDEF = vmull_u8(vi5x89ABCDEF, vk5x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi5x89ABCDEF);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
      const uint8x8_t vk6x01234567 = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);
      const uint8x8_t vi6x89ABCDEF = vld1_u8(i6); i6 += 8;
      const uint8x8_t vk6x89ABCDEF = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);

      vprod01234567 = vmull_u8(vi6x01234567, vk6x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);
      vprod89ABCDEF = vmull_u8(vi6x89ABCDEF, vk6x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi6x89ABCDEF);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      const uint8x8_t vi7x01234567 = vld1_u8(i7); i7 += 8;
      const uint8x8_t vk7x01234567 = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);
      const uint8x8_t vi7x89ABCDEF = vld1_u8(i7); i7 += 8;
      const uint8x8_t vk7x89ABCDEF = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);

      vprod01234567 = vmull_u8(vi7x01234567, vk7x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi7x01234567);
      vprod89ABCDEF = vmull_u8(vi7x89ABCDEF, vk7x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi7x89ABCDEF);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      const uint8x8_t vi8x01234567 = vld1_u8(i8); i8 += 8;
      const uint8x8_t vk8x01234567 = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);
      const uint8x8_t vi8x89ABCDEF = vld1_u8(i8); i8 += 8;
      const uint8x8_t vk8x89ABCDEF = vld1_u8(w); w = (const void*) ((const int8_t*) w + 8);

      vprod01234567 = vmull_u8(vi8x01234567, vk8x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi8x01234567);
      vprod89ABCDEF = vmull_u8(vi8x89ABCDEF, vk8x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi8x89ABCDEF);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));

      vacc0123 = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567), vget_low_u16(vkernel_zero_point)));
      vacc4567 = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567), vget_high_u16(vkernel_zero_point)));
      vacc89AB = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vsum89ABCDEF), vget_low_u16(vkernel_zero_point)));
      vaccCDEF = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vsum89ABCDEF), vget_high_u16(vkernel_zero_point)));

      vacc0123 = vshlq_s32(vacc0123, vright_pre_shift);
      vacc4567 = vshlq_s32(vacc4567, vright_pre_shift);
      vacc89AB = vshlq_s32(vacc89AB, vright_pre_shift);
      vaccCDEF = vshlq_s32(vaccCDEF, vright_pre_shift);

      vacc0123 = vqdmulhq_s32(vacc0123, vmultiplier);
      vacc4567 = vqdmulhq_s32(vacc4567, vmultiplier);
      vacc89AB = vqdmulhq_s32(vacc89AB, vmultiplier);
      vaccCDEF = vqdmulhq_s32(vaccCDEF, vmultiplier);

      vacc0123 = vrshlq_s32(vacc0123, vright_post_shift);
      vacc4567 = vrshlq_s32(vacc4567, vright_post_shift);
      vacc89AB = vrshlq_s32(vacc89AB, vright_post_shift);
      vaccCDEF = vrshlq_s32(vaccCDEF, vright_post_shift);

#if XNN_ARCH_ARM64
      const int16x8_t vacc01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567), voutput_zero_point);
      const int16x8_t vacc89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF), voutput_zero_point);

      uint8x16_t vout0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc01234567), vacc89ABCDEF);
#else
      const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);
      const int16x8_t vacc89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF)), voutput_zero_point);

      uint8x16_t vout0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
#endif

      vout0123456789ABCDEF = vmaxq_u8(vout0123456789ABCDEF, voutput_min);

      vout0123456789ABCDEF = vminq_u8(vout0123456789ABCDEF, voutput_max);

      vst1q_u8(output, vout0123456789ABCDEF); output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8_t* k = (const uint8_t*) ((const int32_t*) w + 16);
      do {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
        int32x4_t vacc4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);

        const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
        const uint8x8_t vk0x01234567 = vld1_u8(k); k += 8;

        uint16x8_t vprod01234567 = vmull_u8(vi0x01234567, vk0x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
        const uint8x8_t vk1x01234567 = vld1_u8((const void*) (k + 8));

        vprod01234567 = vmull_u8(vi1x01234567, vk1x01234567);
        uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
        const uint8x8_t vk2x01234567 = vld1_u8((const void*) (k + 24));

        vprod01234567 = vmull_u8(vi2x01234567, vk2x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
        const uint8x8_t vk3x01234567 = vld1_u8((const void*) (k + 40));

        vprod01234567 = vmull_u8(vi3x01234567, vk3x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
        const uint8x8_t vk4x01234567 = vld1_u8((const void*) (k + 56));

        vprod01234567 = vmull_u8(vi4x01234567, vk4x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
        const uint8x8_t vk5x01234567 = vld1_u8((const void*) (k + 72));

        vprod01234567 = vmull_u8(vi5x01234567, vk5x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
        const uint8x8_t vk6x01234567 = vld1_u8((const void*) (k + 88));

        vprod01234567 = vmull_u8(vi6x01234567, vk6x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi7x01234567 = vld1_u8(i7); i7 += 8;
        const uint8x8_t vk7x01234567 = vld1_u8((const void*) (k + 104));

        vprod01234567 = vmull_u8(vi7x01234567, vk7x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi7x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi8x01234567 = vld1_u8(i8); i8 += 8;
        const uint8x8_t vk8x01234567 = vld1_u8((const void*) (k + 120));

        vprod01234567 = vmull_u8(vi8x01234567, vk8x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi8x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));

        vacc0123 = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567), vget_low_u16(vkernel_zero_point)));
        vacc4567 = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567), vget_high_u16(vkernel_zero_point)));

        vacc0123 = vrshlq_s32(vacc0123, vright_pre_shift);
        vacc4567 = vrshlq_s32(vacc4567, vright_pre_shift);

        vacc0123 = vqdmulhq_s32(vacc0123, vmultiplier);
        vacc4567 = vqdmulhq_s32(vacc4567, vmultiplier);

        vacc0123 = vrshlq_s32(vacc0123, vright_post_shift);
        vacc4567 = vrshlq_s32(vacc4567, vright_post_shift);

#if XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567), voutput_zero_point);
        uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
#else
        const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);
        uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
#endif

        vout01234567 = vmax_u8(vout01234567, vget_low_u8(voutput_min));
        vout01234567 = vmin_u8(vout01234567, vget_low_u8(voutput_max));

        if XNN_LIKELY(c >= 8) {
          vst1_u8(output, vout01234567); output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1_lane_u32((void*) output, vreinterpret_u32_u8(vout01234567), 0); output += 4;
            vout01234567 = vext_u8(vout01234567, vout01234567, 4);
          }
          if (c & 2) {
            vst1_lane_u16((void*) output, vreinterpret_u16_u8(vout01234567), 0); output += 2;
            vout01234567 = vext_u8(vout01234567, vout01234567, 2);
          }
          if (c & 1) {
            vst1_lane_u8(output, vout01234567, 0); output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
