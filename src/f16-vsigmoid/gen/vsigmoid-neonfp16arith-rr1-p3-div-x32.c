// Auto-generated file. Do not edit!
//   Template: src/f16-vsigmoid/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f16_vsigmoid_ukernel__neonfp16arith_rr1_p3_div_x32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch % sizeof(__fp16) == 0);

  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith_rr1_p3.magic_bias));
  const float16x8_t vminus_log2e = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith_rr1_p3.minus_log2e));
  const float16x8_t vln2 = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith_rr1_p3.ln2));
  const float16x8_t vc3 = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith_rr1_p3.c3));
  const float16x8_t vc2 = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith_rr1_p3.c2));
  const float16x8_t vone = vmovq_n_f16(1.0f);
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith_rr1_p3.denorm_cutoff));

  const __fp16* i = (const __fp16*) input;
  __fp16* o = (__fp16*) output;
  for (; batch >= 32 * sizeof(__fp16); batch -= 32 * sizeof(__fp16)) {
    const float16x8_t vx0 = vld1q_f16(i); i += 8;
    const float16x8_t vx1 = vld1q_f16(i); i += 8;
    const float16x8_t vx2 = vld1q_f16(i); i += 8;
    const float16x8_t vx3 = vld1q_f16(i); i += 8;

    const float16x8_t vz0 = vabsq_f16(vx0);
    const float16x8_t vz1 = vabsq_f16(vx1);
    const float16x8_t vz2 = vabsq_f16(vx2);
    const float16x8_t vz3 = vabsq_f16(vx3);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vz0, vminus_log2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vz1, vminus_log2e);
    float16x8_t vn2 = vfmaq_f16(vmagic_bias, vz2, vminus_log2e);
    float16x8_t vn3 = vfmaq_f16(vmagic_bias, vz3, vminus_log2e);

    const float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    const float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));
    const float16x8_t vs2 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn2), 10));
    const float16x8_t vs3 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn3), 10));

    vn0 = vsubq_f16(vn0, vmagic_bias);
    vn1 = vsubq_f16(vn1, vmagic_bias);
    vn2 = vsubq_f16(vn2, vmagic_bias);
    vn3 = vsubq_f16(vn3, vmagic_bias);

    float16x8_t vt0 = vfmaq_f16(vz0, vn0, vln2);
    float16x8_t vt1 = vfmaq_f16(vz1, vn1, vln2);
    float16x8_t vt2 = vfmaq_f16(vz2, vn2, vln2);
    float16x8_t vt3 = vfmaq_f16(vz3, vn3, vln2);

    float16x8_t vp0 = vfmaq_f16(vc2, vc3, vt0);
    float16x8_t vp1 = vfmaq_f16(vc2, vc3, vt1);
    float16x8_t vp2 = vfmaq_f16(vc2, vc3, vt2);
    float16x8_t vp3 = vfmaq_f16(vc2, vc3, vt3);

    vp0 = vfmsq_f16(vone, vp0, vt0);
    vp1 = vfmsq_f16(vone, vp1, vt1);
    vp2 = vfmsq_f16(vone, vp2, vt2);
    vp3 = vfmsq_f16(vone, vp3, vt3);

    vt0 = vmulq_f16(vt0, vs0);
    vt1 = vmulq_f16(vt1, vs1);
    vt2 = vmulq_f16(vt2, vs2);
    vt3 = vmulq_f16(vt3, vs3);

    const float16x8_t ve0 = vfmsq_f16(vs0, vp0, vt0);
    const float16x8_t ve1 = vfmsq_f16(vs1, vp1, vt1);
    const float16x8_t ve2 = vfmsq_f16(vs2, vp2, vt2);
    const float16x8_t ve3 = vfmsq_f16(vs3, vp3, vt3);

    const float16x8_t vd0 = vaddq_f16(ve0, vone);
    const float16x8_t vd1 = vaddq_f16(ve1, vone);
    const float16x8_t vd2 = vaddq_f16(ve2, vone);
    const float16x8_t vd3 = vaddq_f16(ve3, vone);

    float16x8_t vf0 = vdivq_f16(ve0, vd0);
    float16x8_t vf1 = vdivq_f16(ve1, vd1);
    float16x8_t vf2 = vdivq_f16(ve2, vd2);
    float16x8_t vf3 = vdivq_f16(ve3, vd3);

    vf0 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf0), vcagtq_f16(vx0, vdenorm_cutoff)));
    vf1 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf1), vcagtq_f16(vx1, vdenorm_cutoff)));
    vf2 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf2), vcagtq_f16(vx2, vdenorm_cutoff)));
    vf3 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf3), vcagtq_f16(vx3, vdenorm_cutoff)));

    const uint16x8_t vm0 = vcltq_f16(vx0, vmovq_n_f16(0.0f));
    const uint16x8_t vm1 = vcltq_f16(vx1, vmovq_n_f16(0.0f));
    const uint16x8_t vm2 = vcltq_f16(vx2, vmovq_n_f16(0.0f));
    const uint16x8_t vm3 = vcltq_f16(vx3, vmovq_n_f16(0.0f));

    vf0 = vbslq_f16(vm0, vf0, vsubq_f16(vone, vf0));
    vf1 = vbslq_f16(vm1, vf1, vsubq_f16(vone, vf1));
    vf2 = vbslq_f16(vm2, vf2, vsubq_f16(vone, vf2));
    vf3 = vbslq_f16(vm3, vf3, vsubq_f16(vone, vf3));

    vst1q_f16(o, vf0); o += 8;
    vst1q_f16(o, vf1); o += 8;
    vst1q_f16(o, vf2); o += 8;
    vst1q_f16(o, vf3); o += 8;
  }
  for (; batch >= 8 * sizeof(__fp16); batch -= 8 * sizeof(__fp16)) {
    const float16x8_t vx = vld1q_f16(i); i += 8;

    const float16x8_t vz = vabsq_f16(vx);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);
    float16x8_t vt = vfmaq_f16(vz, vn, vln2);

    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vfmsq_f16(vone, vp, vt);

    vt = vmulq_f16(vt, vs);
    const float16x8_t ve = vfmsq_f16(vs, vp, vt);
    const float16x8_t vd = vaddq_f16(ve, vone);

    float16x8_t vf = vdivq_f16(ve, vd);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vmovq_n_f16(0.0f));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    vst1q_f16(o, vf); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vld1q_f16(i);

    const float16x8_t vz = vabsq_f16(vx);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);
    float16x8_t vt = vfmaq_f16(vz, vn, vln2);

    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vfmsq_f16(vone, vp, vt);

    vt = vmulq_f16(vt, vs);
    const float16x8_t ve = vfmsq_f16(vs, vp, vt);
    const float16x8_t vd = vaddq_f16(ve, vone);

    float16x8_t vf = vdivq_f16(ve, vd);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vmovq_n_f16(0.0f));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    float16x4_t vf_lo = vget_low_f16(vf);
    if (batch & (4 * sizeof(__fp16))) {
      vst1_f16(o, vf_lo); o += 4;
      vf_lo = vget_high_f16(vf);
    }
    if (batch & (2 * sizeof(__fp16))) {
      vst1_f16(o, vf_lo); o += 2;
      vf_lo = vext_f16(vf_lo, vf_lo, 2);
    }
    if (batch & (1 * sizeof(__fp16))) {
      vst1_lane_f16(o, vf_lo, 0);
    }
  }
}