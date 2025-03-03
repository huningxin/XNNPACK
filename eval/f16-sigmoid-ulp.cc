// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <cpuinfo.h>
#include <pthreadpool.h>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>

#include "bench/utils.h"
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


struct ComputeErrorContext {
  const uint16_t* input;
  const uint16_t* output;
  float* error;
};

static void ComputeError(
  struct ComputeErrorContext* context,
  size_t start,
  size_t range)
{
  const uint16_t* input = context->input;
  const uint16_t* output = context->output;
  float* error = context->error;
  for (size_t i = start; i < start + range; i++) {
    const float input_val = fp16_ieee_to_fp32_value(input[i]);
    float output_ref = 0.0f;
    if (input_val < 0.0f) {
      const float exp_val = std::exp(input_val);
      output_ref = exp_val / (1.0f + exp_val);
    } else {
      output_ref = 1.0f / (1.0f + std::exp(-input_val));
    }
    const float abs_error = std::abs(output_ref - fp16_ieee_to_fp32_value(output[i]));
    const uint16_t output_abs = fp16_ieee_from_fp32_value(std::abs(output_ref));
    const float output_ulp = fp16_ieee_to_fp32_value(output_abs + 1) - fp16_ieee_to_fp32_value(output_abs);
    error[i] = float(abs_error / output_ulp);
  }
}

static void SigmoidError(benchmark::State& state,
  xnn_f16_unary_math_function sigmoid,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("failed cpuinfo init");
    return;
  }
  if (isa_check && !isa_check(state)) {
    return;
  }

  // The smallest x for which sigmoidf(x) is normalized (-0x1.368p+3h).
  const uint16_t min_input = UINT16_C(0xC8DA);
  // The largest x for which sigmoidf(x) is not 1.0f (0x1.0A0p3h).
  const uint16_t max_input = UINT16_C(0x4828);
  // Number of elements in one block of inputs/outputs.
  // Combining multiple elements in a block reduce function call overhead.
  const size_t block_size = 16384;
  // Number of elements in one parallelization tile. Worker threads process this many elements in each task.
  const size_t tile_size = 64;

  uint32_t num_threads = cpuinfo_get_cores_count();
  #if XNN_ARCH_ARM || XNN_ARCH_ARM64
    // Use all cores except for the least performant cluster
    if (cpuinfo_get_clusters_count() > 1) {
      num_threads -= cpuinfo_get_cluster(cpuinfo_get_clusters_count() - 1)->core_count;
    }
  #endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> x(block_size);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> y(block_size);
  std::vector<float> ulp_error(block_size);
  float max_ulp_error = 0.0f;

  ComputeErrorContext context;
  context.input = x.data();
  context.output = y.data();
  context.error = ulp_error.data();
  for (auto _ : state) {
    for (uint16_t n = min_input; int16_t(n) < 0; n -= block_size) {
      for (uint16_t i = 0; i < block_size; i++) {
        x[i] = std::max<uint16_t>(n - i, UINT16_C(0x8000));
      }
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);

      sigmoid(block_size * sizeof(uint16_t), x.data(), y.data());

      pthreadpool_parallelize_1d_tile_1d(
          threadpool.get(),
          reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(ComputeError),
          static_cast<void*>(&context),
          block_size, tile_size, 0 /* flags */);

      max_ulp_error = std::accumulate(ulp_error.cbegin(), ulp_error.cend(), max_ulp_error,
        static_cast<const float& (*)(const float&, const float&)>(std::max<float>));
    }
    for (uint16_t n = 0; n < max_input; n += block_size) {
      for (uint16_t i = 0; i < block_size; i++) {
        x[i] = std::min<uint16_t>(n + i, max_input);
      }
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);

      sigmoid(block_size * sizeof(uint16_t), x.data(), y.data());

      pthreadpool_parallelize_1d_tile_1d(
          threadpool.get(),
          reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(ComputeError),
          static_cast<void*>(&context),
          block_size, tile_size, 0 /* flags */);

      max_ulp_error = std::accumulate(ulp_error.cbegin(), ulp_error.cend(), max_ulp_error,
        static_cast<const float& (*)(const float&, const float&)>(std::max<float>));
    }
  }

  state.counters["ULPERROR"] = benchmark::Counter(max_ulp_error);
}

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr1_p2_div,
                    xnn_math_f16_sigmoid__neonfp16arith_rr1_p2_div,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr1_p3_div,
                    xnn_math_f16_sigmoid__neonfp16arith_rr1_p3_div,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr2_p2_div,
                    xnn_math_f16_sigmoid__neonfp16arith_rr2_p2_div,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr2_p2_nr1fma,
                    xnn_math_f16_sigmoid__neonfp16arith_rr2_p2_nr1fma,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr2_p2_nr1recps,
                    xnn_math_f16_sigmoid__neonfp16arith_rr2_p2_nr1recps,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr2_p2_recpe,
                    xnn_math_f16_sigmoid__neonfp16arith_rr2_p2_recpe,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr2_p3_div,
                    xnn_math_f16_sigmoid__neonfp16arith_rr2_p3_div,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr2_p3_nr1fma,
                    xnn_math_f16_sigmoid__neonfp16arith_rr2_p3_nr1fma,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr2_p3_nr1recps,
                    xnn_math_f16_sigmoid__neonfp16arith_rr2_p3_nr1recps,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, neonfp16arith_rr2_p3_recpe,
                    xnn_math_f16_sigmoid__neonfp16arith_rr2_p3_recpe,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_p2_div,
                    xnn_math_f16_sigmoid__avx2_rr1_p2_div,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_p2_rcp,
                    xnn_math_f16_sigmoid__avx2_rr1_p2_rcp,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_p3_div,
                    xnn_math_f16_sigmoid__avx2_rr1_p3_div,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SigmoidError, avx2_rr1_p3_rcp,
                    xnn_math_f16_sigmoid__avx2_rr1_p3_rcp,
                    benchmark::utils::CheckAVX2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
