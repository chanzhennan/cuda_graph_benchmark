// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include "baseline/baseline.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/utils.h"

template <typename T>
class Baseline : public benchmark::Fixture {
 public:
  void callKernel(benchmark::State &state) {
    // call kernel
    testAdd(d_array, d_result, dataSize, stream);
    cudaMemcpy(result, d_result, sizeof(T) * dataSize, cudaMemcpyDeviceToHost);
  }

  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    dataSize = state.range(0) * state.range(0);

    // malloc array
    cudaMallocHost(&array, sizeof(T) * dataSize);
    cudaMallocHost(&result, sizeof(T) * dataSize);
    cudaMalloc(&d_array, sizeof(T) * dataSize);
    cudaMalloc(&d_result, sizeof(T) * dataSize);

    // gen random
    cudabm::genRandom(array, dataSize);

    cudaMemcpy(d_array, array, sizeof(T) * dataSize, cudaMemcpyHostToDevice);
    cudaStreamCreate(&stream);
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    cudaFree(d_array);
    cudaFree(d_result);
    cudaFreeHost(array);
    cudaFreeHost(result);

    cudaStreamDestroy(stream);
  }

  double getDataSize() { return (double)dataSize; }

 private:
  T *d_array, *array;
  T *d_result, *result;
  long int dataSize;

  cudaStream_t stream;
};

#define BENCHMARK_GRAPH1_OP(name, dType)                               \
  BENCHMARK_TEMPLATE_DEFINE_F(Baseline, name, dType)                   \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize();                           \
    st.counters["FLOPS"] = benchmark::Counter{                         \
        getDataSize(), benchmark::Counter::kIsIterationInvariantRate}; \
  }                                                                    \
  BENCHMARK_REGISTER_F(Baseline, name)                                 \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Range(512, 1024);

#define BENCHMARK_GRAPH1_OP_TYPE(dType) \
  BENCHMARK_GRAPH1_OP(Graph_##dType, dType)

BENCHMARK_GRAPH1_OP_TYPE(float)
