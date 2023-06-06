#include "shared_kernel.h"

namespace cudabm {

template <typename T>
__global__ void shortKernel(T* out_d, T* in_d, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out_d[idx] = 1.23 * in_d[idx];
}

template __global__ void shortKernel<float>(float* out_d, float* in_d, int N);
}  // namespace cudabm
