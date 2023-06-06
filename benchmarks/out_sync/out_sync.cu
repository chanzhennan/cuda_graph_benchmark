#include "../bm_lib/shared_kernel.h"
#include "baseline/baseline.cuh"
// #define N 500000 // tuned such that kernel takes a few microseconds
#define NSTEP 1000
#define NKERNEL 20

// PRE:
// out_d
// in_d
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
template <typename T>
void testAdd2(T *in_d, T *out_d, size_t N, cudaStream_t &stream) {
  int TPB = 256;
  int blocks = (TPB + N - 1) / TPB;

  // start CPU wallclock timer
  for (int istep = 0; istep < NSTEP; istep++) {
    for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
      cudabm::shortKernel<<<blocks, TPB, 0, stream>>>(in_d, out_d, N);
    }
    cudaStreamSynchronize(stream);
  }
}

template void testAdd2<float>(float *in_d, float *out_d, size_t N,
                              cudaStream_t &stream);
