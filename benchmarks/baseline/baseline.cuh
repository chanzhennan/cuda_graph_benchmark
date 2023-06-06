#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

template <typename T>
void testAdd(T *in_d, T *out_d, size_t N, cudaStream_t &stream);
