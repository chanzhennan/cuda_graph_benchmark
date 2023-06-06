#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cudabm {

template <typename T>
__global__ void shortKernel(T* out_d, T* in_d, int N);

}
