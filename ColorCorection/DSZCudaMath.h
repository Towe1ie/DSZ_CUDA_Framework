#ifndef _DSZ_CUDA_MATH_H_
#define _DSZ_CUDA_MATH_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__
float sum(float a, float b);

__device__
float d_min(float a, float b);

__device__
float d_max(float a, float b);

__device__
float cudaClamp(float x, float min, float max);

#endif