#ifndef _DSZ_CUDA_UTILITIES_H_
#define _DSZ_CUDA_UTILITIES_H_

#include <cuda_runtime.h>

__device__
void __syncthreads();

void checkError(cudaError_t err);
void queryDevices();

// **** Performance mesuring functions
typedef struct
{
	cudaEvent_t start, stop;
} dszCudaMesureInfo_t;
dszCudaMesureInfo_t startMesuring();
float stopMesuring(dszCudaMesureInfo_t mesureInfo);

#endif