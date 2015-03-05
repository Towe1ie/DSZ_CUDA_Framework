#include "PrefixSum.h"
#include "DSZCudaUtility.h"
#include <device_launch_parameters.h>
#include <float.h>
#include "DSZCudaMath.h"

__device__ op_pointer operations[] = { sum, d_min, d_max };
__device__ float op_defaults[] = { 0, FLT_MAX, FLT_MIN };

void sequential_scan(float *x, float *y, unsigned int inputSize)
{
	y[0] = x[0];

	for (unsigned int i = 1; i < inputSize; ++i)
		y[i] = y[i - 1] + x[i];
}

__global__
void add_hierarchy(float *y, float *s, unsigned int inputSize)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < inputSize && blockIdx.x != 0)
		y[i] += s[blockIdx.x - 1];
}

// **** WORK INEFFICIENT SCAN ****
__global__
void work_inefficient_scan_kernel(float *x, float *y, unsigned int inputSize, float *s)
{
	__shared__ float xy[SECTION_SIZE];

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < inputSize)
	{
		xy[threadIdx.x] = x[i];

		for (unsigned int stride = 1; stride <= threadIdx.x; stride <<= 1)
		{
			__syncthreads();
			xy[threadIdx.x] += xy[threadIdx.x - stride];
		}

		y[i] = xy[threadIdx.x];
		
		bool isLastBlock = blockIdx.x == (inputSize + blockDim.x - 1) / blockDim.x;
		unsigned int lastThreadIdx = (inputSize % blockDim.x - 1)*isLastBlock + blockDim.x * (!isLastBlock);

		if ((threadIdx.x == (lastThreadIdx - 1)) && s != nullptr)
			s[blockIdx.x] = xy[lastThreadIdx - 1];
	}
}

void work_inefficient_scan(float *x, float *y, unsigned int inputSize)
{
	size_t size = inputSize * sizeof(float);
	float *d_x, *d_y, *d_s, *d_s1;
	cudaError err;

	err = cudaMalloc(&d_x, size); checkError(err);
	err = cudaMalloc(&d_y, size); checkError(err);
	err = cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice); checkError(err);
	err = cudaMalloc(&d_s, sizeof(float)*(inputSize + SECTION_SIZE - 1) / SECTION_SIZE); checkError(err);
	err = cudaMalloc(&d_s1, sizeof(float)*(inputSize + SECTION_SIZE - 1) / SECTION_SIZE); checkError(err);

	unsigned int sectionsCount = (inputSize + SECTION_SIZE - 1) / SECTION_SIZE;
	dim3 gridDim(sectionsCount, 1, 1);
	dim3 blockDim(SECTION_SIZE, 1, 1);
	work_inefficient_scan_kernel<<<gridDim, blockDim>>>(d_x, d_y, inputSize, d_s);
	err = cudaGetLastError(); checkError(err);

	dim3 gridDim1(1, 1, 1);
	dim3 blockDim1(sectionsCount, 1, 1);
	work_inefficient_scan_kernel << <gridDim1, blockDim1 >> >(d_s, d_s1, sectionsCount, 0);
	err = cudaGetLastError(); checkError(err);

	add_hierarchy<<<gridDim, blockDim>>>(d_y, d_s1, inputSize);

	err = cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost); checkError(err);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_s);
	cudaFree(d_s1);
}

// **** WORK EFFICIENT SCAN ****
__global__
void work_efficient_scan_kernel(const float* __restrict__ x, float *y, unsigned int inputSize, float *s, op_t op)
{
	__shared__ float xy[SECTION_SIZE];

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < inputSize)
	{
		xy[threadIdx.x] = x[i];

		for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1)
		{
			__syncthreads();
			unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
			if (index < blockDim.x)
			{
				xy[index] = operations[op](xy[index], xy[index - stride]);
			}
		}

		for (unsigned int stride = blockDim.x / 4; stride > 0; stride >>= 1)
		{
			__syncthreads();
			unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
			if (index + stride < blockDim.x)
			{
				xy[index + stride] = operations[op](xy[index + stride], xy[index]);
			}
		}

		__syncthreads();
		y[i] = xy[threadIdx.x];

		bool isLastBlock = blockIdx.x == (inputSize + blockDim.x - 1) / blockDim.x;
		unsigned int lastThreadIdx = (inputSize % blockDim.x - 1)*isLastBlock + blockDim.x * (!isLastBlock);

		if ((threadIdx.x == (lastThreadIdx - 1)) && s != nullptr)
			s[blockIdx.x] = xy[lastThreadIdx - 1];
	}
}

void work_efficient_scan(float *x, float *y, unsigned int inputSize, op_t op)
{
	size_t size = inputSize * sizeof(float);
	float *d_x, *d_y, *d_s, *d_s1;;
	cudaError err;

	err = cudaMalloc(&d_x, size); checkError(err);
	err = cudaMalloc(&d_y, size); checkError(err);
	err = cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice); checkError(err);
	err = cudaMalloc(&d_s, sizeof(float)*(inputSize + SECTION_SIZE - 1) / SECTION_SIZE); checkError(err);
	err = cudaMalloc(&d_s1, sizeof(float)*(inputSize + SECTION_SIZE - 1) / SECTION_SIZE); checkError(err);

	unsigned int sectionsCount = (inputSize + SECTION_SIZE - 1) / SECTION_SIZE;
	dim3 gridDim(sectionsCount, 1, 1);
	dim3 blockDim(SECTION_SIZE, 1, 1);
	work_efficient_scan_kernel<<<gridDim, blockDim>>>(d_x, d_y, inputSize, d_s, op);

	dim3 gridDim1(1, 1, 1);
	dim3 blockDim1(sectionsCount, 1, 1);
	work_efficient_scan_kernel << <gridDim1, blockDim1 >> >(d_s, d_s1, sectionsCount, 0, op);
	err = cudaGetLastError(); checkError(err);

	add_hierarchy << <gridDim, blockDim >> >(d_y, d_s1, inputSize);
	err = cudaGetLastError(); checkError(err);

	err = cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost); checkError(err);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_s);
	cudaFree(d_s1);
}

// **** REDUCTION OP ****
#define REDUCTION_SECTION_SIZE 2048
__global__
void reduction_op_kernel(float *x, float *out, unsigned int inputSize, op_t op)
{
	__shared__ float partial_op[REDUCTION_SECTION_SIZE];
	
	unsigned int i = blockIdx.x * REDUCTION_SECTION_SIZE + threadIdx.x;
	unsigned int tx = threadIdx.x;
	if (i < inputSize)
		partial_op[tx] = x[i];
	else
		partial_op[tx] = op_defaults[op];
	partial_op[tx + blockDim.x] = op_defaults[op];

	for (unsigned int stride = blockDim.x; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if (tx < stride)
			partial_op[tx] = operations[op](partial_op[tx], partial_op[tx + stride]);
	}

	if (tx == 0)
		out[blockIdx.x] = partial_op[0];
}

float reduction_op(float *x, unsigned int inputSize, op_t op)
{
	size_t inputSize_bytes = inputSize * sizeof(float);
	unsigned int sectionsCnt = (inputSize + REDUCTION_SECTION_SIZE - 1) / REDUCTION_SECTION_SIZE;
	size_t outSize_bytes = sectionsCnt * sizeof(float);
	float *d_x, *d_out;
	cudaError err;

	err = cudaMalloc(&d_x, inputSize_bytes); checkError(err);
	err = cudaMalloc(&d_out, outSize_bytes); checkError(err);
	err = cudaMemcpy(d_x, x, inputSize_bytes, cudaMemcpyHostToDevice); checkError(err);

	dim3 gridDim(sectionsCnt, 1, 1);
	dim3 blockDim(REDUCTION_SECTION_SIZE / 2, 1, 1);
	reduction_op_kernel<<<gridDim, blockDim>>>(d_x, d_out, inputSize, op);
	err = cudaGetLastError(); checkError(err);

	float ret;
	if (sectionsCnt == 1)
	{
		err = cudaMemcpy(&ret, d_out, sizeof(float), cudaMemcpyDeviceToHost); checkError(err);
	}
	else
	{
		float *d_out2;
		err = cudaMalloc(&d_out2, sizeof(float)); checkError(err);

		gridDim.x = 1;
		reduction_op_kernel << <gridDim, blockDim >> >(d_out, d_out2, sectionsCnt, op);
		err = cudaGetLastError(); checkError(err);

		err = cudaMemcpy(&ret, d_out2, sizeof(float), cudaMemcpyDeviceToHost); checkError(err);
		err = cudaFree(d_out2); checkError(err);
	}
	
	err = cudaFree(d_x); checkError(err);
	err = cudaFree(d_out); checkError(err);

	return ret;
}

#define BLOCK_DIM 1024

__global__
void prob_kernel(const unsigned int* __restrict__ x, float *y, unsigned int inputSize, unsigned int wh)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < inputSize)
		y[i] = x[i] / (float)wh;
}

void prob(unsigned int *x, float *y, unsigned int inputSize, unsigned int wh)
{
	size_t sizeX = inputSize * sizeof(unsigned int);
	size_t sizeY = inputSize * sizeof(float);
	unsigned int *d_x;
	float *d_y;
	cudaError err;

	err = cudaMalloc(&d_x, sizeX); checkError(err);
	err = cudaMalloc(&d_y, sizeY); checkError(err);
	err = cudaMemcpy(d_x, x, sizeX, cudaMemcpyHostToDevice); checkError(err);

	dim3 gridDim((inputSize + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
	dim3 blockDim(BLOCK_DIM, 1, 1);
	prob_kernel<<<gridDim, blockDim>>>(d_x, d_y, inputSize, wh);
	err = cudaGetLastError(); checkError(err);

	err = cudaMemcpy(y, d_y, sizeY, cudaMemcpyDeviceToHost); checkError(err);

	err = cudaFree(d_x); checkError(err);
	err = cudaFree(d_y); checkError(err);
}