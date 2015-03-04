#include "DSZ_Image_Processing.h"
#include "DSZCudaUtility.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sm_20_atomic_functions.h>

#include "bmp_parser.h"


__device__
static Pixel_t createPixel(Byte_t r, Byte_t g, Byte_t b)
{
	Pixel_t p;
	p.r = r;
	p.g = g;
	p.b = b;

	return p;
}

// **** Convolution ****
#define INPUT_TILE_WIDTH 16
#define OUT_TILE_WIDTH (INPUT_TILE_WIDTH - (MASK_WIDTH - 1))
#define TILE_RADIUS (MASK_WIDTH - 1)/2

__global__
void simple_convolution2D_kernel(Pixel_t *inputPicture, Pixel_t *outPicture, unsigned int width, unsigned int height, float *mask)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		Pixel_t p = createPixel(0, 0, 0);
		float r = 0, g = 0, b = 0;

		for (unsigned int i = 0; i < MASK_WIDTH; ++i)
		{
			for (unsigned int j = 0; j < MASK_WIDTH; ++j)
			{
				unsigned int currX = x - MASK_WIDTH / 2 + j, currY = y - MASK_WIDTH / 2 + i;
				if (currX < width && currY < height)
				{
					r += (float)inputPicture[currY*width + currX].r * mask[i*MASK_WIDTH + j];
					g += (float)inputPicture[currY*width + currX].g * mask[i*MASK_WIDTH + j];
					b += (float)inputPicture[currY*width + currX].b * mask[i*MASK_WIDTH + j];
				}
			}
		}

		p.r = r;
		p.g = g;
		p.b = b;
		outPicture[y*width + x] = p;
	}
}
void simple_convolution2D(Pixel_t *inputPicture, Pixel_t *outputPicture, unsigned int width, unsigned int height, float *mask)
{
	Pixel_t *d_in, *d_out;
	cudaError err;
	size_t size = width * height * sizeof(Pixel_t);
	err = cudaMalloc(&d_in, size); checkError(err);
	err = cudaMalloc(&d_out, size); checkError(err);
	cudaMemcpy(d_in, inputPicture, size, cudaMemcpyHostToDevice);

	size_t maskSize = MASK_WIDTH * MASK_WIDTH * sizeof(float);
	float* d_mask;
	err = cudaMalloc(&d_mask, maskSize); checkError(err);
	err = cudaMemcpy(d_mask, mask, maskSize, cudaMemcpyHostToDevice);


	dim3 gridDimension((width + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH, (height + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH, 1);
	dim3 blockDimension(OUT_TILE_WIDTH, OUT_TILE_WIDTH, 1);
	simple_convolution2D_kernel<<<gridDimension, blockDimension>>>(d_in, d_out, width, height, d_mask);
	err = cudaGetLastError(); checkError(err);

	cudaMemcpy(outputPicture, d_out, size, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}

__global__
void tiled_convolution2D_kernel(Pixel_t *inputPicture, Pixel_t *outPicture, unsigned int width, unsigned int height, const float* __restrict__ mask)
{
	__shared__ Pixel_t inputTile[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];

	int out_x = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x;
	int out_y = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y;
	int in_x = out_x - TILE_RADIUS;
	int in_y = out_y - TILE_RADIUS;
	unsigned int localX = threadIdx.x;
	unsigned int localY = threadIdx.y;

	if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height)
		inputTile[localY][localX] = inputPicture[in_y*width + in_x];
	else
		inputTile[localY][localX] = createPixel(0, 0, 0);

	__syncthreads();

	if (localX < OUT_TILE_WIDTH && localY < OUT_TILE_WIDTH && (out_x < width && out_y < height))
	{
		Pixel_t p = createPixel(0, 0, 0);
		float r = 0, g = 0, b = 0;
		for (unsigned int i = 0; i < MASK_WIDTH; ++i)
		{
			for (unsigned int j = 0; j < MASK_WIDTH; ++j)
			{
				unsigned int currX = localX + j, currY = localY + i;
				if (currX < width && currY < height)
				{
					r += (float)inputTile[currY][currX].r * mask[i*MASK_WIDTH + j];
					g += (float)inputTile[currY][currX].g * mask[i*MASK_WIDTH + j];
					b += (float)inputTile[currY][currX].b * mask[i*MASK_WIDTH + j];
				}
			}
		}

		p.r = r;
		p.g = g;
		p.b = b;
		outPicture[out_y*width + out_x] = p;
	}
}
void tiled_convolution2D(Pixel_t *inputPicture, Pixel_t *outputPicture, unsigned int width, unsigned int height, const float* __restrict__ mask)
{
	Pixel_t *d_in, *d_out;
	cudaError err;
	size_t size = width * height * sizeof(Pixel_t);
	err = cudaMalloc(&d_in, size); checkError(err);
	err = cudaMalloc(&d_out, size); checkError(err);
	cudaMemcpy(d_in, inputPicture, size, cudaMemcpyHostToDevice);

	size_t maskSize = MASK_WIDTH * MASK_WIDTH * sizeof(float);
	float* d_mask;
	err = cudaMalloc(&d_mask, maskSize); checkError(err);
	err = cudaMemcpy(d_mask, mask, maskSize, cudaMemcpyHostToDevice);


	dim3 gridDimension((width + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH, (height + OUT_TILE_WIDTH - 1) / OUT_TILE_WIDTH, 1);
	dim3 blockDimension(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH, 1);
	tiled_convolution2D_kernel << <gridDimension, blockDimension >> >(d_in, d_out, width, height, d_mask);
	err = cudaGetLastError(); checkError(err);

	cudaMemcpy(outputPicture, d_out, size, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}

// **** Greyscale ****
#define BLOCK_DIM 16
__global__
void greyscale_kernel(const Pixel_t* __restrict__ inputPicture, Pixel_t *outputPicture, unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	__syncthreads();
	if (x < width && y < height)
	{
		Pixel_t p = inputPicture[y * width + x];
		Byte_t v = (0.21*p.r + 0.71*p.g + 0.07*p.b);
		p.r = v;
		p.g = v;
		p.b = v;

		outputPicture[y * width + x] = p;
	}
}
void greyscale(Pixel_t *inputPicture, Pixel_t *outputPicture, unsigned int width, unsigned int height)
{
	Pixel_t *d_in, *d_out;
	cudaError err;
	size_t size = width * height * sizeof(Pixel_t);
	err = cudaMalloc(&d_in, size); checkError(err);
	err = cudaMalloc(&d_out, size); checkError(err);
	cudaMemcpy(d_in, inputPicture, size, cudaMemcpyHostToDevice);

	dim3 gridDimension((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM, 1);
	dim3 blockDimension(BLOCK_DIM, BLOCK_DIM, 1);
	greyscale_kernel<<<gridDimension, blockDimension>>>(d_in, d_out, width, height);
	err = cudaGetLastError(); checkError(err);

	err = cudaMemcpy(outputPicture, d_out, size, cudaMemcpyDeviceToHost);
	checkError(err);

	cudaFree(d_in);
	cudaFree(d_out);
}

// **** Histogram ****
#define HISTOGRAM_BLOCK_LENGTH 256
#define HISTOGRAM_SIZE 256
__global__
void compute_Image_Histogram_kernel(Pixel_t *inputPicture, unsigned int width, unsigned int height, unsigned int *histogram)
{
	__shared__ unsigned int ds_histogram[HISTOGRAM_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x < 256)
		ds_histogram[threadIdx.x] = 0;
	__syncthreads();

	if (i < width * height)
	{
		unsigned int v = inputPicture[i].r;
		atomicAdd(&ds_histogram[v], 1);
	}

	__syncthreads();
	if (threadIdx.x < 256)
		atomicAdd(&histogram[threadIdx.x], ds_histogram[threadIdx.x]);
}

void compute_Image_Histogram(Pixel_t *inputPicture, unsigned int width, unsigned int height, unsigned int *histogram)
{
	Pixel_t *d_inImage;
	unsigned int *d_outHisto;
	cudaError err;
	size_t sizeImage_bytes = width * height * sizeof(Pixel_t);
	size_t sizeHisto_bytes = HISTOGRAM_SIZE * sizeof(unsigned int);

	err = cudaMalloc(&d_inImage, sizeImage_bytes); checkError(err);
	err = cudaMalloc(&d_outHisto, sizeHisto_bytes); checkError(err);
	err = cudaMemcpy(d_inImage, inputPicture, sizeImage_bytes, cudaMemcpyHostToDevice); checkError(err);
	err = cudaMemset(d_outHisto, 0, sizeHisto_bytes); checkError(err);

	dim3 gridDimension((width * height + HISTOGRAM_BLOCK_LENGTH - 1) / HISTOGRAM_BLOCK_LENGTH, 1, 1);
	dim3 blockDimension(HISTOGRAM_BLOCK_LENGTH, 1, 1);
	compute_Image_Histogram_kernel<<<gridDimension, blockDimension>>>(d_inImage, width, height, d_outHisto);
	err = cudaGetLastError(); checkError(err);

	err = cudaMemcpy(histogram, d_outHisto, sizeHisto_bytes, cudaMemcpyDeviceToHost);
	checkError(err);

	err = cudaFree(d_inImage); checkError(err);
	err = cudaFree(d_outHisto); checkError(err);
}