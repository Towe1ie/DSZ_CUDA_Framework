#ifndef _PREFIX_SUM_H_
#define _PREFIX_SUM_H_

#include <cuda_runtime.h>

#define SECTION_SIZE 1024

typedef float(*op_pointer)(float, float);
typedef enum { SUM = 0 } op_t;


void sequential_scan(float *x, float *y, unsigned int inputSize);
void reduction_op(float *x, float *out, op_t op);
void work_inefficient_scan(float *x, float *y, unsigned int inputSize);
void work_efficient_scan(float *x, float *y, unsigned int inputSize, op_t op);

void prob(unsigned int *x, float *y, unsigned int inputSize, unsigned int wh);

void prefixSum_test();

#endif;