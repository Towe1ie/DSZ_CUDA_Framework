#include "DSZCudaMath.h"

__device__
float sum(float a, float b)
{
	return a + b;
}

__device__
float d_min(float a, float b)
{
	bool a_cb = a < b;
	return a_cb * a + (!a_cb) * b;
}

__device__
float d_max(float a, float b)
{
	bool a_cb = a > b;
	return a_cb * a + (!a_cb) * b;
}

__device__
float cudaClamp(float x, float min, float max)
{
	if (x < min)
		return min;
	else if (x > max)
		return max;
	else
		return x;
}