#include "DSZCudaUtility.h"
#include <stdlib.h>
#include <stdio.h>
#include "PrefixSum.h"
#include <math.h>

#include "DSZ_Image_Processing.h"
#include "PrefixSum.h"


int main()
{
	queryDevices();

	//convolution_test();
	//prefixSum_test();
	histogram_test();

	system("pause");
	return 0;
}