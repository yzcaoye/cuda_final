#include "cuda_runtime.h"
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "matrixmul.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


#define WIDTH 10
#define HEIGHT 10
#define DELTA_T 0.075
#define PARA_A 0.15


__global__ void periodicalize_row(Matrix in) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float frontRow1 = in.elements[0 + i];
	float frontRow2 = in.elements[in.width + i];
	float backRow1 = in.elements[in.width * (in.height - 2) + i];
	float backRow2 = in.elements[in.width * (in.height - 1) + i];
	float leftCol1 = in.elements[in.width * i + 0];
	float leftCol2 = in.elements[in.width * i + 1];
	float rightCol1 = in.elements[in.width * i + in.width - 2];
	float rightCol2 = in.elements[in.width * i + in.width - 1];
	in.elements[0 + i] = backRow1;
	in.elements[in.width + i] = backRow2;
	in.elements[in.width * (in.height - 2) + i] = frontRow1;
	in.elements[in.width * (in.height - 1) + i] = frontRow2;
	in.elements[in.width * i + 0] = rightCol1;
	in.elements[in.width * i + 1] = rightCol2;
	in.elements[in.width * i + in.width - 2] = leftCol1;
	in.elements[in.width * i + in.width - 1] = leftCol2;
}

__device__ float laplaceCal(float front, float back, float deltaX, float deltaY, float num) {
	float res = (1/powf(deltaX, num)) * front + (1 / powf(deltaY, num)) * back;
	return res;
}

__device__ float frontCal(float *in) {
	float res = 0.125 * (in[3 + 3] + in[1 + 3] + in[3 + 1]) + 0.75 * (in[3 + 2] + in[1 + 2]) - 0.25 * (in[3 + 2] + in[1 + 2]) - 1.5 * in[2 + 2];
}

__device__ float backCal(float *in) {
	float res = 0.125 * (in[3 + 3] + in[1 + 3] + in[3 + 1]) + 0.75 * (in[2 + 3] + in[2 + 1]) - 0.25 * (in[2 + 3] + in[2 + 1]) - 1.5 * in[2 + 2];
}

__device__ float* getFOI(Matrix in, int i) {
	//how about using shared mem?
	float foi[9];
	foi[0 + 0] = in.elements[i - WIDTH - 1];
	foi[0 + 1] = in.elements[i - WIDTH];
	foi[0 + 2] = in.elements[i - WIDTH + 1];
	foi[1 + 0] = in.elements[i - 1];
	foi[1 + 1] = in.elements[i];
	foi[1 + 2] = in.elements[i + 1];
	foi[2 + 0] = in.elements[i + WIDTH - 1];
	foi[2 + 1] = in.elements[i + WIDTH];
	foi[2 + 2] = in.elements[i + WIDTH + 1];
	return foi;
}

__device__ float* foiPowOf3(float *foi) {
	float foi3[9];
	foi3[0 + 0] = foi[0 + 0] * foi[0 + 0] * foi[0 + 0];
	foi3[0 + 1] = foi[0 + 1] * foi[0 + 1] * foi[0 + 1];
	foi3[0 + 2] = foi[0 + 2] * foi[0 + 2] * foi[0 + 2];
	foi3[1 + 0] = foi[1 + 0] * foi[1 + 0] * foi[1 + 0];
	foi3[1 + 1] = foi[1 + 1] * foi[1 + 1] * foi[1 + 1];
	foi3[1 + 2] = foi[1 + 2] * foi[1 + 2] * foi[1 + 2];
	foi3[2 + 0] = foi[2 + 0] * foi[2 + 0] * foi[2 + 0];
	foi3[2 + 1] = foi[2 + 1] * foi[2 + 1] * foi[2 + 1];
	foi3[2 + 2] = foi[2 + 2] * foi[2 + 2] * foi[2 + 2];
	return foi3;
}

__device__ void getNowOi(Matrix out, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32, int i) {
	out.elements[i] = oi.elements[i] + DELTA_T * ((1 - PARA_A) * tempRecordOi2.elements[i] + 2 * tempRecordOi4.elements[i] + tempRecordOi6.elements[i] + tempRecordOi32.elements[i]);
}

__constant__ float deltaX;
__constant__ float deltaY;

__global__ void firstCal(Matrix newOi, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32) {
	//consider using tile?
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//use col and row represent i?
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	if (col > 0 && col < WIDTH - 1 && row > 0 && row < HEIGHT - 1) {
		float *foi = getFOI(oi, i);
		float *foi3 = foiPowOf3(foi);
		float tempOi2 = laplaceCal(frontCal(foi), backCal(foi), deltaX, deltaY, 2.0);
		float tempOi32 = laplaceCal(frontCal(foi3), backCal(foi3), deltaX, deltaY, 2.0);
		tempRecordOi2.elements[i] = tempOi2;
		tempRecordOi32.elements[i] = tempOi32;
		__syncthreads;
		float *foi2 = getFOI(tempRecordOi2, i);
		float tempOi4 = laplaceCal(frontCal(foi2), backCal(foi2), deltaX,deltaY, 2);
		tempRecordOi4.elements[i] = tempOi4;
		__syncthreads();
		float *foi4 = getFOI(tempRecordOi4, i);
		float tempOi6 = laplaceCal(frontCal(foi4), backCal(foi4), deltaX, deltaY, 2);
		tempRecordOi6.elements[i] = tempOi6;
		newOi.elements[i] = oi.elements[i] + DELTA_T * ((1 - PARA_A) * tempOi2 + 2 * tempOi4 + tempOi6 + tempOi32);
	}

}


