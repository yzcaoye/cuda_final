#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "matrixmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "helper_cuda.h"
#include "helper_string.h"

#define PI 3.14159265
#define deltaT 0.0075
#define paraA 0.35
#define WIDTH 32

__constant__ float deltaX;
__constant__ float deltaY;

__global__ void periodicalize(Matrix in);
__device__ float laplaceCal(float front, float back, float deltaX, float deltaY, float num);
__device__ float frontCal(float *in);
__device__ float backCal(float *in);
__device__ float* getFOI(Matrix in, int i);
__device__ float* foiPowOf3(float *foi);
__device__ void getNowOi(Matrix out, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32, int i);
__global__ void allCal(Matrix newOi, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32);
__global__ void firstCal(Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi32);
__device__ float laplaceCal_r(float *in, float deltaX, float deltaY, float num);



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	
	float oi0 = 0.3;
	float qh = sqrt(3.0)/2;
	int optionOi = 1;
	int areaX = 22;
	int areaY = 22;
	int numT = 100;
	float deltaX0 = PI / 4;
	float deltaY0 = PI / 4;
	int nucleusR = 11;
	
	float strainR = 1e-5 / deltaT;
//	printf("strainR%f", strainR);
	float totalT = numT*deltaT;
	int numX = areaX / deltaX0;
	int numY = areaY / deltaY0;
	printf("numX: %d, numY:%d\n", numX, numY);
	Matrix oi_host = AllocateMatrix(numX + 2, numY + 2, oi0);
	

	Matrix axisX = AllocateMatrix((numX + 2), numY + 2, 1.0);
	Matrix axisY = AllocateMatrix((numY+2),numX+2,1.0);
	Matrix temp_matrix = AllocateMatrix(numX + 2, numY + 2, 0.0);
	Matrix chooseX = AllocateMatrix(numX + 2, numY + 2, 0);
	Matrix nucleusOi = AllocateMatrix(numX + 2, numY + 2, 0);
	for (int i = 0; i < numX + 2; i++){
		float temp = deltaX0*((i + 1) - numX / 2);
		for (int j = 0; j < numY + 2; j++){
			axisX.elements[j*(numX+2) + i] = temp;
		}
	}
	
	

	for (int i = numY + 2; i >= 1; i--){
		float temp = deltaY0*(i - numY / 2);
		for (int j = 0; j < numX + 2; j++){
			axisY.elements[(numY+2-i)*(numX + 2) + j] = temp;
		}
	}
	

	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			temp_matrix.elements[i*(numX + 2) + j] =
				axisX.elements[i*(numX + 2) + j] * axisX.elements[i*(numX + 2) + j] 
				+ axisY.elements[i*(numX + 2) + j] * axisY.elements[i*(numX + 2) + j];

		}
	}
	printf("\n***temp_matrix***\n");
	//test temp_matrix
/*	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			printf("%f\t", temp_matrix.elements[i*(numX + 2) + j]);
		}
		printf("\n");
	}
*/
	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			if (temp_matrix.elements[i*(numX + 2) + j] <= (nucleusR*nucleusR)){
				chooseX.elements[i*(numX + 2) + j] = 1;
			}
		}
		
	}
	printf("\n***chooseX***\n");
	//test chooseX
/*	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			printf("%f\t", chooseX.elements[i*(numX + 2) + j]);
		}
		printf("\n");
	}
*/
	//float At = -1*4 / 5 * (oi0 + sqrt(5 / 3 * paraA - 4 * oi0 *oi0));
	float At = (oi0+sqrt(5/3.0*paraA-4*oi0*oi0))*(-0.8);
	
	for (int i = 0; i < numX + 2; i++){
		
		for (int j = 0; j < numY + 2; j++){
			axisX.elements[i*(numX + 2) + j] = axisX.elements[i*(numX + 2) + j] * 0.9659
				+ axisY.elements[i*(numX + 2) + j] * 0.2588;
		}
	}
	for (int i = 0; i < numX + 2; i++){

		for (int j = 0; j < numY + 2; j++){
			axisY.elements[i*(numX + 2) + j] = -axisX.elements[i*(numX + 2) + j] * 0.2588
				+ axisY.elements[i*(numX + 2) + j] * 0.9659;
		}
	}

	printf("\n***axisX***\n");
	//test axisX
/*	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			printf("%f\t", axisX.elements[i*(numX + 2) + j]);
		}
		printf("\n");
	}
*/
	printf("\n***axisY***\n");
	//test axisY
/*	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			printf("%f\t", axisY.elements[i*(numX + 2) + j]);
		}
		printf("\n");
	}
*/
	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			nucleusOi.elements[i*(numX + 2) + j] = At*
				(cos(qh*axisX.elements[i*(numX + 2) + j])*cos(1 / sqrt(3.0)*qh*axisY.elements[i*(numX + 2) + j])+
				0.5 * cos(2 / sqrt(3.0)*qh*axisY.elements[i*(numX + 2) + j]));
		}
	}
	printf("\n***nucleusOi***\n");
	//test nucleusOi
/*	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			printf("%f\t", nucleusOi.elements[i*(numX + 2) + j]);
		}
		printf("\n");
	}
*/
	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			oi_host.elements[i*(numX + 2) + j] += chooseX.elements[i*(numX + 2) + j] * nucleusOi.elements[i*(numX + 2) + j];
		}
	}

	printf("\n***oi_host***\n");
	//test nucleusOi
/*	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			printf("%f\t", oi_host.elements[i*(numX + 2) + j]);
		}
		printf("\n");
	}
	*/
	
	Matrix newoi = AllocateDeviceMatrix(oi_host);

	//Matrix u = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int i = 0; i < numT; i++)
	{
		//update oi_host
		//periodicalize x
		for (int m = 0, n = numY; m < numY + 2, n < numY + 2; m++, n++){
			for (int j = 0; j < numX + 2; j++){
				oi_host.elements[m*(numX + 2) + j] = oi_host.elements[n*(numX + 2) + j];
			}
		}
		for (int m = numY+1, n = 1; m < numY + 2, n < numY + 2; m++, n++){
			for (int j = 0; j < numX + 2; j++){
				oi_host.elements[m*(numX + 2) + j] = oi_host.elements[n*(numX + 2) + j];
			}
		}
		//periodicalize y
		for (int j = 0; j < numY + 2; j++){
			for (int m = 0, n = numX; m < numX+2, n < numX+2; m++, n++){
				oi_host.elements[j*(numX + 2) + m] = oi_host.elements[j*(numX + 2) + n];
			}
		}
		for (int j = 0; j < numY + 2; j++){
			for (int m = numX + 1, n = 1; m < numX + 2, n < numX + 2; m++, n++){
				oi_host.elements[j*(numX + 2) + m] = oi_host.elements[j*(numX + 2) + n];
			}
		}

		//test after periodicalize
		/*printf("oi after periodicalize!\n");
		for (int i = 0; i < numY + 2; i++){
			for (int j = 0; j < numX + 2; j++){
				printf("%f\t", oi_host.elements[i*(numX + 2) + j]);
			}
			printf("\n");
		}
		*/
		Matrix oi = AllocateDeviceMatrix(oi_host);

		Matrix tempRecordOi2 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
		Matrix tempRecordOi4 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
		Matrix tempRecordOi6 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
		Matrix tempRecordOi32 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));

		float *tempx = (float*)malloc(sizeof(float));
		float *tempy = (float*)malloc(sizeof(float));

		*tempx = deltaX0 + i*0.00005;
		*tempy = deltaY0*(deltaX0 / *tempx);
		
		printf("tempx:%f\n", *tempx);
		printf("tempy:%f\n", *tempy);

		cudaMemcpyToSymbol(deltaX, tempx, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(deltaY, tempy, sizeof(float), cudaMemcpyHostToDevice);

		/*
		cudaMemcpy(&deltaX, tempx, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&deltaY, tempy, sizeof(float), cudaMemcpyDeviceToHost);

		printf("After tempx:%f\n", *tempx);
		printf("After tempy:%f\n", *tempy);
		*/
		free(tempx);
		free(tempy);



		CopyToDeviceMatrix(oi, oi_host);
		CopyFromDeviceMatrix(oi_host, oi);

		/*for (int i = 0; i < numY + 2; i++){
			for (int j = 0; j < numX + 2; j++){
				printf("%f\t", oi_host.elements[i*(numX + 2) + j]);
			}
			printf("\n");
		}
		*/
		//////////////////////////////////////////////////////////

		dim3 block_size(WIDTH, WIDTH);
		int grid_rows = oi.height / WIDTH + (oi.height % WIDTH ? 1 : 0);
		int grid_cols = oi.width / WIDTH + (oi.width % WIDTH ? 1 : 0);
		dim3 grid_size(grid_cols, grid_rows);

		printf("round: %d\n", i);
		//allCal<<< grid_size, block_size >>>(newoi, oi, tempRecordOi2, tempRecordOi4, tempRecordOi6, tempRecordOi32);
		allCal << < 1, 1024 >> >(newoi, oi, tempRecordOi2, tempRecordOi4, tempRecordOi6, tempRecordOi32);
		//firstCal << < 1, 64 >> >(oi, tempRecordOi2, tempRecordOi32);
		cudaDeviceSynchronize();
		//Check_CUDA_Error("Kernel Execution Failed!");
		


		/////////////////////////////////////////////////////////

		CopyFromDeviceMatrix(oi_host, newoi);
		
		/*for (int i = 0; i < numY + 2; i++){
			for (int j = 0; j < numX + 2; j++){
				printf("%f\t", oi_host.elements[i*(numX + 2) + j]);
			}
			printf("\n");
		}*/
			
			if (i == numT - 1){
			printf("Free device matrix!\n");
			FreeDeviceMatrix(&tempRecordOi2);
			FreeDeviceMatrix(&tempRecordOi4);
			FreeDeviceMatrix(&tempRecordOi6);
			FreeDeviceMatrix(&tempRecordOi32);
			FreeDeviceMatrix(&oi);
			FreeDeviceMatrix(&newoi);
			
		}

			

	}

	
	/*
	printf("Free host matrix!");
	FreeMatrix(&axisX);
	FreeMatrix(&axisY);
	FreeMatrix(&temp_matrix);
	FreeMatrix(&chooseX);
	FreeMatrix(&nucleusOi);
	*/

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Processing Time: %3.1f ms \n", elapsedTime);
	
    return 0;
	
}


__global__ void periodicalize(Matrix in) {
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
	//printf("front:%f, back:%f", front, back);
	float res = (front / powf(deltaX, num)) + (back / powf(deltaY, num));
	//printf("laplace res: %f", res);
	return res;
}

__device__ float laplaceCal_r(float *in, float deltaX, float deltaY, float num){
	float res = (0.125 * (in[2 * 3 + 2] + in[0 * 3 + 2] + in[2 * 3 + 0] + in[0 * 3 + 0]) + 0.75 * (in[2 * 3 + 1] + in[0 * 3 + 1]) - 0.25 * (in[2 * 3 + 1] + in[0 * 3 + 1]) - 1.5 * in[1 * 3 + 1] / powf(deltaX, num));
	//printf("new laplace: %f", res);
	return res;
}

__device__ float frontCal(float *in) {
	float res = 0.125 * (in[2*3 + 2] + in[0*3 + 2] + in[2*3 + 0]+in[0*3+0]) + 0.75 * (in[2*3 + 1] + in[0*3 + 1]) - 0.25 * (in[2*3 + 1] + in[0*3 + 1]) - 1.5 * in[1*3 + 1];
	//printf("front:%f\t", res);
}

__device__ float backCal(float *in) {
	
	float res = 0.125 * (in[2 * 3 + 2] + in[0 * 3 + 2] + in[2 * 3 + 0] + in[0 * 3 + 0]) + 
		0.75 * (in[1*3 + 2] + in[1*3 + 0]) - 0.25 * (in[1*3 + 2] + in[1*3 + 0]) - 1.5 * in[1*3 + 1];
	//printf("back:%f\t", res);
}

__device__ float* getFOI(Matrix in, int i) {
	//how about using shared mem?
	//printf("i: %d\n", i);
	float foi[9];
	foi[0 + 0] = in.elements[i - in.width - 1];
	foi[0 + 1] = in.elements[i - in.width];
	foi[0 + 2] = in.elements[i - in.width + 1];
	foi[1*3 + 0] = in.elements[i - 1];
	foi[1*3+ 1] = in.elements[i];
	foi[1*3+ 2] = in.elements[i + 1];
	foi[2*3 + 0] = in.elements[i + in.width - 1];
	foi[2*3 + 1] = in.elements[i + in.width];
	foi[2*3 + 2] = in.elements[i + in.width + 1];
	return foi;
}

__device__ float* foiPowOf3(float *foi) {
	float foithree[9];
	foithree[0 + 0] = powf(foi[0], 3.0);
	foithree[0 + 1] = powf(foi[1], 3.0);
	foithree[0 + 2] = powf(foi[2], 3.0);
	foithree[1 * 3 + 0] = powf(foi[3], 3.0);
	foithree[1 * 3 + 1] = foi[1 * 3 + 1] * foi[1 * 3 + 1] * foi[1 * 3 + 1];
	foithree[1 * 3 + 2] = foi[1 * 3 + 2] * foi[1 * 3 + 2] * foi[1 * 3 + 2];
	foithree[2 * 3 + 0] = foi[2 * 3 + 0] * foi[2 * 3 + 0] * foi[2 * 3 + 0];
	foithree[2 * 3 + 1] = foi[2 * 3 + 1] * foi[2 * 3 + 1] * foi[2 * 3 + 1];
	foithree[2 * 3 + 2] = foi[2 * 3 + 2] * foi[2 * 3 + 2] * foi[2 * 3 + 2];
	return foithree;
}

__device__ void getNowOi(Matrix out, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32, int i) {
	out.elements[i] = oi.elements[i] + deltaT * ((1 - paraA) * tempRecordOi2.elements[i] + 2 * tempRecordOi4.elements[i] + tempRecordOi6.elements[i] + tempRecordOi32.elements[i]);
}


__global__ void allCal(Matrix newOi, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32) {
	//consider using tile?
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//use col and row represent i?
	int i = blockIdx.x *blockDim.x + threadIdx.x;

	if (!(i%oi.width == 0 || i%oi.width == (oi.width - 1) || (i >= 0 && i <= (oi.width - 1)) || (i <= (oi.height*oi.width - 1) && i >= (oi.height - 1)*oi.width) || i >= oi.height*oi.width)) {
		float *foi = getFOI(oi, i);
		//float tempOi2 = laplaceCal(frontCal(foi), backCal(foi), deltaX, deltaY, 2.0);
		float tempOi2 = laplaceCal_r(foi, deltaX, deltaY, 2.0);
		tempRecordOi2.elements[i] = tempOi2;
		//printf("tempOi2:%f\t", tempOi2);
		float *threefoi = foiPowOf3(foi);
		//float tempOi32 = laplaceCal(frontCal(foi3), backCal(foi3), deltaX, deltaY, 2.0);
		float tempOi32 = laplaceCal_r(threefoi, deltaX, deltaY, 2.0);
		tempRecordOi32.elements[i] = tempOi32;
		//printf("tempOi32:%f\t", tempOi32);
		__syncthreads;
		float *twofoi = getFOI(tempRecordOi2, i);
		//float tempOi4 = laplaceCal(frontCal(foi2), backCal(foi2), deltaX, deltaY, 2);
		float tempOi4 = laplaceCal_r(twofoi, deltaX, deltaY, 2.0);
		tempRecordOi4.elements[i] = tempOi4;
		__syncthreads();
		float *fourfoi = getFOI(tempRecordOi4, i);
		//float tempOi6 = laplaceCal(frontCal(foi4), backCal(foi4), deltaX, deltaY, 2);
		float tempOi6 = laplaceCal_r(fourfoi, deltaX, deltaY, 2.0);
		tempRecordOi6.elements[i] = tempOi6;
		newOi.elements[i] = oi.elements[i] + deltaT * ((1 - paraA) * tempOi2 + 2 * tempOi4 + tempOi6 + tempOi32);
	}

}


__global__ void firstCal(Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi32){
	//consider using tile?
	//int col = blockIdx.x * blockDim.x + threadIdx.x;
	//int row = blockIdx.y * blockDim.y + threadIdx.y;
	//use col and row represent i?
	//int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	
	if (!(i%oi.width == 0 || i%oi.width == (oi.width - 1) || (i >= 0 && i <= (oi.width - 1)) || (i <= (oi.height*oi.width - 1) && i >= (oi.height - 1)*oi.width)||i>=oi.height*oi.width)){
		float *foi = getFOI(oi, i);
		tempRecordOi2.elements[i] = laplaceCal(frontCal(foi), backCal(foi), deltaX, deltaY, 2.0);
		float *foithree = foiPowOf3(foi);
		tempRecordOi32.elements[i] = laplaceCal(frontCal(foithree), backCal(foithree), deltaX, deltaY, 2.0);
	}
	
		
	
}


