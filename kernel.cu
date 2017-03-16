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
#include <windows.h>
#include<fstream>
#include<iostream>
#include<iomanip>

using namespace std;

#define PI 3.14159265
#define deltaT 0.0075
#define paraA 0.35
#define WIDTH 32
#define SIZE 1024
__constant__ float deltaX[1];
__constant__ float deltaY[1];


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
__global__ void callThree(Matrix newOi, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32);
__global__ void calTwo(Matrix tempRecordOi2, Matrix tempRecordOi4);
__global__ void calOne(Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi32);
__device__ float* getFOIf(float *in, int i, int width);
__global__ void allCal_new(Matrix newOi, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32);



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{

	SYSTEMTIME sysstart;
	SYSTEMTIME sysstop;
	GetLocalTime(&sysstart);
	float oi0 = 0.3;
	float qh = sqrt(3.0) / 2;
	int optionOi = 1;
	int areaX = 5;
	int areaY = 5;
	int numT = 10;
	float deltaX0 = PI / 4;
	float deltaY0 = PI / 4;
	int nucleusR = 11;

	float strainR = 1e-5 / deltaT;
	//printf("strainR%f", strainR);
	float totalT = numT*deltaT;
	int numX = areaX / 0.7850;
	int numY = areaY / 0.7850;
	//printf("numX: %d, numY:%d\n", numX, numY);
	Matrix oi_host = AllocateMatrix(numX + 2, numY + 2, oi0);


	Matrix axisX = AllocateMatrix((numX + 2), numY + 2, 1.0);
	Matrix axisY = AllocateMatrix((numY + 2), numX + 2, 1.0);
	Matrix temp_matrix = AllocateMatrix(numX + 2, numY + 2, 0.0);
	Matrix chooseX = AllocateMatrix(numX + 2, numY + 2, 0);
	Matrix nucleusOi = AllocateMatrix(numX + 2, numY + 2, 0);
	for (int i = 0; i < numX + 2; i++){
		float temp = 0.7850*((i + 1) - numX / 2);
		for (int j = 0; j < numY + 2; j++){
			axisX.elements[j*(numX + 2) + i] = temp;
		}
	}



	for (int i = numY + 2; i >= 1; i--){
		float temp = 0.7850*(i - numY / 2);
		for (int j = 0; j < numX + 2; j++){
			axisY.elements[(numY + 2 - i)*(numX + 2) + j] = temp;
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
	/*for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
		printf("%f\t", temp_matrix.elements[i*(numX + 2) + j]);
		}
		printf("\n");
		}*/

	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			if (temp_matrix.elements[i*(numX + 2) + j] <= (nucleusR*nucleusR)){
				chooseX.elements[i*(numX + 2) + j] = 1;
			}
		}

	}
	printf("\n***chooseX***\n");
	//test chooseX
	/*for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
		printf("%f\t", chooseX.elements[i*(numX + 2) + j]);
		}
		printf("\n");
		}*/

	//float At = -1*4 / 5 * (oi0 + sqrt(5 / 3 * paraA - 4 * oi0 *oi0));
	float At = (oi0 + sqrt(5 / 3.0*paraA - 4 * oi0*oi0))*(-0.8);

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
				(cos(qh*axisX.elements[i*(numX + 2) + j])*cos(1 / sqrt(3.0)*qh*axisY.elements[i*(numX + 2) + j]) +
				0.5 * cos(2 / sqrt(3.0)*qh*axisY.elements[i*(numX + 2) + j]));
		}
	}
	printf("\n***nucleusOi***\n");
	//test nucleusOi
	/*for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
		printf("%f\t", nucleusOi.elements[i*(numX + 2) + j]);
		}
		printf("\n");
		}*/

	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			oi_host.elements[i*(numX + 2) + j] += chooseX.elements[i*(numX + 2) + j] * nucleusOi.elements[i*(numX + 2) + j];
		}
	}

	printf("\n***oi_host***\n");
	//test oi_host
	/*for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
		printf("%f\t", oi_host.elements[i*(numX + 2) + j]);
		}
		printf("\n");
		}*/


	Matrix newoi = AllocateDeviceMatrix(oi_host);

	//Matrix u = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	Matrix oi = AllocateDeviceMatrix(oi_host);
	Matrix tempRecordOi2 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
	Matrix tempRecordOi4 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
	Matrix tempRecordOi6 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
	Matrix tempRecordOi32 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));

	for (int i = 0; i < numT; i++)
	{
		printf("round: %d\n", i);
		//printf("********oi before period************\n");

		//update oi_host
		//periodicalize x

		for (int j = 0; j < numX + 2; j++){
			oi_host.elements[0 * (numX + 2) + j] = oi_host.elements[numY*(numX + 2) + j];
		}


		for (int j = 0; j < numX + 2; j++){
			oi_host.elements[(numY + 1)*(numX + 2) + j] = oi_host.elements[1 * (numX + 2) + j];
		}


		//periodicalize y
		for (int j = 0; j < numY + 2; j++){
			oi_host.elements[j*(numX + 2) + 0] = oi_host.elements[j*(numX + 2) + numX];
		}
		for (int j = 0; j < numY + 2; j++){
			oi_host.elements[j*(numX + 2) + numX + 1] = oi_host.elements[j*(numX + 2) + 1];
		}

		//test after periodicalize
		/*printf("oi after periodicalize!\n");
		for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
		printf("%f\t", oi_host.elements[i*(numX + 2) + j]);
		}
		printf("\n");
		}*/

		//Matrix oi = AllocateDeviceMatrix(oi_host);

		/*Matrix tempRecordOi2 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
		Matrix tempRecordOi4 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
		Matrix tempRecordOi6 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
		Matrix tempRecordOi32 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0.0));
		*/
		float tempx[1] = { deltaX0 + i*0.00005 };
		float tempy[1] = { deltaY0*(deltaX0 / tempx[0]) };


		cudaMemcpyToSymbol(deltaX, &tempx, sizeof(float));
		cudaMemcpyToSymbol(deltaY, &tempy, sizeof(float));

		//cudaDeviceSynchronize();
		/*
		cudaMemcpy(&0.785, tempx, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&0.785, tempy, sizeof(float), cudaMemcpyDeviceToHost);

		printf("After tempx:%f\n", *tempx);
		printf("After tempy:%f\n", *tempy);
		*/

		//free(tempx);
		//free(tempy);



		CopyToDeviceMatrix(oi, oi_host);
		//CopyFromDeviceMatrix(oi_host, oi);
		/*printf("********oi before calculation:************\n");
		for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
		printf("%f\t", oi_host.elements[i*(numX + 2) + j]);
		}
		printf("\n");
		}*/

		//////////////////////////////////////////////////////////

		dim3 block_size(WIDTH, WIDTH);
		int grid_rows = oi.height / WIDTH + (oi.height % WIDTH ? 1 : 0);
		int grid_cols = oi.width / WIDTH + (oi.width % WIDTH ? 1 : 0);
		dim3 grid_size(grid_cols, grid_rows);


		allCal<<< 1, 64 >>>(newoi, oi, tempRecordOi2, tempRecordOi4, tempRecordOi6, tempRecordOi32);
		//allCal << < 1, SIZE >> >(newoi, oi, tempRecordOi2, tempRecordOi4, tempRecordOi6, tempRecordOi32);
		//allCal_new << < 1, SIZE >> >(newoi, oi, tempRecordOi2, tempRecordOi4, tempRecordOi6, tempRecordOi32);
		//cudaDeviceSynchronize();
		//firstCal << < 1, 64 >> >(oi, tempRecordOi2, tempRecordOi32);
		/*calOne << <64, SIZE >> >(oi, tempRecordOi2, tempRecordOi32);
		cudaDeviceSynchronize();
		calTwo << <64, SIZE >> >(tempRecordOi2, tempRecordOi4);
		cudaDeviceSynchronize();
		callThree << <64, SIZE >> >(newoi, oi, tempRecordOi2, tempRecordOi4, tempRecordOi6, tempRecordOi32);*/
		cudaDeviceSynchronize();
		Check_CUDA_Error("Kernel Execution Failed!");



		/////////////////////////////////////////////////////////

		CopyFromDeviceMatrix(oi_host, newoi);
		//CopyFromDeviceMatrix(oi_host, tempRecordOi2);
		/*printf("********oi after calculation:************\n");
		for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
		printf("%f\t", oi_host.elements[i*(numX + 2) + j]);
		}
		printf("\n");

		}*/

		/*if (i == numT - 1){
			printf("Free device matrix!\n");
			FreeDeviceMatrix(&tempRecordOi2);
			FreeDeviceMatrix(&tempRecordOi4);
			FreeDeviceMatrix(&tempRecordOi6);
			FreeDeviceMatrix(&tempRecordOi32);
			FreeDeviceMatrix(&oi);
			FreeDeviceMatrix(&newoi);

			}
			*/


	}

	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//float elapsedTime;
	//cudaEventElapsedTime(&elapsedTime, start, stop);


	GetLocalTime(&sysstop);

	for (int i = 0; i < numY + 2; i++){
		for (int j = 0; j < numX + 2; j++){
			printf("%f\t", oi_host.elements[i*(numX + 2) + j]);
		}
		printf("\n");

	}


	//	printf("Processing Time: %3.1f ms \n", elapsedTime);
	printf("Processing Time: %d ms \n", (sysstop.wMilliseconds + sysstop.wSecond * 1000 + sysstop.wMinute * 60000) - (sysstart.wMilliseconds + sysstart.wSecond * 1000 + sysstart.wMinute * 60000));


	//ofstream ofile;               //定义输出文件
	//ofile.open("d:\\myfile.txt");     //作为输出文件打开

	//for (int i = 0; numY + 2; i++){}
	//	for (int j = 0; j < numX + 2; j++){

	//		ofile << oi_host.elements[i*(numX + 2) + j] << "\t";   //数据写入文件
	//	}
	//	ofile << endl;
	//}
	//ofile.close();                //关闭文件
	/*
	printf("Free host matrix!");
	FreeMatrix(&axisX);
	FreeMatrix(&axisY);
	FreeMatrix(&temp_matrix);
	FreeMatrix(&chooseX);
	FreeMatrix(&nucleusOi);
	*/

	
	
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
	float res = (front / powf(0.785, num)) + (back / powf(0.785, num));
	//printf("laplace res: %f", res);
	return res;
}

__device__ float laplaceCal_r(float *in, float deltaX, float deltaY, float num){
	float res = ((0.125 * (in[2 * 3 + 2] + in[0 * 3 + 2] + in[2 * 3 + 0] + in[0 * 3 + 0]) +
		0.75 * (in[2 * 3 + 1] + in[0 * 3 + 1]) - 0.25 * (in[2 * 3 + 1] + in[0 * 3 + 1]) - 1.5 * in[1 * 3 + 1])) / powf(deltaX, num)
		+ (0.125*(in[2 * 3 + 2] + in[0*3+2]+in[2*3+0]+in[0*3+0])+
		0.75*(in[1*3+2]+in[1*3+0])-0.25*(in[1*3+2]+in[1*3+0])-1.5*in[1*3+1])/powf(deltaY,num);
	//printf("new laplace: %f", res);
	//printf("deltaX: %f ", deltaX);
	//printf("deltaY: %f ", deltaY);
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

__device__ float* getFOIf(float *in, int i, int width) {
	//how about using shared mem?
	//printf("i: %d\n", i);
	float foi[9];
	foi[0 + 0] = in[i - width - 1];
	foi[0 + 1] = in[i - width];
	foi[0 + 2] = in[i - width + 1];
	foi[1 * 3 + 0] = in[i - 1];
	foi[1 * 3 + 1] = in[i];
	foi[1 * 3 + 2] = in[i + 1];
	foi[2 * 3 + 0] = in[i + width - 1];
	foi[2 * 3 + 1] = in[i + width];
	foi[2 * 3 + 2] = in[i + width + 1];

	return foi;
}

__device__ float* foiPowOf3(float *foi) {
	float threefoi[9];
	//float foithree[9];
	/*
	foithree[0 + 0] = powf(foi[0], 3.0);
	foithree[0 + 1] = powf(foi[1], 3.0);
	foithree[0 + 2] = powf(foi[2], 3.0);
	foithree[1 * 3 + 0] = powf(foi[3], 3.0);
	foithree[1 * 3 + 1] = powf(foi[4], 3.0);
	foithree[1 * 3 + 2] = powf(foi[5], 3.0);
	foithree[2 * 3 + 0] = powf(foi[6], 3.0);
	foithree[2 * 3 + 1] = powf(foi[7], 3.0);
	foithree[2 * 3 + 2] = powf(foi[8], 3.0);
	*/
	threefoi[0] = foi[0] * foi[0] * foi[0];
	threefoi[1] = foi[1] * foi[1] * foi[1];
	threefoi[2] = foi[2] * foi[2] * foi[2];
	threefoi[3] = foi[3] * foi[3] * foi[3];
	threefoi[4] = foi[4] * foi[4] * foi[4];
	threefoi[5] = foi[5] * foi[5] * foi[5];
	threefoi[6] = foi[6] * foi[6] * foi[6];
	threefoi[7] = foi[7] * foi[7] * foi[7];
	threefoi[8] = foi[8] * foi[8] * foi[8];
	return threefoi;
}

__device__ void getNowOi(Matrix out, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32, int i) {
	out.elements[i] = oi.elements[i] + deltaT * ((1 - paraA) * tempRecordOi2.elements[i] + 2 * tempRecordOi4.elements[i] + tempRecordOi6.elements[i] + tempRecordOi32.elements[i]);
}


__global__ void allCal(Matrix newOi, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32) {
	//consider using tile?
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//use col and row represent i?
	int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	//int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	//int i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	//int i = blockIdx.x *blockDim.x + threadIdx.x;
	//printf("i: %d\t", i);
		float x = deltaX[0];
		float y = deltaY[0];
	
		newOi.elements[i] = oi.elements[i];
		//float *foi = getFOI(oi, i);
		float *in = getFOI(oi, i);
		__syncthreads();
		/*if (i == 9){
			printf("oi:\n");
			printf("%f %f %f\n", in[0], in[1], in[2]);
			printf("%f %f %f\n", in[3], in[4], in[5]);
			printf("%f %f %f\n", in[6], in[7], in[8]);
		}
		*/
		
	if (!(i%oi.width == 0 || i%oi.width == (oi.width - 1) || (i >= 0 && i <= (oi.width - 1)) || 
		(i <= (oi.height*oi.width - 1) && i >= (oi.height - 1)*oi.width) || i >= oi.height*oi.width)) {
		
		//tempOi2 = laplaceCal_r(foi, x, y, 2.0);
		float tempOi2 = ((0.125 * (in[8] + in[2] + in[6] + in[0]) +
			0.75 * (in[7] + in[1]) - 0.25 * (in[7] + in[1]) - 1.5 * in[4]) / (x*x))
			+ ((0.125*(in[8] + in[2] + in[6] + in[0]) +
			0.75*(in[5] + in[3]) - 0.25*(in[5] + in[3]) - 1.5*in[4]) / (y*y));
		tempRecordOi2.elements[i] = tempOi2;
		__syncthreads();

		if (i == 9){
			printf("tempoi2:\n");
			for (int j = 0; j < 64; j++){
				printf("%f\t", tempRecordOi2.elements[j]);
				if (j % 8 == 7){
					printf("\n");
				}
			}
		}

		//float *threefoi = foiPowOf3(foi);

		//tempOi32 = laplaceCal_r(threefoi, x, y, 2.0);
		float tempOi32 = ((0.125 * (in[8] * in[8] * in[8] + in[2] * in[2] * in[2] + in[6] * in[6] * in[6] + in[0] * in[0] * in[0]) +
			0.75 * (in[7] * in[7] * in[7] + in[1] * in[1] * in[1]) - 0.25 * (in[7] * in[7] * in[7] + in[1] * in[1] * in[1]) - 1.5 * in[4] * in[4] * in[4]) / (x*x))
			+ ((0.125*(in[8] * in[8] * in[8] + in[2] * in[2] * in[2] + in[6] * in[6] * in[6] + in[0] * in[0] * in[0]) +
			0.75*(in[5] * in[5] * in[5] + in[3] * in[3] * in[3]) - 0.25*(in[5] * in[5] * in[5] + in[3] * in[3] * in[3]) - 1.5*in[4] * in[4] * in[4]) / (y*y));
		tempRecordOi32.elements[i] = tempOi32;
		__syncthreads();

		if (i == 9){
			printf("tempoi32:\n");
			for (int j = 0; j < 64; j++){
				printf("%f\t", tempRecordOi32.elements[j]);
				if (j % 8 == 7){
					printf("\n");
				}
			}
		}
		
	


		//float *twofoi = getFOI(tempRecordOi2, i);
		float *tin = getFOI(tempRecordOi2, i);
		//float tempOi4 = laplaceCal_r(twofoi, x, y, 2.0);
		float tempOi4 = ((0.125 * (tin[8] + tin[2] + tin[6] + tin[0]) +
			0.75 * (tin[7] + tin[1]) - 0.25 * (tin[7] + tin[1]) - 1.5 * tin[4]) / (x*x))
			+ ((0.125*(tin[8] + tin[2] + tin[6] + tin[0]) +
			0.75*(tin[5] + tin[3]) - 0.25*(tin[5] + tin[3]) - 1.5*tin[4]) / (y*y));
		tempRecordOi4.elements[i] = tempOi4;
		__syncthreads();

		if (i == 9){
			printf("tempoi4:\n");
			for (int j = 0; j < 64; j++){
				printf("%f\t", tempRecordOi4.elements[j]);
				if (j % 8 == 7){
					printf("\n");
				}
			}
		}

		

		/*if (i == 9){
			printf("tempoi2:\n");
			printf("%f %f %f\n", twofoi[0], twofoi[1], twofoi[2]);
			printf("%f %f %f\n", twofoi[3], twofoi[4], twofoi[5]);
			printf("%f %f %f\n", twofoi[6], twofoi[7], twofoi[8]);
		}
		*/
	
	


		//float *fourfoi = getFOI(tempRecordOi4, i);
		float *fin = getFOI(tempRecordOi4, i);
		//float tempOi6 = laplaceCal_r(fourfoi, x, y, 2.0);
		float tempOi6 = ((0.125 * (fin[8] + fin[2] + fin[6] + fin[0]) +
			0.75 * (fin[7] + fin[1]) - 0.25 * (fin[7] + fin[1]) - 1.5 * fin[4]) / (x*x))
			+ ((0.125*(fin[8] + fin[2] + fin[6] + fin[0]) +
			0.75*(fin[5] + fin[3]) - 0.25*(fin[5] + fin[3]) - 1.5*fin[4]) / (y*y));
		tempRecordOi6.elements[i] = tempOi6;
		if (i == 9){
			printf("tempoi6:\n");
			for (int j = 0; j < 64; j++){
				printf("%f\t", tempRecordOi6.elements[j]);
				if (j % 8 == 7){
					printf("\n");
				}
			}
		}
		newOi.elements[i] = oi.elements[i] + deltaT * ((1 - paraA) * tempOi2 + 2 * tempOi4 + tempOi6 + tempOi32);
	}



}


__global__ void allCal_new(Matrix newOi, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32){
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	float x = deltaX[0];
	float y = deltaY[0];
	__shared__ float temp2[SIZE];
	temp2[i] = oi.elements[i];
	__syncthreads();
	if (!(i%oi.width == 0 || i%oi.width == (oi.width - 1) || (i >= 0 && i <= (oi.width - 1)) ||
		(i <= (oi.height*oi.width - 1) && i >= (oi.height - 1)*oi.width) || i >= oi.height*oi.width)) {

		float *in = getFOIf(temp2, i, oi.width);
		float tempOi2 = ((0.125 * (in[8] + in[2] + in[6] + in[0]) +
			0.75 * (in[7] + in[1]) - 0.25 * (in[7] + in[1]) - 1.5 * in[4]) / (x*x))
			+ ((0.125*(in[8] + in[2] + in[6] + in[0]) +
			0.75*(in[5] + in[3]) - 0.25*(in[5] + in[3]) - 1.5*in[4]) / (y*y));
		tempRecordOi2.elements[i] = tempOi2;


		float tempOi32 = ((0.125 * (in[8] * in[8] * in[8] + in[2] * in[2] * in[2] + in[6] * in[6] * in[6] + in[0] * in[0] * in[0]) +
			0.75 * (in[7] * in[7] * in[7] + in[1] * in[1] * in[1]) - 0.25 * (in[7] * in[7] * in[7] + in[1] * in[1] * in[1]) - 1.5 * in[4] * in[4] * in[4]) / (x*x))
			+ ((0.125*(in[8] * in[8] * in[8] + in[2] * in[2] * in[2] + in[6] * in[6] * in[6] + in[0] * in[0] * in[0]) +
			0.75*(in[5] * in[5] * in[5] + in[3] * in[3] * in[3]) - 0.25*(in[5] * in[5] * in[5] + in[3] * in[3] * in[3]) - 1.5*in[4] * in[4] * in[4]) / (y*y));
		tempRecordOi32.elements[i] = tempOi32;

		__syncthreads();
	}

	__shared__ float temp4[SIZE];
	temp4[i] = tempRecordOi2.elements[i];
	__syncthreads();


	if (!(i%tempRecordOi2.width == 0 || i%tempRecordOi2.width == (tempRecordOi2.width - 1) || (i >= 0 && i <= (tempRecordOi2.width - 1)) ||
		(i <= (tempRecordOi2.height*tempRecordOi2.width - 1) && i >= (tempRecordOi2.height - 1)*tempRecordOi2.width) || i >= tempRecordOi2.height*tempRecordOi2.width)) {

		float *in = getFOIf(temp4, i, tempRecordOi2.width);
		float tempOi4 = ((0.125 * (in[8] + in[2] + in[6] + in[0]) +
			0.75 * (in[7] + in[1]) - 0.25 * (in[7] + in[1]) - 1.5 * in[4]) / (x*x))
			+ ((0.125*(in[8] + in[2] + in[6] + in[0]) +
			0.75*(in[5] + in[3]) - 0.25*(in[5] + in[3]) - 1.5*in[4]) / (y*y));
		tempRecordOi4.elements[i] = tempOi4;
		__syncthreads();

	}


	newOi.elements[i] = oi.elements[i];
	__shared__ float temp6[SIZE];
	temp6[i] = tempRecordOi4.elements[i];
	__syncthreads();

	if (!(i%oi.width == 0 || i%oi.width == (oi.width - 1) || (i >= 0 && i <= (oi.width - 1)) ||
		(i <= (oi.height*oi.width - 1) && i >= (oi.height - 1)*oi.width) || i >= oi.height*oi.width)) {



		float x = deltaX[0];
		float y = deltaY[0];


		float *in = getFOIf(temp6, i, oi.width);
		float tempOi6 = ((0.125 * (in[8] + in[2] + in[6] + in[0]) +
			0.75 * (in[7] + in[1]) - 0.25 * (in[7] + in[1]) - 1.5 * in[4]) / (x*x))
			+ ((0.125*(in[8] + in[2] + in[6] + in[0]) +
			0.75*(in[5] + in[3]) - 0.25*(in[5] + in[3]) - 1.5*in[4]) / (y*y));
		tempRecordOi6.elements[i] = tempOi6;
		__syncthreads();

		newOi.elements[i] = oi.elements[i] + deltaT * ((1 - paraA) * tempRecordOi2.elements[i] + 2 * tempRecordOi4.elements[i] + tempOi6 + tempRecordOi32.elements[i]);
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
		tempRecordOi2.elements[i] = laplaceCal(frontCal(foi), backCal(foi), 0.785, 0.785, 2.0);
		float *foithree = foiPowOf3(foi);
		tempRecordOi32.elements[i] = laplaceCal(frontCal(foithree), backCal(foithree), 0.785, 0.785, 2.0);
	}
	


		
	
}


__global__ void calOne(Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi32) {
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	/*if (i == 9){
	printf("oi!\n");
	for (int i = 0; i < 8; i++){
		for (int j = 0; j < 8; j++){
			printf("%f\t", oi.elements[i*(8) + j]);
		}
		printf("\n");
	}
}*/
	__shared__ float temp2[SIZE];
	temp2[i] = oi.elements[i];
	__syncthreads();
	if (!(i%oi.width == 0 || i%oi.width == (oi.width - 1) || (i >= 0 && i <= (oi.width - 1)) ||
		(i <= (oi.height*oi.width - 1) && i >= (oi.height - 1)*oi.width) || i >= oi.height*oi.width)) {		
		/*if (i == 9){
			printf("temp2 print!\n");
			for (int i = 0; i < 64; i++){
				printf("%f\t", temp2[i]);
				if (i % 8 == 7) printf("\n");
			}
		}
		*/
		float *in = getFOIf(temp2, i, oi.width);
		__syncthreads();
		/*
		if (i == 9){
			printf("i:%d check:%f \n", i,in[0]);
			printf("i:%d check:%f \n", i, in[1]);
			printf("i:%d check:%f \n", i, in[2]);
			printf("i:%d check:%f \n", i, in[3]);
			printf("i:%d check:%f \n", i, in[4]);
			printf("i:%d check:%f \n", i, in[5]);
			printf("i:%d check:%f \n", i, in[6]);
			printf("i:%d check:%f \n", i, in[7]);
			printf("i:%d check:%f \n", i, in[8]);
		}*/
		/*	printf("************getFOI***************\n");
		for (int i = 0; i < 9; i++){
		printf("foi[%d]:%f\t", i, foi[i]);
		}*/
		float x = deltaX[0];
		float y = deltaY[0];
		/*if (i == 9){
			printf("x:%f, y:%f \n",x,y);
		}*/
		//printf("deltaX=%f\tdeltaY=%f\n", deltaX, deltaY);
		//tempRecordOi2.elements[i] = laplaceCal_r(foi, x, y, 2.0);
		float tempOi2 = ((0.125 * (in[8] + in[2] + in[6] + in[0]) +
			0.75 * (in[7] + in[1]) - 0.25 * (in[7] + in[1]) - 1.5 * in[4]) / (x*x))
			+ ((0.125*(in[8] + in[2] + in[6] + in[0]) +
			0.75*(in[5] + in[3]) - 0.25*(in[5] + in[3]) - 1.5*in[4]) / (y*y));
		tempRecordOi2.elements[i] = tempOi2;		
		__syncthreads();		
		//printf("tempOi2:%f\t", tempOi2);
		//float *threefoi = foiPowOf3(in);
		//printf("in: %f, oi3:%f", in[5], threefoi[5]);
		//float tempOi32 = laplaceCal(frontCal(foi3), backCal(foi3), 0.785, 0.785, 2.0);
		//float tempOi32 = laplaceCal_r(threefoi, x, y, 2.0);
		float tempOi32 = ((0.125 * (in[8] * in[8] * in[8] + in[2] * in[2] * in[2] + in[6] * in[6] * in[6] + in[0] * in[0] * in[0]) +
			0.75 * (in[7] * in[7] * in[7] + in[1] * in[1] * in[1]) - 0.25 * (in[7] * in[7] * in[7] + in[1] * in[1] * in[1]) - 1.5 * in[4] * in[4] * in[4]) / (x*x))
			+ ((0.125*(in[8] * in[8] * in[8] + in[2] * in[2] * in[2] + in[6] * in[6] * in[6] + in[0] * in[0] * in[0]) +
			0.75*(in[5] * in[5] * in[5] + in[3] * in[3] * in[3]) - 0.25*(in[5] * in[5] * in[5] + in[3] * in[3] * in[3]) - 1.5*in[4] * in[4]*in[4]) / (y*y));
		tempRecordOi32.elements[i] = tempOi32;		
	}
}

__global__ void calTwo(Matrix tempRecordOi2, Matrix tempRecordOi4) {	
	int i = blockIdx.x *blockDim.x + threadIdx.x;	
	__shared__ float temp4[SIZE];
	temp4[i] = tempRecordOi2.elements[i];
	__syncthreads();
	if (!(i%tempRecordOi2.width == 0 || i%tempRecordOi2.width == (tempRecordOi2.width - 1) || (i >= 0 && i <= (tempRecordOi2.width - 1)) ||
		(i <= (tempRecordOi2.height*tempRecordOi2.width - 1) && i >= (tempRecordOi2.height - 1)*tempRecordOi2.width) || i >= tempRecordOi2.height*tempRecordOi2.width)) {

		
		float x = deltaX[0];
		float y = deltaY[0];

		
		//float *twofoi = getFOI(tempRecordOi2, i);
		//float tempOi4 = laplaceCal(frontCal(foi2), backCal(foi2), 0.785, 0.785, 2);
		float *in = getFOIf(temp4, i, tempRecordOi2.width);
		float tempOi4 = ((0.125 * (in[8] + in[2] + in[6] + in[0]) +
			0.75 * (in[7] + in[1]) - 0.25 * (in[7] + in[1]) - 1.5 * in[4]) / (x*x))
			+ ((0.125*(in[8] + in[2] + in[6] + in[0]) +
			0.75*(in[5] + in[3]) - 0.25*(in[5] + in[3]) - 1.5*in[4]) / (y*y));
		tempRecordOi4.elements[i] = tempOi4;
		__syncthreads();
		//float tempOi4 = laplaceCal_r(twofoi, x, y, 2.0);
		//tempRecordOi4.elements[i] = tempOi4;
		//printf("tempOi4:%f\t", tempOi4);
		
	}
}


__global__ void callThree(Matrix newOi, Matrix oi, Matrix tempRecordOi2, Matrix tempRecordOi4, Matrix tempRecordOi6, Matrix tempRecordOi32) {
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	//printf("i: %d\t", i);

	newOi.elements[i] = oi.elements[i];
	__shared__ float temp6[SIZE];
	temp6[i] = tempRecordOi4.elements[i];
	__syncthreads();

	if (!(i%oi.width == 0 || i%oi.width == (oi.width - 1) || (i >= 0 && i <= (oi.width - 1)) ||
		(i <= (oi.height*oi.width - 1) && i >= (oi.height - 1)*oi.width) || i >= oi.height*oi.width)) {
		float x = deltaX[0];
		float y = deltaY[0];		
		//float *fourfoi = getFOI(tempRecordOi4, i);
		//float tempOi6 = laplaceCal(frontCal(foi4), backCal(foi4), 0.785, 0.785, 2);
		//float tempOi6 = laplaceCal_r(fourfoi, x, y, 2.0);
		//printf("tempOi6:%f\t", tempOi6);
		//tempRecordOi6.elements[i] = tempOi6;

		float *in = getFOIf(temp6, i, oi.width);
		float tempOi6 = ((0.125 * (in[8] + in[2] + in[6] + in[0]) +
			0.75 * (in[7] + in[1]) - 0.25 * (in[7] + in[1]) - 1.5 * in[4]) / (x*x))
			+ ((0.125*(in[8] + in[2] + in[6] + in[0]) +
			0.75*(in[5] + in[3]) - 0.25*(in[5] + in[3]) - 1.5*in[4]) / (y*y));
		tempRecordOi6.elements[i] = tempOi6;
		__syncthreads();

		newOi.elements[i] = oi.elements[i] + deltaT * ((1 - paraA) * tempRecordOi2.elements[i] + 2 * tempRecordOi4.elements[i] + tempOi6 + tempRecordOi32.elements[i]);
	}
}