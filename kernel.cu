
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrixmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernelFunctions.cu"

#define PI 3.14159265
#define deltaT 0.0075
#define paraA 0.35

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, float init);

__constant__ float deltaX;
__constant__ float deltaY;

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void initialization()
{

}
int main()
{
	
	float oi0 = 0.3;
	float qh = sqrt(3.0)/2;
	int optionOi = 1;
	int areaX = 5;
	int areaY = 5;
	int numT = 15;
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

	Matrix u = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 0));

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

		Matrix oi = AllocateDeviceMatrix(oi_host);

		Matrix tempRecordOi2 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 1.0));
		Matrix tempRecordOi4 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 1.0));
		Matrix tempRecordOi6 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 1.0));
		Matrix tempRecordOi32 = AllocateDeviceMatrix(AllocateMatrix(numX + 2, numY + 2, 1.0));

		float *tempx = (float*)malloc(sizeof(float));
		float *tempy = (float*)malloc(sizeof(float));

		*tempx = deltaX0 + i*strainR*deltaT;
		*tempy = deltaY0*(deltaX0 / *tempx);
		
		printf("tempx:%f\n", *tempx);
		printf("tempy:%f\n", *tempy);

		cudaMemcpyToSymbol(&deltaX, tempx, sizeof(float));
		cudaMemcpyToSymbol(&deltaY, tempy, sizeof(float));




	}





    return 0;
}

Matrix AllocateDeviceMatrix(const Matrix M)
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

Matrix AllocateMatrix(int height, int width, float init)
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;

	M.elements = (float*)malloc(size*sizeof(float));

	for (unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = init;
	}
	return M;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
