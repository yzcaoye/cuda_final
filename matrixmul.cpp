#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cuda_runtime.h>

#include "matrixmul.h"

void cudasafe(cudaError_t error, char* message){
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s : %i\n", message, error); exit(-1);
	}
}

void Check_CUDA_Error(const char *message){
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
		exit(-1);
	}
}

Matrix AllocateDeviceMatrix(const Matrix M)
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudasafe(cudaMalloc((void**)&Mdevice.elements, size), "cudaMalloc");
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

void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size,
		cudaMemcpyHostToDevice);
}

void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size,
		cudaMemcpyDeviceToHost);
}

void FreeDeviceMatrix(Matrix* M)
{
	cudasafe(cudaFree(M->elements), "cudaFree");
	M->elements = NULL;
}

void FreeMatrix(Matrix* M)
{
	free(M->elements);
	M->elements = NULL;
}