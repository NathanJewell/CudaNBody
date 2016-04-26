
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

//defining program attributes
static const int N = 100;
static const size_t size = N * sizeof(int);
static const int TPB = 512;	//threads per block

__global__ void ARR_ADD(int* res, int* in1, int *in2, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n)
		res[index] = in1[index] + in2[index];
}

int main()
{



	//allocating host memory
	int *res = (int*) malloc(size);
	int *in1 = (int*) malloc(size);
	int *in2 = (int*) malloc(size);

	//defining pointers to device memory 
	int *d_res, *d_in1, *d_in2;

	cudaMalloc(&d_res, size);
	cudaMalloc(&d_in1, size);
	cudaMalloc(&d_in2, size);

	//initializing values
	for (int i = 0; i < N; i++)
	{
		in1[i] = i + 1;
		in2[i] = i + 2;
	}


	//copying to device memory
	cudaMemcpy(d_res, res, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

	//call function to execute on device
	ARR_ADD << <N/TPB, TPB>> >(d_res, d_in1, d_in2, N);

	//copy result back to host
	cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);

	//free memory on device
	cudaFree(d_res);
	cudaFree(d_in1);
	cudaFree(d_in2);

	//print result to window
	long int checksum = 0;
	for (int i = 0; i < N; i++)
	{
		checksum += res[i];
	}
	std::cout << "Parrelelized N=" << N << " : " << checksum << std::endl;

	//free host memory
	delete res;
	delete in1;
	delete in2;

	int resN[N], in1N[N], in2N[N];
	for (int i = 0; i < N; i++)
	{
		in1N[i] = i + 1;
		in2N[i] = i + 2;
	}
	for (int i = 0; i < N; i++)
	{
		resN[i] = in1N[i] + in2N[i];
	}
	checksum = 0;
	for (int i = 0; i < N; i++)
	{
		checksum += resN[i];
	}
	std::cout << "Standard N=" << N << " : " << checksum << std::endl;

}