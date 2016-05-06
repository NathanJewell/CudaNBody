

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <chrono>

#include "ParticleRenderer.hpp"
#include "ParticleSystem.hpp"

typedef std::chrono::high_resolution_clock Clock;



__global__ void ARR_ADD(float* res, const float* in1, const float *in2, int n)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < n)
	{
		res[index] = sqrt(in1[index] * in2[index]);
	}

}

int main(int argc, char** argv)
{
	ParticleSystem sys;
	ParticleRenderer ren;

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

	glutInitWindowSize(250, 250);

	glutInitWindowPosition(100, 100);

	glutCreateWindow("N-BODY");

	glutDisplayFunc(ParticleRenderer::drawFrame);

	
	glewInit();




	ren.initGL();




	sys.allocate(5000);

	ren.setParticleVector(sys.getParticleVector());


	glutMainLoop();



	/*
	//defining program attributes
	int N = 10000000;
	size_t size = N * sizeof(float);
	int TPB = 1024;	//threads per block
	int times = 100;


	//allocating host memory


	float *res = (float *)malloc(size);
	float *in1 = (float *)malloc(size);
	float *in2 = (float *)malloc(size);

	//defining pointers to device memory 
	float *d_res = NULL;
	float *d_in1 = NULL;
	float *d_in2 = NULL;

	//initializing values
	for (int i = 0; i < N; i++)
	{
		in1[i] = rand() / (float)RAND_MAX;
		in2[i] = rand() / (float)RAND_MAX;
	}
	auto t1 = Clock::now();
	cudaMalloc((void**)&d_res, size);
	cudaMalloc((void**)&d_in1, size);
	cudaMalloc((void**)&d_in2, size);
	//copying to device memory
	cudaError_t err = cudaMemcpy(d_res, res, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);

	//call function to execute on device

	int numblocks = (N + TPB - 1) / TPB;
	for (int i = 0; i < times; i++)
	{
		ARR_ADD << <numblocks, TPB >> >(d_res, d_in1, d_in2, N);
	}

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//copy result back to host

	err = cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);

	//free memory on device
	cudaFree(d_res);
	cudaFree(d_in1);
	cudaFree(d_in2);

	auto t2 = Clock::now();
	std::cout << "Delta t2-t1: "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
		<< " nanoseconds" << std::endl;

	//print result to window
	double checksum = 0;
	for (int i = 0; i < N; i++)
	{
		checksum += res[i];
	}


	std::cout << "Parrelelized N=" << N << " : " << checksum << std::endl;
	float *in1N = in2;
	float *in2N = in1;
	//free host memory
	free(res);




	auto t1N = Clock::now();

	float *resN = (float*)malloc(size);


	for (int ii = 0; ii < times; ii++)
	{
		for (int i = 0; i < N; i++)
		{
			resN[i] = sqrt(in1N[i] * in2N[i]);
		}
	}


	auto t2N = Clock::now();
	std::cout << "Delta t2-t1: "
		<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2N - t1N).count()
		<< " nanoseconds" << std::endl;

	checksum = 0;
	for (int i = 0; i < N; i++)
	{
		checksum += resN[i];
	}
	std::cout << "Standard N=" << N << " : " << checksum << std::endl;

	free(in1);
	free(in2);
	*/
}