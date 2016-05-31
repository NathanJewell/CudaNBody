#include "math.cuh"


__device__ float fInvSqrt_D(const float& in)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = in * 0.5F;
	y = in;
	i = *(long *)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float *)&i;
	y = y * (threehalfs - (x2 * y * y));
	y = y * (threehalfs - (x2 * y * y));	//extra precision
	return abs(y);
}


__device__ void doParticle(p_type* pos, p_type* vel, p_type* acc, p_type* mass, int numParticles, int pIndex2, int index2, int thisIndex, float tstep)
{
	int index = thisIndex;
	int pIndex1 = index * 3;
	//printf("index1 %d \n", index);
	//printf("index2 %d \n", index2);
	if (pIndex1 != pIndex2 && index < numParticles)
	{

		p_type diffx = (pos[pIndex1] - pos[pIndex2]);			//calculating difference between points
		p_type diffy = (pos[pIndex1 + 1] - pos[pIndex2 + 1]);
		p_type diffz = (pos[pIndex1 + 2] - pos[pIndex2 + 2]);

		p_type distsqr = diffx*diffx + diffy*diffy + diffz*diffz;

		if (distsqr < 0)
		{
			distsqr *= -1;
		}
		if (distsqr < 500)
		{
			distsqr = 500;
		}
		//else
		//{

		p_type attraction = (mass[index2] * mass[index]) / (distsqr);	//gravity equation


			p_type invsqrt = fInvSqrt_D((float)distsqr);
			p_type normx = invsqrt*diffx;
			p_type normy = invsqrt*diffy;
			p_type normz = invsqrt*diffz;

			p_type forcex = normx * -attraction;
			p_type forcey = normy * -attraction;
			p_type forcez = normz * -attraction;

			acc[pIndex1] += (forcex * tstep)/mass[index];
			acc[pIndex1 + 1] += (forcey * tstep)/mass[index];
			acc[pIndex1 + 2] += (forcez * tstep)/mass[index];
		//}

	}
}

__global__ void doParticles(p_type* pos, p_type* vel, p_type* acc, p_type* mass, int numParticles, float tstep)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	int Y = threadId /numParticles;	//slower changing iterator
	int X = threadId %numParticles;	//fast changing iterator

	int pIndex1 = X * 3;
	int pIndex2 =Y * 3;
	//printf("index1 %d \n", index);
	//printf("index2 %d \n", index2);
	if (pIndex1 != pIndex2 && Y < numParticles)
	{

		p_type diffx = (pos[pIndex1] - pos[pIndex2]);			//calculating difference between points
		p_type diffy = (pos[pIndex1 + 1] - pos[pIndex2 + 1]);
		p_type diffz = (pos[pIndex1 + 2] - pos[pIndex2 + 2]);

		p_type distsqr = diffx*diffx + diffy*diffy + diffz*diffz;

		if (distsqr < 0)
		{
			distsqr *= -1;
		}
		if (distsqr < 500)
		{
			distsqr = 500;
		}

		//else
		//{

		p_type attraction = (mass[X] * mass[Y]) / (distsqr);	//gravity equation


		p_type invsqrt = fInvSqrt_D((float)distsqr);
		p_type normx = invsqrt*diffx;
		p_type normy = invsqrt*diffy;
		p_type normz = invsqrt*diffz;

		p_type forcex = normx * -attraction;
		p_type forcey = normy * -attraction;
		p_type forcez = normz * -attraction;

		acc[pIndex1] += (forcex * tstep) / mass[X];
		acc[pIndex1 + 1] += (forcey * tstep) / mass[X];
		acc[pIndex1 + 2] += (forcez * tstep) / mass[X];
		//}

	}
}

__global__ void beginFrame(p_type* pos, p_type* vel, p_type* acc, p_type* mass, int numParticles, int numBlocks, float dt)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int pIndex1 = index * 3;

	if (index < numParticles)
	{
		for (int i = 0; i < numParticles; i++)
		{
			doParticle(pos, vel, acc, mass, numParticles, pIndex1, index, i, dt);
		}
		//pos[index] = 0;
	}

}


__global__ void ARR_ADD(p_type* getter, const p_type *giver, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		getter[index] = getter[index] + giver[index];
	}

}

__global__ void POS_ADD(p_type* getter, const p_type *giver, int N, float dt)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		getter[index] = getter[index] + (giver[index]*dt);
	}

}


__global__ void ARR_ADDC(float* result, float* in1, float* in2, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		result[index] = in1[index] + in2[index];
	}
}


__global__ void ARR_SET(p_type* getter, const p_type value, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N)
	{
		getter[index] = value;
	}
}

__host__ void doFrame(p_type* d_pos, p_type* d_vel, p_type* d_acc, p_type* d_mass, int numParticles)
{
	float dt = .5;

	dim3 block(TPB2d, TPB2d);
	dim3 grid;
	grid.x = (numParticles + TPB2d - 1) / TPB2d;
	grid.y = (numParticles + TPB2d - 1) / TPB2d;

	//beginFrame << <numBlocks, TPB >> >(d_pos, d_vel, d_acc, d_mass, numParticles, numBlocks, dt);
	cudaError_t err = cudaSuccess;
	doParticles<<< grid, block >>>(d_pos, d_vel, d_acc, d_mass, numParticles, dt);
	
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
		cudaDeviceSynchronize();

	int numBlocks2 = (numParticles * 3 + TPB - 1) / TPB;
	ARR_ADD << <numBlocks2, TPB >> >(d_vel, d_acc, numParticles * 3);

	//p_type* test;
	//test = (p_type*)malloc(sizeof(p_type) * 3 * numParticles);
	//cudaMemcpy(test, d_pos, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(test, d_vel, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(test, d_acc, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaDeviceSynchronize();
	ARR_SET << <numBlocks2, TPB >> >(d_acc, 0.0f, numParticles * 3);

	//cudaMemcpy(test, d_pos, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(test, d_vel, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(test, d_acc, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);

	POS_ADD << <numBlocks2, TPB >> >(d_pos, d_vel, numParticles * 3, dt);

	//cudaMemcpy(test, d_pos, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(test, d_vel, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(test, d_acc, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}