#include "ParticleSystem.hpp"

ParticleSystem::ParticleSystem()
{
	allocated = false;
}

ParticleSystem::~ParticleSystem()
{
	if (allocated)
	{
		free(h_pos);
		free(h_vel);
		free(h_acc); 

		cudaFree(d_pos);
		cudaFree(d_vel);
		cudaFree(d_acc);
		cudaFree(d_mass);
	}
}

void ParticleSystem::allocate(const unsigned int& newNumParticles)
{
	//define distribution here or something
	numParticles = newNumParticles;
	if (numParticles > 0)
	{
		int pointsPerParticleVec = 3;
		size_t size = sizeof(p_type) * pointsPerParticleVec * numParticles;

		h_pos = (p_type*)malloc(size);
		h_vel = (p_type*)malloc(size);
		h_acc = (p_type*)malloc(size);
		h_mass = (p_type*)malloc(size / pointsPerParticleVec);

		d_pos = NULL;
		d_vel = NULL;
		d_acc = NULL;
		cudaError_t err = cudaSuccess;
		//allocate space on GPU
		err = cudaMalloc((void **)&d_pos, size);
		err  = cudaMalloc((void **)&d_vel, size);
		err = cudaMalloc((void **)&d_acc, size);
		err = cudaMalloc((void **)&d_mass, size/3);

		//copy from cpu to GPU
		
		printf("ERROR!!!!???? %s", cudaGetErrorString(err));
		allocated = true;
	}

}

p_type* ParticleSystem::getHostParticleVector()
{
	return h_pos;
}

p_type* ParticleSystem::getDeviceParticleVector()
{
	return d_pos;
}
void ParticleSystem::initialize(/*distribution type?*/)
{
	module::Perlin noiseMaker;
	noiseMaker.SetOctaveCount(1);
	noiseMaker.SetFrequency(.001);
	noiseMaker.SetPersistence(.5);


	utils::NoiseMap nm;

	float xDim = 15000;
	float yDim = 15000;
	float zDim = 15000;

	float crt = cbrt(numParticles*30);
	float dx = xDim/crt;
	float dy = yDim/crt;
	float dz = zDim/crt;

	int particlesGenerated = 0;
	for (float xIt = -xDim/2; xIt <= xDim/2; xIt += dx)
	{
		for (float yIt = -yDim/2; yIt <= yDim/2; yIt += dy)
		{
			for (float zIt = -zDim/2; zIt <= zDim/2; zIt += dz)
			{
				if (particlesGenerated < numParticles)
				{
					double p = noiseMaker.GetValue(xIt/3, yIt/3, zIt/3);
					if (p < 0)
					{
						p = 0;
					}
					p = pow(p, 3);
					if (random(1) < p)
					{
						int index = particlesGenerated * 3;
							
						h_pos[index] = xIt + random(dx);
						h_pos[index + 1] = yIt + random(dy);
						h_pos[index + 2] = random(200);

						h_vel[index] = 0;
						h_vel[index + 1] = 0;
						h_vel[index + 2] = 0;

						h_acc[index] = 0;
						h_acc[index + 1] = 0;
						h_acc[index + 2] = 0;

						h_mass[particlesGenerated] = EARTH_KG;

						particlesGenerated++;
					}
				}
			}
		}
	}


	size_t size = sizeof(p_type) * 3 * numParticles;
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_vel, h_vel, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_acc, h_acc, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_mass, h_mass, size/3, cudaMemcpyHostToDevice);


}

void ParticleSystem::doFrameCPU()
{
	for (unsigned int partItA = 0; partItA < numParticles; partItA++)
	{
		int indexA = partItA * 3;
		for (unsigned int partItB = 0; partItB < numParticles; partItB++)
		{
			
			int indexB = partItB * 3;	//index of x coord in vector

			if (indexA != indexB)
			{
				p_type diffx = (h_pos[indexB] - h_pos[indexA]);			//calculating difference between points
				p_type diffy = (h_pos[indexB + 1] - h_pos[indexA + 1]);
				p_type diffz = (h_pos[indexB + 2] - h_pos[indexA + 2]);

				p_type distsqr = abs(diffx*diffx + diffy*diffy + diffz*diffz);
				if (distsqr < 6000)
				{
					distsqr = 6000;
				}


					p_type attraction = (h_mass[partItA] * h_mass[partItB]) / (distsqr);	//gravity equation

					p_type invsqrt = fInvSqrt((float)distsqr);
					p_type normx = invsqrt*diffx;
					p_type normy = invsqrt*diffy;
					p_type normz = invsqrt*diffz;

					p_type forcex = normx * -attraction;
					p_type forcey = normy * -attraction;
					p_type forcez = normz * -attraction;

					h_acc[indexB] += forcex;
					h_acc[indexB + 1] += forcey;
					h_acc[indexB + 2] += forcez;

			}
		}

		for (unsigned int partIt = 0; partIt < numParticles; partIt++)
		{
			int index = partIt * 3;

			h_vel[index] += h_acc[index];
			h_vel[index + 1] += h_acc[index + 1];
			h_vel[index + 2] += h_acc[index + 2];

			h_pos[index] += h_vel[index];
			h_pos[index + 1] += h_vel[index + 1];
			h_pos[index + 2] += h_vel[index + 2];

			h_acc[index] = 0.0f;
			h_acc[index + 1] = 0.0f;
			h_acc[index + 2] = 0.0f;
		}
	}
}

void ParticleSystem::doFrameGPU()
{
	int numBlocks = (numParticles + TPB - 1) / TPB;
	int numBlocks2 = numBlocks = (numParticles * 3 + TPB - 1) / TPB;
	doFrame(d_pos, d_vel, d_acc, d_mass, numParticles, numBlocks, numBlocks2);

	//copy vector back to cpu (until opengl-cuda gets implemented)
	cudaMemcpy(h_pos, d_pos, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);

}

int ParticleSystem::getNumParticles()
{
	return numParticles;
}
//PRIVATE FUNCTIONS

