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
	}
}

void ParticleSystem::allocate(const unsigned int& numParticles)
{
	//define distribution here or something
	if (numParticles > 0)
	{
		int pointsPerParticleVec = 3;
		size_t size = sizeof(double) * pointsPerParticleVec * numParticles;

		h_pos = (double*)malloc(size);
		h_vel = (double*)malloc(size);
		h_acc = (double*)malloc(size);
		h_mass = (double*)malloc(size / pointsPerParticleVec);

		d_pos = NULL;
		d_vel = NULL;
		d_acc = NULL;

		allocated = true;
	}

}

double* ParticleSystem::getParticleVector()
{
	return h_pos;
}
void ParticleSystem::initialize(/*distribution type?*/)
{

}

void ParticleSystem::doFrameCPU()
{
	for (unsigned int partItA = 0; partItA < numParticles; partItA++)
	{
		int indexA = partItA * 3;
		for (unsigned int partItB = 0; partItB < numParticles; partItB++)
		{
			int indexB = partItB * 3;	//index of x coord in vector

			double diffx = (h_pos[indexB] - h_pos[indexA]);			//calculating difference between points
			double diffy = (h_pos[indexB + 1] - h_pos[indexA + 1]);
			double diffz = (h_pos[indexB + 2] - h_pos[indexA + 2]);

			double distsqr = diffx*diffx + diffy*diffy + diffz*diffz;

			double attraction = (9.81 * h_mass[partItA] * h_mass[partItB]) / distsqr;	//gravity equation
			double invsqrt = fInvSqrt((float)distsqr);
			double normx = invsqrt*diffx;
			double normy = invsqrt*diffy;
			double normz = invsqrt*diffz;

			double forcex = normx * attraction;
			double forcey = normy * attraction;
			double forcez = normz * attraction;

			h_acc[indexB] += forcex;
			h_acc[indexB+1] += forcey;
			h_acc[indexB + 2] += forcez;

		}
	}

	for (unsigned int partIt = 0; partIt < numParticles; partIt++)
	{
		int index = partIt * 3;

		h_vel[index] += h_acc[index];
		h_vel[index+1] += h_acc[index+1];
		h_vel[index+2] += h_acc[index+2];

		h_pos[index] += h_vel[index];
		h_pos[index+1] += h_vel[index+1];
		h_pos[index+2] += h_vel[index+2];

		h_acc[index] = 0.0f;
		h_acc[index + 1] = 0.0f;
		h_acc[index + 2] = 0.0f;
	}
}

void ParticleSystem::doFrameGPU()
{

}

//PRIVATE FUNCTIONS

float ParticleSystem::fInvSqrt(const float& in)
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
	return y;
}