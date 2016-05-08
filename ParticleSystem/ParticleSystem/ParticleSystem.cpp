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
#ifdef SPIRAL_DIST
	//spiral distribution using polar equation r(t)=e^(theta*b) modified for variety in branch 
	//density is 1/(current theta/(pi/2)
	//using formula from this study https://arxiv.org/ftp/arxiv/papers/0908/0908.0892.pdf

	//constants for equation
	float scale = 500;
	float N = 4;
	float B = .5;

	int numArms = 2;	//this is how many times the curve will be rotated and populated

	float intervalStart = 0;
	float intervalEnd = PI;
	float intervalLength = (intervalEnd - intervalStart);

	float particleStart = 0;
	float particleEnd = numParticles;

	for (int armIt = 0; armIt < numArms; armIt++)
	{
		//seperate rotations must be performed for each arm
		float rotation = armIt * (2*PI / numArms);	//equally spaced rotations around circle
		float cr = cos(rotation);
		float sr = sin(rotation);

		particleEnd = .5 * (armIt + 1) * numParticles / numArms;
		particleStart = .5 * (armIt) * numParticles / numArms;

		float step = intervalLength/(particleEnd-particleStart);
		float cTheta = 0;	//current theta


		for (int partIt = particleStart; partIt < particleEnd; partIt++)
		{
			//step = intervalLength / (particleEnd - partIt);
			int index = partIt * 3;

			//find r
			float r = scale / (B * log(tan(cTheta / (2 * N))));

			//calculate x and y coordinates
			p_type x = cos(cTheta) * r;
			p_type y = sin(cTheta) * r;

			//rotate point around origin
			x = x*cr - y * sr;
			y = x*sr + y*cr;

			//apply variation
			p_type xVary = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX)* 1000;	//400 pixel width of arm
			p_type yVary = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX)* 1000;

			p_type z = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX)* 1000;	//const depth of arm

			//change step to reflect particle density
			//step = intervalLength / (particleEnd);// -partIt);


			
			//initialize particle
			h_pos[index] = x+xVary;
			h_pos[index + 1] = y+yVary;
			h_pos[index + 2] = z;

			auto tang = getTangentO(h_pos, index);
			p_type dist = getMagO(h_pos, index);
			p_type oV = sqrt(EARTH_KG*2 / (dist * 100));
			h_vel[index] = std::get<0>(tang) * oV;
			h_vel[index + 1] = std::get<1>(tang) * oV;
			h_vel[index + 2] = std::get<2>(tang) * oV;

			h_acc[index] = 0;
			h_acc[index + 1] = 0;
			h_acc[index + 2] = 0;

			h_mass[partIt] = EARTH_KG;

			//increment theta
			cTheta += step;

		}

	}
	/*
	size_t size = sizeof(p_type) * 3 * numParticles;
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_vel, h_vel, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_acc, h_acc, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_mass, h_mass, size/3, cudaMemcpyHostToDevice);
	*/
//	free(h_pos);
//	h_pos = (p_type*)malloc(size);

#endif
#ifdef DISK_DIST
	float maxRadius = 3000;
	for (unsigned int partIt = numParticles/2; partIt < numParticles; partIt++)
	{

		int index = partIt * 3;
		float angle = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX)* 2* PI ;
		float r = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX)* (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX)*   (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * maxRadius;

		p_type x = cos(angle) * r;
		p_type y = sin(angle) * r;
		p_type z = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX)* 2000;

		h_pos[index] = x;
		h_pos[index + 1] = y;
		h_pos[index + 2] = z;

		auto tang = getTangentO(h_pos, index);
		p_type dist = getMagO(h_pos, index);
		p_type oV = sqrt(EARTH_KG*2 / (dist * 100));
		h_vel[index] = std::get<0>(tang) * oV;
		h_vel[index + 1] = std::get<1>(tang) * oV;
		h_vel[index + 2] = std::get<2>(tang) * oV;

		h_acc[index] = 0;
		h_acc[index + 1] = 0;
		h_acc[index + 2] = 0;

		h_mass[partIt] = EARTH_KG;
	}

	size_t size = sizeof(p_type) * 3 * numParticles;
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_vel, h_vel, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_acc, h_acc, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_mass, h_mass, size/3, cudaMemcpyHostToDevice);
#else
	for (unsigned int partIt = 0; partIt < numParticles; partIt++)
	{

		int index = partIt * 3;
		p_type x = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 40000;
		p_type y = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 40000;
		p_type z = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 20000;
		h_pos[index] = x;
		h_pos[index + 1] = y;
		h_pos[index + 2] = z;

		h_vel[index] = 0;
		h_vel[index + 1] = 0;
		h_vel[index + 2] = 0;

		h_acc[index] = 0;
		h_acc[index + 1] = 0;
		h_acc[index + 2] = 0;

		h_mass[partIt] = EARTH_KG;
	}
#endif
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

				if (distsqr > -.01 && distsqr < .01)	//want to prevent errors and simulate collision
				{
					//add mass to other particle
					h_mass[partItA] += h_mass[partItB];
					h_mass[partItB] = 0;

					//move it out of view
					h_pos[indexB] = 100000;
					h_pos[indexB + 1] = 100000;
					h_pos[indexB + 2] = 100000;
				}
				else
				{
					p_type attraction = (h_mass[partItA] * h_mass[partItB]) / (distsqr * 1000000000000000000);	//gravity equation

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

