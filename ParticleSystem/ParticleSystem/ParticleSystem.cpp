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

		particleEnd = (armIt + 1) * numParticles / numArms;
		particleStart = (armIt) * numParticles / numArms;

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
	float maxRadius = 1000;
	float depth = 500;
	long long massAvg = 1;
	for (unsigned int partIt = 0; partIt < numParticles; partIt++)
	{

		int index = partIt * 3;
		float angle = (p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX)* 2* PI ;

		float rp = partIt - numParticles;

		float power = 0.0;
		float rstep = (maxRadius/numParticles) * partIt;
		float r = rstep;// numParticles / (log(maxRadius)*rstep);//(maxRadius * (pow(partIt, power) / pow(numParticles, power)));
		//float r = (((p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX))*maxRadius)

		p_type x = cos(angle) * r;
		p_type y = sin(angle) * r;
		p_type z = pow((p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX), 1.2) * depth;

		h_pos[index] = x;
		h_pos[index + 1] = y;
		h_pos[index + 2] = z;

		auto tang = getTangentO(h_pos, index);
		p_type distsqr = getMagO(h_pos, index);
		p_type invDist = fInvSqrt((float)distsqr);
		p_type dist = 1 / invDist;

		h_mass[partIt] = EARTH_KG;// ((pow((p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX), 1) * 100) + EARTH_KG) / dist;
		massAvg += h_mass[partIt]/distsqr;

		p_type Rm = massAvg;


		p_type oV = sqrt(Rm / dist);
		//p_type oV = 2*PI; // dist / pow(1 + distsqr, 3 / 4);
		h_vel[index] =  std::get<0>(tang) * oV;
		h_vel[index + 1] =  std::get<1>(tang) * oV;
		h_vel[index + 2] =  std::get<2>(tang) * oV;

		h_acc[index] = 0;
		h_acc[index + 1] = 0;
		h_acc[index + 2] = 0;


	}

	size_t size = sizeof(p_type) * 3 * numParticles;
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_vel, h_vel, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_acc, h_acc, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_mass, h_mass, size/3, cudaMemcpyHostToDevice);
#endif
	float maxRadius = 4000;
	float maxDepth = 500;
	float k = numParticles / 49.147007;// 564.319;	//k for 1/ln(r)
	//float k = numParticles / log(maxRadius);			//k for 1/r
	float lee = li(4000, 100);
	float dr = maxRadius / 200;
	float radiusVariationFrac = 1;
	int particlesGenerated = 0;
	float totalMass = 0;

	for (float r = 2; r < maxRadius+.01; r += dr)
	{
		int toGen = 0;
		if (r > maxRadius)
		{
			toGen = numParticles - particlesGenerated;
		}
		else
		{
			toGen = dr * ((1 / log(r) + (1 / log(r + dr))) / 2) * k + 1;	//trapezoid method for integral approximation 1/ln(r)		
			//toGen = dr * (((1 / r)+(1/(r+dr)))/2) * k + 1;					//trapezoid method for integral approximation 1/r
		}

		for (int partIt = 0; partIt < toGen && particlesGenerated < numParticles; partIt++, particlesGenerated++)
		{
			int index = particlesGenerated * 3;


			float angle = random(2*PI);
			float radius = random(dr*radiusVariationFrac, r);

			p_type x = cos(angle) * r;
			p_type y = sin(angle) * r;
			p_type z = pow(random(1), 1.2) * maxDepth;

			h_pos[index] = x;
			h_pos[index + 1] = y;
			h_pos[index + 2] = z;

			auto tang = getTangentO(h_pos, index);
			p_type distsqr = getMagO(h_pos, index);
			p_type invDist = fInvSqrt((float)distsqr);
			p_type dist = 1 / invDist;

			h_mass[partIt] = EARTH_KG/distsqr;// ((pow((p_type)static_cast <float> (rand()) / static_cast <float> (RAND_MAX), 1) * 100) + EARTH_KG) / dist;
			totalMass += h_mass[partIt] / distsqr;

			p_type Rm =  totalMass;


			p_type oV = sqrt(Rm / dist);
			//p_type oV = 2*PI; // dist / pow(1 + distsqr, 3 / 4);
			h_vel[index] = std::get<0>(tang) * oV;
			h_vel[index + 1] = std::get<1>(tang) * oV;
			h_vel[index + 2] = std::get<2>(tang) * oV;

			h_acc[index] = 0;
			h_acc[index + 1] = 0;
			h_acc[index + 2] = 0;
		}
	}
	size_t size = sizeof(p_type) * 3 * numParticles;
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_vel, h_vel, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_acc, h_acc, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_mass, h_mass, size / 3, cudaMemcpyHostToDevice);
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
				if (distsqr < 30000)
				{
					distsqr = 30000;
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

