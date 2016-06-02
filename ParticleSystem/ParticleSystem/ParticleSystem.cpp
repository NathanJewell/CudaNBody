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

	float noiseScale = 3;
	utils::NoiseMap nm;

	float xDim = 300000;
	float yDim = 300000;
	float zDim = 300000;
	float radius = 300000 / 2;

	float crt = cbrt(numParticles*30);
	float dx = xDim/crt;
	float dy = yDim/crt;
	float dz = zDim/crt;
	float px, py, pz, vx, vy, vz, calcR2, r2;
	int particlesGenerated = 0;
	r2 = radius*radius;
	
	int numSpheres = 70;
	int pInSphere = numParticles/numSpheres;
	int sSize = 100000;
	int sVariance = 100000;

	int rSize = 100000;
	int rVariance = 75000;
	for (int sIt = 0; sIt < numSpheres; sIt++)
	{
		double ty, tx, tz;	//translation factors
		
		tx = random(sSize, -sSize/2);
		ty = random(sSize, -sSize/2);
		tz = random(sSize, -sSize/2);

		radius = random(rVariance * 2, rSize - rVariance);
		r2 = radius*radius;
		
		pInSphere = random(numParticles / numSpheres, numParticles / (2*numSpheres));

		for (int i = 0; i < pInSphere && particlesGenerated < numParticles; i++)
		{
			if (sIt == numSpheres - 1)
			{
				pInSphere = numParticles;
			}
			int index = particlesGenerated * 3;

		here:
			px = h_pos[index] = random(radius * 2, -radius);
			py = h_pos[index + 1] = random(radius * 2, -radius);
			pz = h_pos[index + 2] = random(radius * 2, -radius);

			//px -= tx; py -= ty; pz -= tz;
			calcR2 = px*px + py*py + pz*pz;
			if (calcR2 > r2)
			{
				goto here;
			}

			h_pos[index] += tx;
			h_pos[index + 1] += ty;
			h_pos[index + 2] += tz;

			vx = h_vel[index] = random(50, -25);
			vy = h_vel[index + 1] = random(50, -25);
			vz = h_vel[index + 2] = random(12.5, -6);
			h_acc[index] = 0;
			h_acc[index + 1] = 0;
			h_acc[index + 2] = 0;

			h_mass[particlesGenerated] = EARTH_KG + random(EARTH_KG * 100);

			particlesGenerated++;

		}
	}

	
	/*
	for (float xIt = -xDim/2; xIt <= xDim/2; xIt += dx)
	{
		for (float yIt = -yDim/2; yIt <= yDim/2; yIt += dy)
		{
			for (float zIt = -zDim/2; zIt <= zDim/2; zIt += dz)
			{
				if (particlesGenerated < numParticles)
				{
					double p = noiseMaker.GetValue(xIt/noiseScale, yIt/noiseScale, zIt/noiseScale);
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
						h_pos[index + 2] = zIt + random(dz);// random(200);

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
	*/

#ifdef LOAD_FILE
	std::ifstream in;
	in.open("E:/CudaOutput/final2/data/50600.txt");

	std::vector<std::string> lines;
	std::string line;
	while (std::getline(in, line))
	{
		lines.push_back(line);
	}

	in.close();

	for (int i = 0; i < numParticles; i++)
	{
		double nums[10];
		std::string current = "";
		int c = 0;
		for (int sIt = 0; sIt < lines[i].size(); sIt++)
		{
			if (lines[i][sIt] == ' ')
			{
				nums[c] = std::stod(current);
				current = "";
				c++;
			}
			else
			{
				current += lines[i][sIt];
			}
		}
		int index = i * 3;

		h_pos[index] = nums[0];
		h_pos[index + 1] = nums[1];
		h_pos[index + 2] = nums[2];

		h_vel[index] = nums[3];
		h_vel[index + 1] = nums[4];
		h_vel[index + 2] = nums[5];

		h_acc[index] = 0;
		h_acc[index + 1] = 0;
		h_acc[index + 2] = 0;
	}
#endif

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

	doFrame(d_pos, d_vel, d_acc, d_mass, numParticles);

	//copy vector back to cpu (until opengl-cuda gets implemented)
	cudaMemcpy(h_pos, d_pos, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);
	//std::cout << h_pos[0] << std::endl;

}

int ParticleSystem::getNumParticles()
{
	return numParticles;
}

void ParticleSystem::writeData(const int& frameCounter)
{
	std::ofstream data("E:/CudaOutput/final2/data/" + toString<int>(frameCounter) +".txt");

	cudaMemcpy(h_vel, d_vel, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_acc, d_acc, sizeof(p_type) * 3 * numParticles, cudaMemcpyDeviceToHost);

	data << std::fixed << std::showpoint;
	data << std::setprecision(20);

	double totalMass = 0;
	double xM = 0;
	double yM = 0;
	double zM = 0;
	for (int i = 0; i < numParticles; i++)
	{
		int index = i * 3;

		data << h_pos[index] << " " << h_pos[index + 1] << " " << h_pos[index + 2] << " ";
		data << h_vel[index] << " " << h_vel[index + 1] << " " << h_vel[index + 2] << " ";
		data << h_acc[index] << " " << h_acc[index + 1] << " " << h_acc[index + 2] << " ";
		data << h_mass[i] << " ";

		totalMass += h_mass[i];
		xM += h_pos[index] * h_mass[i];
		yM += h_pos[index + 1] * h_mass[i];
		zM += h_pos[index + 2] * h_mass[i];

		data << "\n";
	}

	int *center = (int*)malloc(sizeof(int) * 3);
	center[0] = xM / totalMass;
	center[1] = yM / totalMass;
	center[2] = zM / totalMass;

	data.close();

	std::ofstream dv("E:/CudaOutput/final2/dv/" + toString<int>(frameCounter) +".txt");
	dv << std::fixed << std::showpoint;
	dv << std::setprecision(20);

	dv << "---------------------------------------------------------\n";
	dv << "CENTER OF MASS " << center[0] << " " << center[1] << " " << center[2] << "\n";
	dv << "---------------------------------------------------------\n";

	//double *distances = (double*)malloc(sizeof(double) * numParticles);
	//double *sVelocity = (double*)malloc(sizeof(double) * numParticles);
	for (int i = 0; i < numParticles; i++)
	{
		int index = i * 3;
		double distance = getMagTwo(h_pos, i * 3, center);
		double sVelocity = sqrt(pow(h_vel[index], 2) + pow(h_vel[index + 1], 2) + pow(h_vel[index + 2], 2));
		dv << i << " " << distance << " " << sVelocity << "\n";
	}

	dv.close();


}
//PRIVATE FUNCTIONS

