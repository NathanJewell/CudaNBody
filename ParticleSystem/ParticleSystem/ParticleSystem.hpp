#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>
#include "Defines.hpp"
#include <math.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math.cuh"
#include "MathUtil.hpp"

#include "noiseutils.h"

class ParticleSystem
{
public:
	ParticleSystem();
	~ParticleSystem();

	void allocate(const unsigned int& numParticles);	//allocated memory on host
	void initialize();									//defines starting positions
	p_type* getHostParticleVector();
	p_type* getDeviceParticleVector();
	int getNumParticles();
	//void copyPositionFromDevice();						//copies calculated vector from device to host
	void doFrameCPU();						//calculates stuff (cpu based)
	void doFrameGPU();					//calculates stuff (gpu based)

private:

	unsigned int numParticles;
	p_type* h_pos;
	p_type* h_vel;
	p_type* h_acc;
	p_type* h_mass;

	p_type* d_pos;
	p_type* d_vel;
	p_type* d_acc;
	p_type* d_mass;

	bool allocated;
};