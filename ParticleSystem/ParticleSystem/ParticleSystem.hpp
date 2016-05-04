#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>
#include "inc/GL/glew.h"
#include "inc/GL/freeglut.h"


class ParticleSystem
{
public:
	ParticleSystem();
	~ParticleSystem();

	void allocate(const unsigned int& numParticles);	//allocated memory on host
	void initialize();									//defines starting positions
	double* getParticleVector();
	//void copyPositionFromDevice();						//copies calculated vector from device to host
	void doFrameCPU();						//calculates stuff (cpu based)
	void doFrameGPU();					//calculates stuff (gpu based)

private:
	float fInvSqrt(const float& in);

	unsigned int numParticles;
	double* h_pos;
	double* h_vel;
	double* h_acc;
	double* h_mass;

	double* d_pos;
	double* d_vel;
	double* d_acc;
	double* d_mass;

	bool allocated;
};