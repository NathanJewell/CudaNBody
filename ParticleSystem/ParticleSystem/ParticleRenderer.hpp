#pragma once

#include <cuda_runtime.h>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "ParticleSystem.hpp"
#include "Defines.hpp"
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;



class ParticleRenderer
{
public:
	ParticleRenderer();
	~ParticleRenderer();

	void initGL();
	void initRender(const int& newNumParticles);
	void initSystem();	//allocate memory and initialize particle positionss
	void begin();
	static void drawFrame();

	void setParticleVector(p_type* positions);

private:
	static GLuint vbo;		//buffer

	static GLsizei numParticles;

	static p_type* particles;

	static ParticleSystem sys;

	static float fps;
};