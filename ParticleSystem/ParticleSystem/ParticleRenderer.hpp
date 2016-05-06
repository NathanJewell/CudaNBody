#pragma once

#include <cuda_runtime.h>
#include "ParticleSystem.hpp"
#include "GL\glew.h"
#include "GL\freeglut.h"
#include "Defines.hpp"


static class ParticleRenderer
{
public:
	ParticleRenderer();
	~ParticleRenderer();

	void initGL();
	void initRender(const int& newNumParticles);
	void drawFrame();

	void setParticleVector(double* positions);

private:
	GLuint vbo;		//buffer
	GLuint vao;		//vertex array
	GLsizei numParticles;

	static p_type* particles;
};