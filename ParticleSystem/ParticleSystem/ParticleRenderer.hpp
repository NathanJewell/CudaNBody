#pragma once

#include "ParticleSystem.hpp"
#include "inc/GL/freeglut.h"
#include "inc/GL/glew.h"


class ParticleRenderer
{
public:
	ParticleRenderer();
	~ParticleRenderer();

	void initGL();
	static void drawFrame();

	void setParticleVector(double* positions);
	double* getParticleVector();

private:
	GLuint vbo;
};