#pragma once

#include <cuda_runtime.h>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "ParticleSystem.hpp"
#include "Defines.hpp"
#include <chrono>
#include "inc/FreeImage.h"
#include "ShaderLoader.hpp"


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
	static void keyboardFunc(unsigned char Key, int x, int y);

	void setParticleVector(p_type* positions);

private:
	static GLuint vbo, cbo;		//buffers

	static GLsizei numParticles;

	static p_type* particles;
	static p_type* screenParticles;

	static ParticleSystem sys;

	static float fps;

	static int width, height;

	static int frameCounter;

	static GLuint vertexShader, fragmentShader, program;

	static float* rotation;
};