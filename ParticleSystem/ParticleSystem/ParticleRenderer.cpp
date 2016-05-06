#include "ParticleRenderer.hpp"

ParticleRenderer::ParticleRenderer(){}
ParticleRenderer::~ParticleRenderer(){}

void ParticleRenderer::initGL()
{
	/* select clearing (background) color */
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(particles), particles, GL_STATIC_DRAW);

	cudaGLRegisterBufferObject(vbo);

	glClearColor(0.0, 0.0, 0.0, 0.0);

	/* initialize viewing values */
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);

	glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

	glLoadIdentity();

	glEnable(GL_DEPTH_TEST);
	glClearColor(1.0f, 1.0f, 1.0f, 1.5f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


}

void ParticleRenderer::drawFrame()
{
	/* clear all pixels */

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//parameters
	glColor4f(0, 1, 0, 0.5f);
	glPointSize(1);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//drawing vertex array
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
	glVertexPointer(2, GL_FLOAT, 0, NULL);
	glDrawArrays(GL_POINTS, 0, numParticles);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	/* draw white polygon (rectangle) with corners at

	* (0.25, 0.25, 0.0) and (0.75, 0.75, 0.0)

	*/
	glBindVertexArray(vao);
	glDrawArrays(GL_POINT, 0, numParticles);

	glColor3f(1.0, 1.0, 1.0);

	/* don't wait!

	* start processing buffered OpenGL routines

	*/

	glFlush();

}

void ParticleRenderer::setParticleVector(double* positions)
{

}


