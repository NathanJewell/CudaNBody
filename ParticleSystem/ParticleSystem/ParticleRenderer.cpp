#include "ParticleRenderer.hpp"

ParticleRenderer::ParticleRenderer(){}
ParticleRenderer::~ParticleRenderer(){}

void ParticleRenderer::initGL()
{
	/* select clearing (background) color */

	glClearColor(0.0, 0.0, 0.0, 0.0);

	/* initialize viewing values */

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
}

void ParticleRenderer::drawFrame()
{
	/* clear all pixels */

	glClear(GL_COLOR_BUFFER_BIT);

	/* draw white polygon (rectangle) with corners at

	* (0.25, 0.25, 0.0) and (0.75, 0.75, 0.0)

	*/

	glColor3f(1.0, 1.0, 1.0);

	glBegin(GL_POLYGON);

	glVertex3f(0.25, 0.25, 0.0);

	glVertex3f(0.75, 0.25, 0.0);

	glVertex3f(0.75, 0.75, 0.0);

	glVertex3f(0.25, 0.75, 0.0);

	glEnd();

	/* don't wait!

	* start processing buffered OpenGL routines

	*/

	glFlush();

}
