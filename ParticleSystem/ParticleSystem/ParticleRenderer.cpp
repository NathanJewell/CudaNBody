#include "ParticleRenderer.hpp"

ParticleRenderer::ParticleRenderer(){}
ParticleRenderer::~ParticleRenderer(){}



void ParticleRenderer::initGL()
{
	/* select clearing (background) color */


	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

	glutInitWindowSize(1024, 1024);

	glutCreateWindow("N-BODY");

	glutDisplayFunc(drawFrame);


	glewInit();


	glGenBuffers(1, &vbo);
	
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(p_type) * 3 * numParticles, particles, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//cudaGLRegisterBufferObject(vbo);

	glClearColor(0.0, 0.0, 0.0, 0.0);

	/* initialize viewing values */
	glViewport(512, 0, 128, 128);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-2240, 2240, -2240, 2240, -1000000000, 1000000000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
	//glMatrixMode(GL_PROJECTION);

	//glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

	glLoadIdentity();

	//glEnable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	fps = 60;
}

void ParticleRenderer::drawFrame()
{
	/* clear all pixels */
	auto t1 = Clock::now();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//parameters
	glColor4f(1.0f, 0.0f, 1.0f, .4f);
	glPointSize(2);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_CULL_FACE);

	//drawing vertex array
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(p_type) * 3 * numParticles, particles, GL_STATIC_DRAW);
	glVertexPointer(3, GL_FLOAT, sizeof(p_type) * 3, 0);
	glDrawArrays(GL_POINTS, 0, numParticles);

	//glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);


	glDisable(GL_BLEND);
	glDisable(GL_POINT_SMOOTH);

	sys.doFrameGPU();
	//sys.doFrameCPU();
	std::cout << "big mama" << std::endl;
	auto t2 = Clock::now();
	//float frametime = pow(10, 9) / 60;
	//float time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
	//if (time > 0)
	//{
//		Sleep(abs(time / 1000));
	//}


	glutPostRedisplay();


	//glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	/* draw white polygon (rectangle) with corners at

	* (0.25, 0.25, 0.0) and (0.75, 0.75, 0.0)

	*/



	/* don't wait!

	* start processing buffered OpenGL routines

	*/

}

void ParticleRenderer::initSystem()
{
	sys.allocate(5000);
	sys.initialize();
	numParticles = sys.getNumParticles();

	setParticleVector(sys.getHostParticleVector());


	if (!vbo)
	{
		glGenBuffers(1, &vbo);
	}

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(p_type) * 3 * numParticles, particles, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void ParticleRenderer::begin()
{
	glutMainLoop();
}

void ParticleRenderer::setParticleVector(p_type* positions)
{
	particles = positions;
}

GLuint ParticleRenderer::vbo;
GLsizei ParticleRenderer::numParticles;
p_type* ParticleRenderer::particles;

ParticleSystem ParticleRenderer::sys;
float ParticleRenderer::fps;
