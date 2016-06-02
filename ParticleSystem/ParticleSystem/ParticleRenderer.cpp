#include "ParticleRenderer.hpp"

int saveCounter = 0;
double* camera;
bool calculate = true;
bool spin = true;
bool save = true;
bool savedata = true;

ParticleRenderer::ParticleRenderer(){}
ParticleRenderer::~ParticleRenderer()
{
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}



void ParticleRenderer::initGL()
{
	/* select clearing (background) color */
	width = 10000;
	height =10000;

	frameCounter = 0;
	rotation = (float*)malloc(sizeof(float) * 4);
	rotation[0] = 0;
	rotation[1] = 0;
	rotation[2] = 0;
	rotation[3] = 0;

	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

	glutInitWindowSize(1024, 1024);

	glutCreateWindow("N-BODY");

	glutDisplayFunc(drawFrame);
	glutKeyboardFunc(keyboardFunc);


	glewInit();



	//cudaGLRegisterBufferObject(vbo);

	glClearColor(0.0, 0.0, 0.0, 0.0);

	/* initialize viewing values */
	glViewport(0, 0, 1024, 1024);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-width/2, width/2, -height/2, height/2, -1000000000, 1000000000);
	//gluPerspective(360.f, 1.f, 1.f, 100000.f);


	glutPostRedisplay();
	//glMatrixMode(GL_PROJECTION);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

	glLoadIdentity();

	//glEnable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	fps = 60;


}

void ParticleRenderer::drawFrame()
{
	/* clear all pixels */
	auto t1 = Clock::now();

	if (calculate)
	{
		sys.doFrameGPU();
	}

	//std::cout << "frame" << std::endl;
	//sys.doFrameCPU();
	if (COORD_TO_PIXEL != 1)
	{
		for (int i = 0; i < numParticles * 3; i++)	//normalize coordinates for display
		{
			screenParticles[i] = particles[i] / COORD_TO_PIXEL;
		}
	}
	else
	{
		screenParticles = particles;
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



	//parameters
	glColor4f(0.0f, 0.0f, 1.0f, .4f);
	glPointSize(2);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glDisable(GL_CULL_FACE);

	//drawing vertex array
	glEnableClientState(GL_VERTEX_ARRAY);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(p_type) * 3 * numParticles, screenParticles, GL_STATIC_DRAW);
	glVertexPointer(3, GL_DOUBLE, sizeof(double) * 3, 0);


	//glPushMatrix();
	//glRotatef(rotation[0], rotation[1], rotation[2], rotation[3]);
	gluLookAt(camera[0], camera[1], camera[2], 0, 0, 0, 0, 1, 0);
	glDrawArrays(GL_POINTS, 0, numParticles);
	//glPopMatrix();


	//glUseProgram(0);
	//glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);


	glDisable(GL_BLEND);
	glDisable(GL_POINT_SMOOTH);



	auto t2 = Clock::now();


	// Make the BYTE array, factor of 3 because it's RBG.
#ifdef SAVE_IMAGES
	if (frameCounter%3 == 0 && save)
	{
		saveCounter++;
		int renderWidth = 1024;
		int renderHeight = 1024;
		GLubyte* pixels = new GLubyte[3 * renderWidth * renderHeight];

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glReadPixels(0, 0, renderWidth, renderHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels);

		std::string filePath = "E:/CudaOutput/final2/frames/" + toString<int>(saveCounter) +".bmp";
	
		// Convert to FreeImage format & save to file
		FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, renderWidth, renderHeight, renderWidth * 3, 24, 0x00FF00, 0x0000FF, 0xFF0000, false);

		FreeImage_Save(FIF_BMP, image, filePath.c_str(), 0);

		// Free resources
		FreeImage_Unload(image);
		delete[] pixels;


	}
#endif
#ifdef SAVE_DATA
	if (savedata && frameCounter % 100 == 0)
	{
		sys.writeData(frameCounter);
		std::cout << "saved stuff" << std::endl;
	}

#endif
	if (spin)
	{
		rotation[0] += .0002;
		camera[0] = sin(rotation[1]) * cos(rotation[0])*rotation[2];
		camera[1] = cos(rotation[1]) * rotation[2];
		camera[2] = sin(rotation[1]) * sin(rotation[0])*rotation[2];
	}
	frameCounter++;
	glutPostRedisplay();

}

void ParticleRenderer::keyboardFunc(unsigned char Key, int x, int y)
{
	/*
	rotation[0] += 1;
	switch (Key)
	{
	case 'w': rotation[1] += 1.0f;
	case 's': rotation[1] += -1.0f;
	case 'a': rotation[2] += 1.0f;
	case 'd': rotation[2] += -1.0f;
	case 'z': rotation[3] += 1.0f;
	case 'x': rotation[3] += -1.0f;
	}
	*/
	if (Key == 'w')
	{
		rotation[1] += -.03;
	}
	else if (Key == 's')
	{
		rotation[1] += .03;
	}
	if (Key == 'a')
	{
		rotation[0] += -.03;
	}
	else if (Key == 'd')
	{
		rotation[0] += .03;
	}
	if (Key == 'r')
	{
		rotation[2] += 100;
	}
	else if (Key == 'f')
	{
		rotation[2] += -100;
	}
	if (Key == 'p')
	{
		spin = !spin;
	}
	if (Key == 'c')
	{
		calculate = !calculate;
	}
	if (Key == 'i')
	{
		save = !save;
	}
	if (Key == 'l')
	{
		savedata = !savedata;
	}
	camera[0] = sin(rotation[1]) * cos(rotation[0])*rotation[2];
	camera[1] = cos(rotation[1]) * rotation[2];
	camera[2] = sin(rotation[1]) * sin(rotation[0])*rotation[2];
};

void ParticleRenderer::initSystem()
{
	camera = (double*)malloc(sizeof(double) * 3);
	//frameCounter = 609600;
	//saveCounter = 31090;
	rotation[0] = 0;
	rotation[1] = 0;
	rotation[2] = 1000000;
	rotation[3] = 0;

	camera[0] = 2000;
	camera[1] = 0;
	camera[2] = 2000;

	sys.allocate(20000);
	sys.initialize();
	numParticles = sys.getNumParticles();

	setParticleVector(sys.getHostParticleVector());

	screenParticles = (p_type*)malloc(sizeof(p_type) * 3 * numParticles);
	for (int i = 0; i < numParticles * 3; i++)	//normalize coordinates for display
	{
		screenParticles[i] = particles[i] / COORD_TO_PIXEL;
	}

	if (!vbo)
	{
		glGenBuffers(1, &vbo);
	}


	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(p_type) * 3 * numParticles, screenParticles, GL_STATIC_DRAW);
	glVertexPointer(3, GL_DOUBLE, sizeof(double) * 3, 0);

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
GLuint ParticleRenderer::cbo;
GLsizei ParticleRenderer::numParticles;
p_type* ParticleRenderer::particles;
p_type* ParticleRenderer::screenParticles;
int ParticleRenderer::width;
int ParticleRenderer::height;
int ParticleRenderer::frameCounter;
GLuint ParticleRenderer::fragmentShader;
GLuint ParticleRenderer::vertexShader;
GLuint ParticleRenderer::program;
float* ParticleRenderer::rotation;

ParticleSystem ParticleRenderer::sys;
float ParticleRenderer::fps;
