#include "ParticleRenderer.hpp"

ParticleRenderer::ParticleRenderer(){}
ParticleRenderer::~ParticleRenderer()
{
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}



void ParticleRenderer::initGL()
{
	/* select clearing (background) color */
	width = 3000;
	height =3000;

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
	glViewport(512, 0, 128, 128);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-width/2, width/2, -height/2, height/2, -1000000000, 1000000000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
	//glMatrixMode(GL_PROJECTION);

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

	sys.doFrameGPU();
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
	glColor4f(1.0f, 0.0f, 0.0f, .4f);
	glPointSize(2);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glDisable(GL_CULL_FACE);

	//drawing vertex array
	glEnableClientState(GL_VERTEX_ARRAY);



	for (int i = 0; i < numParticles * 3; i++)	//normalize coordinates for display
	{
		screenParticles[i] = particles[i]/ COORD_TO_PIXEL;
	}


	//glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(p_type)*3, screenParticles);
	//glEnableVertexAttribArray(0);

	//glGetUniformLocation(program, "in_color");
	//glGetUniformLocation(program, "in_position");


	//glUseProgram(program);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();



	glPushMatrix();
	glRotatef(rotation[0], rotation[1], rotation[2], rotation[3]);
	glDrawArrays(GL_POINTS, 0, numParticles);
	glPopMatrix();


	//glUseProgram(0);
	//glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);


	glDisable(GL_BLEND);
	glDisable(GL_POINT_SMOOTH);



	auto t2 = Clock::now();


	// Make the BYTE array, factor of 3 because it's RBG.
#ifdef SAVE_IMAGES
	int renderWidth = 1024;
	int renderHeight = 1024;
	GLubyte* pixels = new GLubyte[3 * renderWidth * renderHeight];

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glReadPixels(0, 0, renderWidth, renderHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels);

	std::string filePath = "C:/CudaOutput/" + toString<int>(frameCounter) +".bmp";
	frameCounter++;
	// Convert to FreeImage format & save to file
	FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, renderWidth, renderHeight, renderWidth*3, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
	FreeImage_Save(FIF_BMP, image, filePath.c_str(), 0);

	// Free resources
	FreeImage_Unload(image);
	delete[] pixels;
#endif

	glutPostRedisplay();

}

void ParticleRenderer::keyboardFunc(unsigned char Key, int x, int y)
{
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
};

void ParticleRenderer::initSystem()
{
	sys.allocate(5000);
	sys.initialize();
	numParticles = sys.getNumParticles();

	setParticleVector(sys.getHostParticleVector());

	screenParticles = (p_type*)malloc(sizeof(p_type) * 3 * numParticles);
	for (int i = 0; i < numParticles * 3; i++)	//normalize coordinates for display
	{
		screenParticles[i] = particles[i];// / COORD_TO_PIXEL;
	}

	if (!vbo)
	{
		glGenBuffers(1, &vbo);
	}


	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(p_type) * 3 * numParticles, screenParticles, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(double) * 3, &screenParticles[0]);
	glEnableVertexAttribArray(0);


	glGenBuffers(1, &cbo);
	glBindBuffer(GL_ARRAY_BUFFER, cbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*4*numParticles, &colors[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);



	vertexShader = createShader(GL_VERTEX_SHADER, loadShaderFile("vertex.glsl"), "vertex");
	fragmentShader = createShader(GL_FRAGMENT_SHADER, loadShaderFile("fragment.glsl"), "fragment");



	program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glBindAttribLocation(program, 0, "in_position");

	glLinkProgram(program);

	int linkResult = 0;

	glGetProgramiv(program, GL_LINK_STATUS, &linkResult);

	//check for link errors
	if (linkResult == GL_FALSE)
	{
		int infoLogLength = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
		std::vector<char> programLog(infoLogLength);
		glGetProgramInfoLog(program, infoLogLength, NULL, &programLog[0]);
		std::cout << "Shader Loader: LINK ERROR\n" << &programLog[0] << "\n";
	}
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
