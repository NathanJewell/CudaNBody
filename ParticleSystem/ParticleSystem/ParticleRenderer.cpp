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
	width = 30000;
	height =30000;
	frameCounter = 0;



	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

	glutInitWindowSize(1024, 1024);

	glutCreateWindow("N-BODY");

	glutDisplayFunc(drawFrame);


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

	vertexShader = createShader(GL_VERTEX_SHADER, loadShaderFile("vertex.glsl"), "vertex");
	fragmentShader = createShader(GL_FRAGMENT_SHADER, loadShaderFile("fragment.glsl"), "fragment");

	program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
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
	glPointSize(10);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glDisable(GL_CULL_FACE);

	//drawing vertex array
	glEnableClientState(GL_VERTEX_ARRAY);


	glUseProgram(program);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(p_type) * 3 * numParticles, screenParticles, GL_STATIC_DRAW);
	glVertexPointer(3, GL_DOUBLE, sizeof(p_type) * 3, 0);



	glDrawArrays(GL_POINTS, 0, numParticles);

	glUseProgram(0);
	//glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);


	glDisable(GL_BLEND);
	glDisable(GL_POINT_SMOOTH);



	auto t2 = Clock::now();


	// Make the BYTE array, factor of 3 because it's RBG.
#ifdef SAVE_IMAGES
	BYTE* pixels = new BYTE[3 * width * height];

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

	std::string filePath = "C:/CudaOutput/" + toString<int>(frameCounter) +".bmp";
	frameCounter++;
	// Convert to FreeImage format & save to file
	FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
	FreeImage_Save(FIF_BMP, image, filePath.c_str(), 0);

	// Free resources
	FreeImage_Unload(image);
	delete[] pixels;
#endif

	glutPostRedisplay();

}

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
p_type* ParticleRenderer::screenParticles;
int ParticleRenderer::width;
int ParticleRenderer::height;
int ParticleRenderer::frameCounter;
GLuint ParticleRenderer::fragmentShader;
GLuint ParticleRenderer::vertexShader;
GLuint ParticleRenderer::program;

ParticleSystem ParticleRenderer::sys;
float ParticleRenderer::fps;
