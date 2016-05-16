#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <GL\freeglut.h>
#include <GL\GL.h>
#include <GL\glew.h>

inline std::string loadShaderFile(const std::string& fileName)
{
	std::string shaderCode;
	std::ifstream file(fileName, std::ios::in);

	if (!file.good())
	{
		std::cout << "Cannot read file " << fileName << "\n";
		std::terminate();
	}

	file.seekg(0, std::ios::end);
	shaderCode.resize((unsigned int)file.tellg());
	file.seekg(0, std::ios::beg);
	file.read(&shaderCode[0], shaderCode.size());
	file.close();

	return shaderCode;
}


inline GLuint createShader(GLenum shaderType, const std::string source, const std::string& shaderName)
{
	int compileResult = 0;
	GLuint shader = glCreateShader(shaderType); //create empty shader object of type shaderType
	const char *shaderCodePtr = source.c_str();
	const int shaderCodeSize = source.size();

	glShaderSource(shader, 1, &shaderCodePtr, &shaderCodeSize); //sets shader code (shadeObject, #charArrays, referenceToArray, #filledArraylocations)
	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compileResult);

	//checking for compile errors
	if (compileResult == GL_FALSE)
	{
		int infoLogLength = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
		std::vector<char> shaderLog(infoLogLength);

		glGetShaderInfoLog(shader, infoLogLength, NULL, &shaderLog[0]);
		std::cout << "ERROR compiling shader: " << shaderName << "\n" << &shaderLog[0] << "\n";
		return 0;
	}

	return shader;

}