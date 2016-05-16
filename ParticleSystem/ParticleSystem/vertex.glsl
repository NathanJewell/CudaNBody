#version 330 core                           //lower your version if GLSL 4.5 is not supported by your GPU
layout(location = 0) in vec3 in_position;  //set the frist input on location (index) 0 ; in_position is our attribute 


out vec4 color;

void main()
{
	color = vec4(0, 1, 0, .3);
	gl_Position = vec4(in_position, 1.0);//w is 1.0, also notice cast to a vec4
}