

void main()
{
	vec4 vert = gl_Vertex;// vec4(in_position, 1.0);//w is 1.0, also notice cast to a vec4
	vert.w = 1.0;

	gl_Position = vert;
}