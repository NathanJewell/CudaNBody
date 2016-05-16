#version 330 core

in vec4 color;

void main(void)
{
	vec3 light = vec3(.54901, .843137, .90196);	//bluey
	vec3 dense = vec3(.96078, .57647, .258823);	//orangeish
	
	vec3 product = (light * cos(color.a*3.141592)) + (dense * sin(color.a*3.141592));
	vec4 newColor = vec4(1.0,0.0,0.0, 1.0);
	gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}