#include <windows.h>

#include <GL/glew.h>

#include <GL/freeglut.h>

#include <GL/gl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "InitShader.h"
#include "LoadMesh.h"
#include "LoadTexture.h"
#include "imgui_impl_glut.h"

#include "cuda_gl_interop.h"

#include "cutil_math.h"

const int width = 1280;
const int height = 720;
unsigned int frame = 0;

GLuint pbo = 1;
GLuint textureID = 1;
struct cudaGraphicsResource* resource;

extern "C" void launch_kernel(uchar4*, float3*, unsigned int, unsigned int, unsigned int);


void createPBO(GLuint *pbo)
{
	if (pbo)
	{
		int num_texels = width * height;
		int num_values = num_texels * 4;

		int size_tex_data = sizeof(GLubyte) * num_values;

		glGenBuffers(1, pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);

		cudaGraphicsGLRegisterBuffer(&resource, *pbo, cudaGraphicsMapFlagsWriteDiscard);
	}
}
void createTexture(GLuint *textureID, unsigned int size_x, unsigned int size_y)
{
	glEnable(GL_TEXTURE_2D);

	glGenTextures(1, textureID);
	glBindTexture(GL_TEXTURE_2D, *textureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}
uchar4* dptr;
float3* accu;
void runCuda()
{
	size_t num_bytes;

	cudaGraphicsMapResources(1, &resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, resource);

	launch_kernel(dptr, accu, width, height, frame);

	cudaGraphicsUnmapResources(1, &resource, 0);
}


// glut display callback function.
// This function gets called every time the scene gets redisplayed 
void display()
{
	
}



// glut idle callback.
//This function gets called between frames
void idle()
{
	frame++;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	runCuda();

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();

	glutSwapBuffers();
}



// Display info about the OpenGL implementation provided by the graphics driver.
// Your version should be > 4.0 for CGT 521 
void printGlInfo()
{
	std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
}

void initCuda()
{
	cudaMalloc(&accu, width * height * sizeof(float3));
	createPBO(&pbo);
	createTexture(&textureID, width, height);
	runCuda();
}

void initOpenGl()
{
	frame = 0;
	//Initialize glew so that new OpenGL function names can be used
	glewInit();

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

}

// glut callbacks need to send keyboard and mouse events to imgui
void keyboard(unsigned char key, int x, int y)
{
	ImGui_ImplGlut_KeyCallback(key);
	std::cout << "key : " << key << ", x: " << x << ", y: " << y << std::endl;

	switch (key)
	{
	default:
		break;
	}
}

void keyboard_up(unsigned char key, int x, int y)
{
	ImGui_ImplGlut_KeyUpCallback(key);
}

void special_up(int key, int x, int y)
{
	ImGui_ImplGlut_SpecialUpCallback(key);
}

void passive(int x, int y)
{
	ImGui_ImplGlut_PassiveMouseMotionCallback(x, y);
}

void special(int key, int x, int y)
{
	ImGui_ImplGlut_SpecialCallback(key);
}

void motion(int x, int y)
{
	ImGui_ImplGlut_MouseMotionCallback(x, y);
}

void mouse(int button, int state, int x, int y)
{
	ImGui_ImplGlut_MouseButtonCallback(button, state);
}


int main(int argc, char **argv)
{
	//Configure initial window state using freeglut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(5, 5);
	glutInitWindowSize(width, height);
	int win = glutCreateWindow("Path Tracer");

	printGlInfo();

	//Register callback functions with glut. 
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutKeyboardUpFunc(keyboard_up);
	glutSpecialUpFunc(special_up);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(motion);

	glutIdleFunc(idle);

	
	initOpenGl();
	initCuda();
	ImGui_ImplGlut_Init(); // initialize the imgui system

	//Enter the glut event loop.
	glutMainLoop();
	cudaThreadExit();
	glutDestroyWindow(win);

	cudaFree(dptr);
	cudaFree(accu);

	return 0;
}


