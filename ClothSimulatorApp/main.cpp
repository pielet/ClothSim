// OpenGL
#include <GL\glew.h>
#include <GL\freeglut.h>

#include <iostream>
#include <sstream>
#include <thread>

#include "Utility.h"
#include "Render/Camera.h"
#include "Render/Shader.h"
#include "Render/ClothRenderer.h"
#include "../ClothSim/ClothSim.h"

// parameters
int g_window_width = 800;
int g_window_height = 600;

Timer g_timer;
int frame_count = 0;

Shader* g_shader;
Camera* g_camera;
cloth::ClothRenderer* g_renderer;
cloth::ClothSim* g_cloth;

bool g_exit = false;
bool g_stop = true;
bool g_step = false;

void computePFS()
{
	float dt = g_timer.elapsedSeconds();

	std::stringstream oss;
	oss << "StrandSim: " << frame_count / dt << " fps";
	glutSetWindowTitle(oss.str().c_str());

	frame_count = 0;
	g_timer.start();
}

void reshape(int w, int h)
{
	g_window_width = w;
	g_window_height = h;

	g_camera->setViewPort(w, h);
	glViewport(0, 0, w, h);

	glutPostRedisplay();
}

void mouseClick(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		switch (button)
		{
		case GLUT_LEFT_BUTTON:
			g_camera->beginMotion(Camera::Motion::ROTATE, x, y);
			break;
		case GLUT_MIDDLE_BUTTON:
			g_camera->beginMotion(Camera::Motion::TRANSLATE, x, y);
			break;
		case GLUT_RIGHT_BUTTON:
			g_camera->beginMotion(Camera::Motion::SCALE, x, y);
			break;
		default:
			break;
		}
	}
}

void mouseMotion(int x, int y)
{
	g_camera->move(x, y);
}

void wheelScroll(int wheel, int direction, int x, int y)
{
	g_camera->scroll(direction);
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'q': case 'Q':
		g_exit = true;
		break;
	case ' ':
		g_stop = !g_stop;
		break;
	case 's': case 'S':
		g_step = true;
		break;
	}
}

void display()
{
	if (g_exit)
	{
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
		glutLeaveMainLoop();
	}

	glClearColor(1.f, 1.f, 1.f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	g_shader->use();
	Mat4f MVP = g_camera->getPerspectiveMatrix() * g_camera->getViewMatrix();
	g_shader->setMat4f("MVP", MVP);

	g_renderer->draw(g_shader);
	glutSwapBuffers();

	if (g_timer.elapsedSeconds() > 1.0)
		computePFS();
}

void simulation()
{
	while (!g_exit)
	{
		bool update = false;

		try
		{
			if (g_stop)
			{
				if (g_step)
				{
					update = g_cloth->step();
					g_step = false;
				}
				else std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			else update = g_cloth->step();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << std::endl;
			g_exit = true;
			break;
		}

		if (update) g_renderer->ackGeometryChange();
		++frame_count;
	}
}

int main(int argc, char** argv)
{
	// glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(g_window_width, g_window_height);
	glutCreateWindow("ClothSim");
	g_timer.start();

	// glew
	glewInit();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glViewport(0, 0, g_window_width, g_window_height);

	// glut callback
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMotion);
	glutMouseWheelFunc(wheelScroll);

	// scene
	g_shader = new Shader("Render/shader.vs", "Render/shader.fs");
	g_camera = new Camera(g_window_width, g_window_height);
	g_cloth = new cloth::ClothSim();
	try
	{
		g_cloth->initialize("../config/scene.json");
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what();
		return -1;
	}
	g_renderer = new cloth::ClothRenderer(g_cloth);

	std::thread sim(simulation);
	glutMainLoop();
	sim.join();

	delete g_shader;
	delete g_camera;
	delete g_cloth;

	cudaDeviceReset();

	std::cout << "\nEXIT.";

	return 0;
}
