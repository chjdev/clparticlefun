/*
Copyright (c) 2011, Christian Junker
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// borrows quite a bit from http://enja.org/2010/08/27/adventures-in-opencl-part-2-particles-with-opengl/ :)

#include <iostream>
#include <sstream>
#include <vector>
#include <assert.h>
#include <tr1/memory>
#include <cmath>

#include "cllGravity.h"

#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glu.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cstring>
#include <ctime>

static const unsigned FPS = 30;
static const clock_t STEP = CLOCKS_PER_SEC/FPS;
static const float CONVERT = 1000.f/(float)CLOCKS_PER_SEC;
static const unsigned int NUM_PARTICLES = 100000;
static const float DT = 1.f/(float)NUM_PARTICLES;
static const float PI2 = 2.f*3.14f;
static const float INIT_VEL = 0.f;
static const GLfloat POINT_SIZE = 5.f;
static GLsizei windowWidth = 768;
static GLsizei windowHeight = 768;
static GLfloat translateZ = -1.f;
static int glutWindowHandle = 0;

static void init_gl(int, char**);
static void appRender(void);
static void appDestroy(void);
static void appKeyboard(const unsigned char, const int, const int);
static void timerCB(const int);
static void appMouse(int, int, int, int);

static inline float rand_float(const float mn, const float mx)
{
    float r = random() / (float) RAND_MAX;
    return mn + (mx-mn)*r;
}

typedef std::tr1::shared_ptr< cll::Executor<cll::Gravity> > exec_ptr_t;
static exec_ptr_t exec_gravity;

typedef std::tr1::shared_ptr< cll::Gravity::data_t > data_ptr_t;
static data_ptr_t gravity_data;

//dependency injection of v_host vector... for efficient sharing
//yeah kinda stupid, but i wanted to try it ;)
std::vector<cl_float4>& inj_v_host()
{
    static std::vector<cl_float4> v_host(NUM_PARTICLES);
    return v_host;
}

struct inj_ident_t {};

int main(int argc, char** argv)
{
    init_gl(argc, argv);

    std::vector<cl_float4> pos(NUM_PARTICLES);
    std::vector<cl_float4>& vel = inj_v_host();
    float t;
    std::vector<cl_float4>::iterator pit, vit;
    for(pit = pos.begin(), vit = vel.begin(), t=0.f;
        pit<pos.end();
        ++pit, ++vit, t+=DT)
    {
        //distribute the particles in a random circle around z axis
        const float rad = rand_float(.1f, .95f);
        float dxy = std::cos(PI2 * t);
        //c++0x has problems with the cl_float4 union somehow
        vit->s[0] = dxy*INIT_VEL;
        pit->s[0] = rad*dxy;

        dxy = std::sin(PI2 * t);
        vit->s[1] = dxy*INIT_VEL;
        pit->s[1] = rad*dxy;

        vit->s[2] = INIT_VEL;
        pit->s[2] = rad-0.5f;
        vit->s[3] = 1.f;
        pit->s[3] = 1.f;
    }
    cl_float4 colorv = {{1.0f, 0.0f, 0.0f, 1.0f}};
    std::vector<cl_float4> color(NUM_PARTICLES, colorv);

    gravity_data = data_ptr_t(new data_ptr_t::element_type(pos, inj_v_host, color));
    exec_gravity = exec_ptr_t(new exec_ptr_t::element_type());
    exec_gravity->load(gravity_data);

    glutMainLoop();
}

static void init_gl(int argc, char** argv)
{

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - windowWidth/2,
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - windowHeight/2);


    std::ostringstream ss;
    ss << "Gravity with OpenCL/OpenGL, using " << NUM_PARTICLES << " particles" << std::ends;
    glutWindowHandle = glutCreateWindow(ss.str().c_str());

    glutDisplayFunc(appRender); //main rendering function
    glutTimerFunc(0, timerCB, 0); //determin a minimum time between frames
    glutKeyboardFunc(appKeyboard);
    glutMouseFunc(appMouse);

    glewInit();

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, windowWidth, windowHeight);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0, (GLfloat)windowWidth / (GLfloat)windowHeight, 0.1, 1000.0);

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translateZ);
}

static void appRender(void)
{
    clock_t endwait = clock() + STEP;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    exec_gravity->exec(gravity_data);

    //render the particles from VBOs
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(POINT_SIZE);

    //printf("color buffer\n");
    glBindBuffer(GL_ARRAY_BUFFER, gravity_data->c_vbo->id());
    glColorPointer(4, GL_FLOAT, 0, 0);

    //printf("vertex buffer\n");
    glBindBuffer(GL_ARRAY_BUFFER, gravity_data->p_vbo->id());
    glVertexPointer(4, GL_FLOAT, 0, 0);

    //printf("enable client state\n");
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    //printf("draw arrays\n");
    glDrawArrays(GL_POINTS, 0, gravity_data->p_vbo->nelem());

    //printf("disable stuff\n");
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    clock_t done = clock();
    if(done < endwait)
        glutTimerFunc((float)(endwait-done)*CONVERT, timerCB, 0);
    else
    {
        std::cout << "perf!!!" << std::endl;
        glutTimerFunc(0, timerCB, 0);
    }
}

static void appDestroy(void)
{
    //this makes sure we properly cleanup our OpenCL context
    if(glutWindowHandle)
        glutDestroyWindow(glutWindowHandle);

    std::cout << "about to exit!" << std::endl;
    exit(EXIT_SUCCESS);
}

static void appKeyboard(const unsigned char key, const int, const int)
{
    //this way we can exit the program cleanly
    switch(key)
    {
        case '\033': // escape quits
        case '\015': // Enter quits
        case 'Q': // Q quits
        case 'q': // q (or escape) quits
            // Cleanup up and quit
            appDestroy();
            break;
    }
}

static void timerCB(const int)
{
    glutPostRedisplay();
}

void appMouse(int button, int, int x, int y)
{
    gravity_data->m_pos.s[0] = 2.f*((GLfloat)x/(GLfloat)windowWidth - 0.5f);
    gravity_data->m_pos.s[1] = -2.f*((GLfloat)y/(GLfloat)windowHeight - 0.5f);
    if(button == GLUT_MIDDLE_BUTTON)
        gravity_data->m_pos.s[2] = -gravity_data->m_pos.s[2];
    std::cout << "X: " << gravity_data->m_pos.s[0] << " Y: " << gravity_data->m_pos.s[1] << " Z: " << gravity_data->m_pos.s[2] << std::endl;
}

