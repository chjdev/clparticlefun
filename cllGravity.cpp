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

#include "cllGravity.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "cllError.h"

#define USE_HOST_PTR 1
#define TOSTRING(A) #A

namespace cll {

enum ARGS {
    POS,
    VEL,
    COLOR,
    MOUSE
};

const std::string Gravity::name("cl_gravity");
const std::string Gravity::opts("-cl-nv-verbose -cl-nv-opt-level=3 -cl-unsafe-math-optimizations -cl-fast-relaxed-math");
const std::string Gravity::source = TOSTRING(
__kernel void cl_gravity(__global float4* pos,
                      __global float4* vel,
                      __global float4* color,
                      float4 mouse)
{
    //get our index in the array
    unsigned int i = get_global_id(0);
    float4 p = pos[i];
    float4 v = vel[i];
    float4 c = color[i];

    float4 d = mouse - p;
    float l = fast_length(d)/2.f; //2.f is the maximal distance possible [-1,-1][1,1]
    l = l>0.9f ? 0.9f : l; //prevent particles from escaping ;)
    float4 dn = fast_normalize(d);
    v*=0.99f; //friction
    v += dn * 0.000918f * (1.f-l*l);
    p = (mouse - d)+v;
    p.w = 1.f;

    c.x = (2.f+p.z)*0.5f;

    pos[i] = p;
    vel[i] = v;
    color[i] = c;
}
);

struct Gravity::data_t::Events
{
#if !USE_HOST_PTR
    cl::Event LOAD_V;
#endif
    cl::Event ACQ_GL;
    cl::Event REL_GL;
    cl::Event EXEC;
};

Gravity::data_t::data_t(const std::vector<cl_float4>& pos,
       const std::tr1::function< std::vector<cl_float4>& () >& v_host_injector,
       const std::vector<cl_float4>& col)
    : p_vbo(new cll::VBO<cl_float4>(pos)),
      c_vbo(new cll::VBO<cl_float4>(col)),
      v_host(v_host_injector()),
      m_pos({{0.f, 0.f, -1.f, 1.f}}),
      events(new Events())
{
}

void
Gravity::cl_load(ExecutorBundle& bundle, data_t& data)
{
    glFinish();
    bundle.queue.finish();

    try{
        data.cl_buffers.push_back(cl::BufferGL(bundle.context, CL_MEM_READ_WRITE, data.p_vbo->id(), NULL));
        data.cl_buffers.push_back(cl::BufferGL(bundle.context, CL_MEM_READ_WRITE, data.c_vbo->id(), NULL));

        size_t bytes = data.v_host.size()*sizeof(data.v_host[0]);

#if USE_HOST_PTR
        data.v_cl = cl::Buffer(bundle.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, bytes, (void*)data.v_host.data(), NULL);
#else
        data.v_cl = cl::Buffer(bundle.context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
        bundle.queue.enqueueWriteBuffer(data.v_cl, CL_FALSE, 0,
                                        bytes,
                                        data.v_host.data(),
                                        NULL,
                                        &data.events->LOAD_V);
#endif

        bundle.kernel.setArg(POS, data.cl_buffers[0]); //position vbo
        bundle.kernel.setArg(VEL, data.v_cl); //position vbo
        bundle.kernel.setArg(COLOR, data.cl_buffers[1]); //color vbo
    }
    catch (cl::Error er) {
        std::cout << "ERROR @cl_load: " << er.what() << " " << cll::ErrorString(er.err()) << std::endl;
    }
}

void
Gravity::cl_exec(ExecutorBundle& bundle, data_t& data)
{
    glFinish();

    try{
        bundle.kernel.setArg(MOUSE, data.m_pos); //dt

        bundle.queue.enqueueAcquireGLObjects(&data.cl_buffers, NULL, &data.events->ACQ_GL);
        bundle.queue.finish();

        std::vector<cl::Event> kevents;
#if !USE_HOST_PTR
        kevents.push_back(data.events->LOAD_V);
#endif
        kevents.push_back(data.events->ACQ_GL);
        bundle.queue.enqueueNDRangeKernel(bundle.kernel,
                                          cl::NullRange,
                                          cl::NDRange(data.c_vbo->nelem()),
                                          cl::NullRange,
                                          &kevents,
                                          &data.events->EXEC);
        std::vector<cl::Event> revents;
        revents.push_back(data.events->EXEC);

        bundle.queue.enqueueReleaseGLObjects(&data.cl_buffers, &revents, &data.events->REL_GL);
        bundle.queue.flush();
        data.events->REL_GL.wait();
    }
    catch (cl::Error er) {
        std::cout << "ERROR @cl_exec: " << er.what() << " " << cll::ErrorString(er.err()) << std::endl;
    }

}

}
