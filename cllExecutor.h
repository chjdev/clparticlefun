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

#pragma once
#ifndef CLLEXECUTOR_H
#define CLLEXECUTOR_H

#include <iostream>
#include <string>
#include <tr1/functional>
#include <tr1/memory>

#include <GL/glew.h> //needed for compile
#include <GL/glx.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <boost/utility.hpp>

#include "cllError.h"

namespace cll {

struct ExecutorBundle
{
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    unsigned int deviceUsed;
};

template<typename T>
class Executor : boost::noncopyable
{
public:
    typedef std::tr1::function<void (ExecutorBundle&, typename T::data_t&)> strategy_func_t;

    Executor();

    void exec(typename T::data_t& data) { m_exec_func(bundle, data); }
    void exec(std::tr1::shared_ptr<typename T::data_t> data) { m_exec_func(bundle, *data); }

    void load(typename T::data_t& data) { m_load_func(bundle, data); }
    void load(std::tr1::shared_ptr<typename T::data_t> data) { m_load_func(bundle, *data); }

    void set_exec_func(const strategy_func_t& e) { m_exec_func = e; }
    void set_load_func(const strategy_func_t& l) { m_load_func = l; }

private:
    strategy_func_t m_load_func;
    strategy_func_t m_exec_func;

    ExecutorBundle bundle;
};


template<typename T>
Executor<T>::Executor()
    : m_load_func(T::cl_load), m_exec_func(T::cl_exec)
{
    try{
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::cout << "platforms.size(): " <<  platforms.size() << std::endl;

        //atm first device of first platform is used... yeah
        bundle.deviceUsed = 0;
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        std::cout << "devices.size(): " << devices.size() << std::endl;
        int t = devices.front().getInfo<CL_DEVICE_TYPE>();
        std::cout << "type: device: " << t << " CL_DEVICE_TYPE_GPU: " << CL_DEVICE_TYPE_GPU << std::endl;

        cl_context_properties props[] =
        {
            CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
            CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(),
            0
        };

//        context = cl::Context(CL_DEVICE_TYPE_GPU), props);
        bundle.context = cl::Context(devices, props);
        bundle.queue = cl::CommandQueue(bundle.context, devices[bundle.deviceUsed], 0, 0);
        cl::Program::Sources source(1, std::make_pair(T::source.c_str(), T::source.size()));
        bundle.program = cl::Program(bundle.context, source);
        bundle.program.build(devices, T::opts.c_str());

        bundle.kernel = cl::Kernel(bundle.program, T::name.c_str(), 0);

        std::cout << "Build Status: " << bundle.program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
        std::cout << "Build Options:\t" << bundle.program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
        std::cout << "Build Log:\t " << bundle.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    }
    catch (cl::Error er) {
        std::cout << "ERROR @Executor: " << er.what() << " " << ErrorString(er.err()) << std::endl;
    }
}

}

#endif

