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
#ifndef CLLGRAVITY_H
#define CLLGRAVITY_H

#include <tr1/memory>
#include <tr1/functional>
#include <vector>
#include "cllExecutor.h"
#include "cllVBO.h"

namespace cll {

struct Gravity {
    static const std::string source;
    static const std::string name;
    static const std::string opts;

    struct data_t {
        struct inject_point {};

        data_t(const std::vector<cl_float4>& pos,
               const std::tr1::function< std::vector<cl_float4>& () >& v_host_injector,
               const std::vector<cl_float4>& col);
        const std::tr1::shared_ptr< cll::VBO<cl_float4> > p_vbo;
        const std::tr1::shared_ptr< cll::VBO<cl_float4> > c_vbo;
        const std::vector<cl_float4>& v_host;
        cl_float4 m_pos;
        std::vector<cl::Memory> cl_buffers;
        cl::Buffer v_cl;

    private:
        friend struct Gravity;
        struct Events;
        const std::tr1::shared_ptr< Events > events;
    };

    static void cl_load(ExecutorBundle&, data_t& data);

    static void cl_exec(ExecutorBundle&, data_t& data);
};

}

#endif
