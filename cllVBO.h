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
#ifndef CLLVBO_H
#define CLLVBO_H

#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <boost/utility.hpp>

namespace cll {

template<typename T, GLenum TARGET = GL_ARRAY_BUFFER, GLenum USAGE = GL_DYNAMIC_DRAW>
class VBO : boost::noncopyable
{
public:
    explicit VBO(const std::vector<T>&);
    virtual ~VBO();

    GLsizei nelem() const { return m_nelem; }
    GLuint id() const { return m_id; }

    static GLenum target() { return TARGET; }
    static GLenum usage() { return USAGE; }

private:
    GLsizei m_nelem;
    GLuint m_id;
};


template<typename T, GLenum TARGET, GLenum USAGE>
VBO<T,TARGET,USAGE>::VBO(const std::vector<T>& elems)
    : m_nelem(elems.size()), m_id(0)
{
    glGenBuffers(1, &m_id); // create a vbo
    glBindBuffer(TARGET, m_id); // activate vbo id to use
    glBufferData(TARGET, sizeof(T)*m_nelem, elems.data(), USAGE);

    //TODO error checking

    //TODO restore state
    glBindBuffer(TARGET, 0);
}



template<typename T, GLenum TARGET, GLenum USAGE>
VBO<T,TARGET,USAGE>::~VBO()
{
    glDeleteBuffers(1, &m_id);
}

}

#endif
