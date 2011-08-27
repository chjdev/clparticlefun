CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(cllparticlefun)

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
message("CMake module path: ${CMAKE_MODULE_PATH}\n")

FIND_PACKAGE (OPENCL)
FIND_PACKAGE(OpenGL)
FIND_PACKAGE(GLUT)
FIND_PACKAGE(GLEW)

INCLUDE_DIRECTORIES(
    ${GLUT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}
)

message("include directories: \n")
get_directory_property(includes INCLUDE_DIRECTORIES)
message("${includes}\n")

ADD_DEFINITIONS(-O0 -g -std=c++0x -Wall -Wextra -Werror -Wfatal-errors -pedantic)
#ADD_DEFINITIONS(-O3 -DNDEBUG -std=c++0x -Wall -Wextra -Werror -Wfatal-errors -pedantic)

SET(LIBSRCS
    cllExecutor.h
    cllVBO.h
    cllError.h
    cllError.cpp
)
ADD_LIBRARY(cll ${LIBSRCS})
TARGET_LINK_LIBRARIES(cll
   ${OPENGL_LIBRARIES}
   ${GLEW_LIBRARY}
   ${OPENCL_LIBRARIES}
)


SET(MAINSRCS
    cllGravity.h
    cllGravity.cpp
    main.cpp
)
ADD_EXECUTABLE(clbin ${MAINSRCS})
TARGET_LINK_LIBRARIES(clbin
   cll
   ${GLUT_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${GLEW_LIBRARY}
   ${OPENCL_LIBRARIES}
)
