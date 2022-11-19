cmake_minimum_required(VERSION 3.21)
project(Raytracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_CXX_STANDARD 20)


set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -DNO_FREETYPE -D_GLIBCXX_ASSERTIONS -fasynchronous-unwind-tables -fexceptions -fpie -Wl,-pie -fstack-clash-protection -fstack-protector-all -fstack-protector-strong -g -grecord-gcc-switches -fcf-protection -pipe -Wall -Werror -Wextra ")

set(CMAKE_CUDA_FLAGS "-extended-lambda ")

add_library(GLAD "libs/glad/glad.cu")
add_library(pugixml STATIC libs/pugixml/src/pugixml.cpp)

include_directories(libs/stb)


find_package(PNGwriter 0.7.0 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/includes)
include_directories(libs/pugixml/src)



add_executable(Raytracing
        main.cu
        src/utility/vector.h
        src/cudaHelpers.cu
        src/cudaHelpers.h
        src/scene/scene.cu
        src/scene/scene.h
        src/utility/ray.h
        src/hittable.cu
        src/hittable.h
        src/camera.cu
        src/camera.h
        src/constants.h
        src/utility/warp.h
        src/utility/sampler.h
        src/acceleration/bvh.h
        src/shapes/triangle.h
        src/utility/meshLoader.h
        src/utility/meshLoader.cu
        src/bsdf.cu
        src/bsdf.h
        src/acceleration/aabb.h
        src/utility/vector.cu src/scene/sceneLoader.cu src/scene/sceneLoader.h)



set_target_properties(Raytracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_dependencies(Raytracing pugixml)

target_link_libraries(Raytracing PRIVATE PNGwriter::PNGwriter glfw3 X11 GL GLAD dl pthread ${OPENEXR_LIBRARIES})

