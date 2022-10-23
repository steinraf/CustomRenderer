cmake_minimum_required(VERSION 3.21)
project(Raytracing LANGUAGES CXX CUDA)

set(CMAKE_CUDA_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -DNO_FREETYPE -D_GLIBCXX_ASSERTIONS -fasynchronous-unwind-tables -fexceptions -fpie -Wl,-pie -fstack-clash-protection -fstack-protector-all -fstack-protector-strong -g -grecord-gcc-switches -fcf-protection -pipe -Wall -Werror ")

set(CMAKE_CUDA_FLAGS "-extended-lambda ")

add_library(GLAD "libs/glad/glad.cu")


find_package(PNGwriter 0.7.0 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)


    add_executable(Raytracing
        main.cu
        src/utility/vector.h
        src/cuda_helpers.cu
        src/cuda_helpers.h
        src/scene.cu
        src/scene.h
        src/utility/ray.cu
        src/utility/ray.h
        src/hittable.cu
        src/hittable.h
        src/shapes/sphere.cu
        src/shapes/sphere.h
        src/hittableList.cu
        src/hittableList.h
        src/camera.cu
        src/camera.h
        src/material.cu
        src/material.h
        src/constants.h
        src/utility/warp.h
        src/utility/sampler.h
        src/Acceleration/bvh.h
        src/Acceleration/bvh.cu
            src/shapes/triangle.cu
        src/shapes/triangle.h)

set_target_properties(Raytracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Raytracing PRIVATE PNGwriter::PNGwriter glfw3 X11 GL GLAD dl pthread )

include_directories(${CMAKE_SOURCE_DIR}/includes)