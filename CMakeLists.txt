cmake_minimum_required(VERSION 3.21)
project(CustomRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



include_directories(libs/pugixml/src)


add_library(GLAD "libs/glad/glad.cu")

add_library(pugixml STATIC libs/pugixml/src/pugixml.cpp)

include_directories(libs/stb)


find_package(PNGwriter 0.7.0 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/includes)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -funroll-loops -DNO_FREETYPE -D_GLIBCXX_ASSERTIONS -fasynchronous-unwind-tables -fexceptions -fpie -Wl,-pie -fstack-clash-protection -fstack-protector-all -fstack-protector-strong -g -grecord-gcc-switches -fcf-protection -pipe -Wall -Werror -Wextra ")

set(CMAKE_CUDA_FLAGS "-extended-lambda --expt-relaxed-constexpr -dopt=on -extra-device-vectorization --restrict --compiler-options -Wall")




add_executable(CustomRenderer
        main.cu
        src/utility/vector.h
        src/cudaHelpers.cu
        src/cudaHelpers.h
        src/scene/scene.cu
        src/scene/scene.h
        src/utility/ray.h
        src/hittable.h
        src/camera/camera.cu
        src/camera/camera.h
        src/utility/warp.h
        src/utility/sampler.h
        src/acceleration/bvh.h
        src/shapes/triangle.h
        src/utility/meshLoader.h
        src/utility/meshLoader.cu
        src/bsdf.h
        src/acceleration/aabb.h
        src/scene/sceneLoader.h
        src/emitters/areaLight.h
        src/emitters/areaLight.cu
        src/utility/frame.h
        src/textures/imageTexture.cu
        src/textures/imageTexture.h src/emitters/environmentEmitter.h src/shapes/triangle.cu src/medium/medium.h src/medium/medium.cu src/medium/isotropicPhaseFunction.cu src/medium/isotropicPhaseFunction.h src/denoise/denoise.cu src/denoise/denoise.h)

add_dependencies(CustomRenderer pugixml)


set_target_properties(CustomRenderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(CustomRenderer PRIVATE PNGwriter::PNGwriter glfw3 X11 GL GLAD dl pthread pugixml ${OPENEXR_LIBRARIES})

