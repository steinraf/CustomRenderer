//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <filesystem>
#include <functional>

#include <pngwriter.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>


#include "cudaHelpers.h"
#include "utility/vector.h"
#include "hittableList.h"
#include "utility/meshLoader.h"

enum Device{
    CPU,
    GPU
};

struct PixelInfo{
    Color color;
    Vector3f intersection;
    Vector3f normal;
};


class Scene{
public:
    __host__ explicit Scene(HostMeshInfo &&mesh, int width = 384, int height = 216/*, int numHittables = 10*/,
                            Device= CPU);

    __host__ ~Scene();

    __host__ void render();

    __host__ void renderGPU();

    __host__ void renderCPU();


private:

    __host__ void initOpenGL();

    __host__ void OpenGLDraw(Vector3f *deviceVector, volatile bool &isRendering);

    __host__ void loadShader();

    __host__ void checkShaderCompileError(unsigned int shader, std::string type) const;

    const std::string fragmentShaderPath = "/home/steinraf/ETH/CG/CustomRenderer/shaders/fragmentShader.glsl";
    const std::string vertexShaderPath = "/home/steinraf/ETH/CG/CustomRenderer/shaders/vertexShader.glsl";

    HostMeshInfo mesh;

    const unsigned int blockSizeX = 16, blockSizeY = 16;

    const dim3 threadSize{blockSizeX, blockSizeY};

    const dim3 blockSize;

    const Device device;

    Vector3f *deviceImageBuffer;
    Vector3f *deviceImageBufferDenoised;
    const size_t imageBufferSize;

    Vector3f *hostImageBuffer;
    Vector3f *hostImageBufferDenoised;

    thrust::device_vector<Triangle> deviceTriangles;

//    Hittable **deviceHittables;
//    const size_t numHittables;

    Camera /* ** */deviceCamera;


//    HittableList **deviceHittableList;



    BVH<Triangle> *bvh;

    curandState *deviceCurandState;

    const int width, height;

    GLuint VAO, VBO, EBO, PBO;

    unsigned int shaderID;

    GLFWwindow *window;

    struct cudaGraphicsResource *cudaPBOResource;

};

