//
// Created by steinraf on 19/08/22.
//

#pragma once


#include <filesystem>
#include <functional>

#include "sceneLoader.h"

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "cuda_gl_interop.h"
#include "pngwriter.h"


#include "../cudaHelpers.h"
#include "../utility/meshLoader.h"
#include "../utility/vector.h"


enum Device {
    CPU,
    GPU
};

struct PixelInfo {
    Color3f color;
    Vector3f intersection;
    Vector3f normal;
};


class Scene {
public:
    __host__ explicit Scene(SceneRepresentation &&sceneRepr, Device = CPU);

    __host__ ~Scene();

    __host__ void render();

    __host__ void renderGPU();

    __host__ void renderCPU();


private:
    __host__ void initOpenGL();

    __host__ void OpenGLDraw(Vector3f *deviceVector, volatile bool &isRendering);

    __host__ void loadShader();

    __host__ static void checkShaderCompileError(unsigned int shader, const std::string &type);

    const std::string fragmentShaderPath = "/home/steinraf/ETH/CG/CustomRenderer/shaders/fragmentShader.glsl";
    const std::string vertexShaderPath = "/home/steinraf/ETH/CG/CustomRenderer/shaders/vertexShader.glsl";

    SceneRepresentation sceneRepresentation;
    //    HostMeshInfo mesh;

    const unsigned int blockSizeX = 16, blockSizeY = 16;

    const dim3 threadSize{blockSizeX, blockSizeY};

    const dim3 blockSize;

    const Device device;

    Vector3f *deviceImageBuffer;

    FeatureBuffer *deviceFeatureBuffer;

    Vector3f *deviceImageBufferDenoised;
    const size_t imageBufferByteSize;

    Vector3f *hostImageBuffer;
    Vector3f *hostImageBufferDenoised;

    std::vector<thrust::device_vector<Triangle>> hostDeviceMeshTriangleVec;
    std::vector<thrust::device_vector<float>> hostDeviceMeshCDF;
    std::vector<float> totalMeshArea;

    std::vector<thrust::device_vector<Triangle>> hostDeviceEmitterTriangleVec;
    std::vector<thrust::device_vector<float>> hostDeviceEmitterCDF;
    std::vector<float> totalEmitterArea;

    Camera deviceCamera;


    TLAS *meshAccelerationStructure;
//    TLAS *emitterAccelerationStructure;

    curandState *deviceCurandState;

    GLuint VAO, VBO, EBO, PBO;

    unsigned int shaderID;

    GLFWwindow *window;

    struct cudaGraphicsResource *cudaPBOResource;
};
