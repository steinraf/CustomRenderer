//
// Created by steinraf on 19/08/22.
//

#include "scene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include <fstream>
#include <thread>

__host__ Scene::Scene(SceneRepresentation &&sceneRepr, Device dev) : sceneRepresentation(sceneRepr),
                                                                     imageBufferByteSize(sceneRepr.sceneInfo.width * sceneRepr.sceneInfo.height * sizeof(Vector3f)),
                                                                     blockSize(sceneRepr.sceneInfo.width / blockSizeX + 1, sceneRepr.sceneInfo.height / blockSizeY + 1),
                                                                     device(dev),
                                                                     hostDeviceMeshTriangleVec(sceneRepresentation.meshInfos.size()),
                                                                     hostDeviceMeshCDF(sceneRepresentation.meshInfos.size()),
                                                                     totalMeshArea(sceneRepresentation.meshInfos.size()),
                                                                     hostDeviceEmitterTriangleVec(sceneRepresentation.emitterInfos.size()),
                                                                     hostDeviceEmitterCDF(sceneRepresentation.emitterInfos.size()),
                                                                     totalEmitterArea(sceneRepresentation.emitterInfos.size()),
                                                                     deviceCamera(sceneRepr.cameraInfo.origin,
                                                                                  sceneRepr.cameraInfo.target,
                                                                                  sceneRepr.cameraInfo.up,
                                                                                  sceneRepr.cameraInfo.fov,
                                                                                  static_cast<float>(sceneRepr.sceneInfo.width) / static_cast<float>(sceneRepr.sceneInfo.height),
                                                                                  sceneRepr.cameraInfo.aperture,
                                                                                  sceneRepr.cameraInfo.focusDist) {//(customRenderer::getCameraOrigin() - customRenderer::getCameraLookAt()).norm()){


    if(dev == CPU) {
        checkCudaErrors(cudaMalloc(&deviceImageBuffer, imageBufferByteSize));
        checkCudaErrors(cudaMalloc(&deviceImageBufferDenoised, imageBufferByteSize));
    } else {

        //        checkCudaErrors(cudaMalloc(&deviceImageBuffer, imageBufferByteSize)); Allocated by OpenGL instead
        checkCudaErrors(cudaMalloc(&deviceImageBufferDenoised, imageBufferByteSize));
        initOpenGL();
    }

    checkCudaErrors(cudaMalloc(&deviceFeatureBuffer, sizeof(FeatureBuffer) * sceneRepresentation.sceneInfo.width *
                                                             sceneRepresentation.sceneInfo.height));


    checkCudaErrors(cudaMalloc(&deviceCurandState,
                               sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height *
                                       sizeof(curandState)));

    cudaHelpers::initRng<<<blockSize, threadSize>>>(sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, deviceCurandState);
    checkCudaErrors(cudaGetLastError());


    checkCudaErrors(cudaMalloc(&meshAccelerationStructure, sizeof(TLAS)));
//    checkCudaErrors(cudaMalloc(&emitterAccelerationStructure, sizeof(TLAS)));

    // No need to sync because can run independently
    //    checkCudaErrors(cudaDeviceSynchronize());

    auto numMeshes = sceneRepresentation.meshInfos.size();


    std::vector<BLAS *> hostMeshBlasVector(numMeshes);


    clock_t meshLoadStart = clock();
#pragma omp parallel for
    for(size_t i = 0; i < numMeshes; ++i) {
        hostMeshBlasVector[i] = getMeshFromFile(sceneRepr.meshInfos[i].filename,
                                                hostDeviceMeshTriangleVec[i],
                                                hostDeviceMeshCDF[i],
                                                totalMeshArea[i],
                                                sceneRepr.meshInfos[i].transform,
                                                sceneRepr.meshInfos[i].bsdf,
                                                sceneRepr.meshInfos[i].normalMap);
    }

    auto numEmitters = sceneRepresentation.emitterInfos.size();

    std::vector<BLAS *> hostEmitterBlasVector(numEmitters);

    std::vector<AreaLight> hostAreaLights(numEmitters);
    for(size_t i = 0; i < numEmitters; ++i) {
        hostAreaLights[i] = AreaLight(sceneRepr.emitterInfos[i].radiance);
    }

    AreaLight *deviceAreaLights;
    checkCudaErrors(cudaMalloc(&deviceAreaLights, sizeof(AreaLight) * numEmitters));
    checkCudaErrors(cudaMemcpy(deviceAreaLights, hostAreaLights.data(), sizeof(AreaLight) * numEmitters,
                               cudaMemcpyHostToDevice));


#pragma omp parallel for
    for(size_t i = 0; i < numEmitters; ++i) {
        hostEmitterBlasVector[i] = getMeshFromFile(sceneRepr.emitterInfos[i].filename,
                                                   hostDeviceEmitterTriangleVec[i],
                                                   hostDeviceEmitterCDF[i],
                                                   totalEmitterArea[i],
                                                   sceneRepr.emitterInfos[i].transform,
                                                   sceneRepr.emitterInfos[i].bsdf,
                                                   sceneRepr.emitterInfos[i].normalMap,
                                                   deviceAreaLights + i);
    }


    std::cout << "Loading all Geometry took "
              << ((double) (clock() - meshLoadStart)) / CLOCKS_PER_SEC
              << " seconds.\n";


    BLAS **deviceBlasArr = cudaHelpers::hostVecToDeviceRawPtr(hostMeshBlasVector);
    BLAS **deviceEmitterBlasArr = cudaHelpers::hostVecToDeviceRawPtr(hostEmitterBlasVector);

    cudaHelpers::constructTLAS<<<1, 1>>>(meshAccelerationStructure,
                                         deviceBlasArr, numMeshes,
                                         deviceEmitterBlasArr, numEmitters,
                                         EnvironmentEmitter{sceneRepresentation.environmentInfo.texture});

    checkCudaErrors(cudaGetLastError());

    hostImageBuffer = new Vector3f[imageBufferByteSize];
    hostImageBufferDenoised = new Vector3f[imageBufferByteSize];
}

__host__ Scene::~Scene() {


    delete[] hostImageBuffer;
    delete[] hostImageBufferDenoised;

    if(device == CPU) {
        checkCudaErrors(cudaDeviceSynchronize());
        //                checkCudaErrors(cudaFree(deviceImageBuffer));
        //        glDeleteVertexArrays(1, &VAO);
        //        glDeleteBuffers(1, &VBO);
        //        glDeleteBuffers(1, &EBO);
    } else {
        cudaGraphicsUnmapResources(1, &cudaPBOResource, nullptr);
    }
    //    glfwTerminate();
    cudaHelpers::freeVariables<<<blockSize, threadSize>>>();
}


void Scene::render() {

    volatile bool currentlyRendering = true;

    std::thread drawingThread;

    if(device == GPU) {
        drawingThread = std::thread{[this](Vector3f *v, volatile bool &render) {
                                        OpenGLDraw(v, render);
                                    },
                                    deviceImageBuffer, std::ref(currentlyRendering)};
    }


    std::cout << "Starting render...\n";

    unsigned *deviceCounter;

    checkCudaErrors(cudaMalloc(&deviceCounter, sizeof(unsigned)));
    checkCudaErrors(cudaMemset(deviceCounter, 0, sizeof(unsigned)));

    std::cout << "Starting draw thread...\n";


    cudaHelpers::render<<<blockSize, threadSize>>>(deviceImageBuffer, deviceCamera, meshAccelerationStructure,
                                                   sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, sceneRepresentation.sceneInfo.samplePerPixel, sceneRepresentation.sceneInfo.maxRayDepth,
                                                   deviceCurandState, deviceFeatureBuffer, deviceCounter);
    checkCudaErrors(cudaGetLastError());


    std::cout << "Synchronizing GPU...\n";
    checkCudaErrors(cudaDeviceSynchronize());


    std::cout << "Starting denoise...\n";
    checkCudaErrors(cudaMemcpy(hostImageBuffer, deviceImageBuffer, imageBufferByteSize, cudaMemcpyDeviceToHost));
    float *deviceWeights;
    checkCudaErrors(cudaMalloc(&deviceWeights, sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height * sizeof(float)));
    checkCudaErrors(cudaMemset(deviceWeights, 0.f, sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height * sizeof(float)));

    cudaHelpers::denoise<<<blockSize, threadSize>>>(deviceImageBuffer, deviceImageBufferDenoised, deviceFeatureBuffer, deviceWeights, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, sceneRepresentation.cameraInfo.origin);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaHelpers::denoiseApplyWeights<<<blockSize, threadSize>>>(deviceImageBufferDenoised, deviceWeights, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(deviceWeights));

    checkCudaErrors(
            cudaMemcpy(hostImageBufferDenoised, deviceImageBufferDenoised, imageBufferByteSize,
                       cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());


    currentlyRendering = false;

    if(device == GPU) {
        drawingThread.join();
    }


    std::cout << "Joined draw thread...\n";
}

__host__ void Scene::renderGPU() {


    for(int i = 0; i < 10000; ++i) {
        std::cout << "Rendering frame " << i << '\n';
        render();
    }
}

__host__ void Scene::renderCPU() {
    //    initOpenGL();

    clock_t start, stop;
    start = clock();

    render();

    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cout << "Computation took " << timer_seconds << " seconds.\n";

    std::cout << "Writing resulting image to disk...\n";

    if(!std::filesystem::exists("./data"))
        std::filesystem::create_directory("./data");

    const std::string pngPath = "./data/image.png";
    const std::string pngPathDenoised = "./data/imageDenoised.png";

    const std::string hdrPath = "./data/image.hdr";
    const std::string hdrPathDenoised = "./data/imageDenoised.hdr";

    //    stbi_set_flip_vertically_on_load(true);

    const bool didHDR = stbi_write_hdr(hdrPath.c_str(), sceneRepresentation.sceneInfo.width,
                                       sceneRepresentation.sceneInfo.height, 3, (float *) hostImageBuffer);
    assert(didHDR);

    const bool didHDRDenoised = stbi_write_hdr(hdrPathDenoised.c_str(), sceneRepresentation.sceneInfo.width,
                                       sceneRepresentation.sceneInfo.height, 3, (float *) hostImageBufferDenoised);
    assert(didHDRDenoised);


    pngwriter png(sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, 1., pngPath.c_str());
    pngwriter pngDenoised(sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, 1.,
                          pngPathDenoised.c_str());

    auto gammaCorrect = [](float value) {
        if(value <= 0.0031308f) return std::clamp(12.92f * value, 0.f, 1.f);
        return std::clamp(1.055f * std::pow(value, 1.f / 2.4f) - 0.055f, 0.f, 1.f);
    };


#pragma omp parallel for
    for(int j = 0; j < sceneRepresentation.sceneInfo.height; j++) {
        for(int i = 0; i < sceneRepresentation.sceneInfo.width; i++) {
            const int idx = j * sceneRepresentation.sceneInfo.width + i;
            png.plot(i + 1, sceneRepresentation.sceneInfo.height - j,
                     gammaCorrect(hostImageBuffer[idx][0]),
                     gammaCorrect(hostImageBuffer[idx][1]),
                     gammaCorrect(hostImageBuffer[idx][2]));
            pngDenoised.plot(i + 1, sceneRepresentation.sceneInfo.height - j,
                             gammaCorrect(hostImageBufferDenoised[idx][0]),
                             gammaCorrect(hostImageBufferDenoised[idx][1]),
                             gammaCorrect(hostImageBufferDenoised[idx][2]));
        }
    }

    png.close();
    pngDenoised.close();
}

__host__ void Scene::initOpenGL() {
    assert(false);
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, "Raytracing",
                              nullptr, nullptr);
    if(!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        throw std::runtime_error("GLFW WINDOW ERROR");
    }

    glfwMakeContextCurrent(window);
    //    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    //    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glfwSwapInterval(1);

    if(!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        throw std::runtime_error("GLAD INIT ERROR");
    }

    loadShader();

    glEnable(GL_DEPTH_TEST);


    switch(device) {
        break;
        case GPU:

            glGenBuffers(1, &PBO);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
            glBufferData(GL_PIXEL_UNPACK_BUFFER,
                         sizeof(Vector3f) * sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height,
                         nullptr, GL_STREAM_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            //    glGenTextures(1, &tex);
            //    glBindTexture(GL_TEXTURE_2D, tex);
            //    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

            checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaPBOResource, PBO, cudaGraphicsRegisterFlagsNone));

            checkCudaErrors(cudaGraphicsMapResources(1, &cudaPBOResource, nullptr));

            checkCudaErrors(
                    cudaGraphicsResourceGetMappedPointer((void **) &deviceImageBuffer,
                                                         const_cast<size_t *>(&imageBufferByteSize), cudaPBOResource));

            checkCudaErrors(cudaDeviceSynchronize());
            break;
        case CPU:
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);

            //            glGenBuffers(1, &EBO);
    }
}

__host__ void Scene::OpenGLDraw(Vector3f *deviceVector, volatile bool &isRendering) {

    glGenVertexArrays(1, &VAO);

    std::cout << "Starting OpenGLDraw\n";

    float vertices[] = {
            -0.5f, -0.5f, 0.0f,
            0.5f, -0.5f, 0.0f,
            0.0f, 0.5f, 0.0f};

    std::cout << "Hehe 2";

    //    float *vertices;

    glBindVertexArray(VAO);

    std::cout << "Hehe 2.5";

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    std::cout << "Hehe 3";

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0 * 3 * sizeof(float), nullptr);

    std::cout << "Hehe 4";

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    std::cout << "Hehe 1";
    while(isRendering) {
        //        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);


        glUseProgram(shaderID);
        glBindVertexArray(VAO);
        //        checkCudaErrors(cudaMemcpy(hostImageBuffer, deviceImageBuffer, imageBufferByteSize, cudaMemcpyDeviceToHost));

        //        std::cout << deviceImageBuffer[0] << '\n';
        //        glBufferData(GL_ARRAY_BUFFER, 2 * width * height * sizeof(Vector3f), NULL, GL_STATIC_DRAW);
        //
        //        glBufferSubData(GL_ARRAY_BUFFER, 0, width * height * sizeof(Vector3f), hostImageBuffer);
        //        glBufferSubData(GL_ARRAY_BUFFER, width * height * sizeof(Vector3f), width * height * sizeof(Vector3f), hostCoordinateVector);

        //        checkCudaErrors(cudaDeviceSynchronize());
        //        glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(Vector3f), hostImageBuffer, GL_STATIC_DRAW);


        //            glDrawArrays(GL_POINTS, 0, width*height);

        //        glEnable(GL_PROGRAM_POINT_SIZE);


        glDrawArrays(GL_TRIANGLES, 0, sizeof(vertices) / 3);


        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    std::cout << "Exiting :sadge:\n";
}

__host__ void Scene::loadShader() {
    std::string vertexCode, fragmentCode;
    std::ifstream vShaderFile, fShaderFile;

    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {

        vShaderFile.open(vertexShaderPath);
        fShaderFile.open(fragmentShaderPath);

        std::stringstream vShaderStream, fShaderStream;

        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();

        vShaderFile.close();
        fShaderFile.close();

        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();
    } catch(std::ifstream::failure &e) {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
        std::cout << fragmentShaderPath << " " << vertexShaderPath << '\n';
        throw std::runtime_error("Shader File not readable");
    }
    const char *vShaderCode = vertexCode.c_str();
    const char *fShaderCode = fragmentCode.c_str();


    unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, nullptr);
    glCompileShader(vertex);
    checkShaderCompileError(vertex, "VERTEX");

    unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, nullptr);
    glCompileShader(fragment);
    checkShaderCompileError(fragment, "FRAGMENT");

    shaderID = glCreateProgram();
    glAttachShader(shaderID, vertex);
    glAttachShader(shaderID, fragment);
    glLinkProgram(shaderID);
    checkShaderCompileError(shaderID, "PROGRAM");

    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

__host__ void Scene::checkShaderCompileError(unsigned int shader, const std::string &type) {
    GLint success;
    GLchar infoLog[1024];
    if(type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success) {
            glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                      << infoLog
                      << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if(!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                      << infoLog
                      << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}
