//
// Created by steinraf on 19/08/22.
//

#include "scene.h"

__host__ Scene::Scene(int width, int height, int numHittables, Device dev) : width(width), height(height),
                                               imageBufferSize(width * height * sizeof(Vector3f)),
                                               blockSize(width / blockSizeX + 1, height / blockSizeY + 1),
                                               numHittables(numHittables), device(dev){

//    std::cout << "Initializing scene with " << width  << ' ' << blockSizeX << '\n';

    if(dev == CPU){
        checkCudaErrors(cudaMalloc((void **) &deviceImageBuffer, imageBufferSize));
        checkCudaErrors(cudaMalloc((void **) &deviceImageBufferDenoised, imageBufferSize));
    }else{

        checkCudaErrors(cudaMalloc((void **) &deviceImageBuffer, imageBufferSize));

        initOpenGL();
    }


    checkCudaErrors(cudaMalloc((void **) &deviceCurandState, width * height * sizeof(curandState)));

    checkCudaErrors(cudaMalloc((void **) &deviceHittables, numHittables * sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc((void **) &deviceHittableList, sizeof(HittableList *)));

    checkCudaErrors(cudaMalloc((void **) &deviceCamera, sizeof(Camera *)));


    cuda_helpers::initRng<<<blockSize, threadSize>>>(width, height, deviceCurandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cuda_helpers::initVariables<<<1, 1>>>(deviceCamera, deviceHittables, deviceHittableList, numHittables, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hostImageBuffer = new Vector3f[imageBufferSize];
    hostImageBufferDenoised = new Vector3f[imageBufferSize];

}

__host__ Scene::~Scene() {


    delete[] hostImageBuffer;
    delete[] hostImageBufferDenoised;

    if(device == CPU){
        //        checkCudaErrors(cudaDeviceSynchronize());
        //        checkCudaErrors(cudaFree(deviceImageBuffer));
    }else{
        cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);
    }

    cuda_helpers::freeVariables<<<blockSize, threadSize>>>(width, height);
}



void Scene::render(){

    cuda_helpers::render<<<blockSize, threadSize>>>(deviceImageBuffer, deviceCamera, deviceHittableList, width, height, deviceCurandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    checkCudaErrors(cudaMemcpy(hostImageBuffer, deviceImageBuffer, imageBufferSize, cudaMemcpyDeviceToHost));
    cuda_helpers::denoise<<<blockSize, threadSize>>>(deviceImageBuffer, deviceImageBufferDenoised, width, height);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(hostImageBufferDenoised, deviceImageBufferDenoised, imageBufferSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

}

__host__ void Scene::renderGPU() {


//    for(int i = 0; i < 10000; ++i){
//        std::cout << "Rendering frame " << i << '\n';
//        render();
//    }






}

__host__ void Scene::renderCPU() {

    render();

    const std::string base_path = std::filesystem::path(__FILE__).parent_path().parent_path();
    const std::string pngPath = base_path + "/data/image.png";
    const std::string pngPathDenoised = base_path + "/data/imageDenoised.png";



    pngwriter png(width, height, 1., pngPath.c_str());
    pngwriter pngDenoised(width, height, 1., pngPathDenoised.c_str());


    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            const int idx = j * width + i;
            png.plot(i + 1, j + 1, hostImageBuffer[idx][0], hostImageBuffer[idx][1], hostImageBuffer[idx][2]);
            pngDenoised.plot(i + 1, j + 1, hostImageBufferDenoised[idx][0], hostImageBufferDenoised[idx][1],
                             hostImageBufferDenoised[idx][2]);
        }
    }

    png.close();
    pngDenoised.close();
}

__host__ void Scene::initOpenGL(){
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    auto window = glfwCreateWindow(width, height, "OpenGL Project", NULL, NULL);
    if (!window){
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        throw std::runtime_error("GLFW WINDOW ERROR");
    }

    glfwMakeContextCurrent(window);
//    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
//    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "Failed to initialize GLAD" << std::endl;
        throw std::runtime_error("GLAD INIT ERROR");
        return;
    }

    glEnable(GL_DEPTH_TEST);


    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(Vector3f) * width * height, nullptr, GL_STREAM_DRAW);


//    glGenTextures(1, &tex);
//    glBindTexture(GL_TEXTURE_2D, tex);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaPBOResource, PBO, cudaGraphicsRegisterFlagsNone));

    checkCudaErrors(cudaGraphicsMapResources(1, &cudaPBOResource, nullptr));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&deviceImageBufferDenoised, NULL, cudaPBOResource));



}
