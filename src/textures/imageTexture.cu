//
// Created by steinraf on 02/12/22.
//

#include "../cudaHelpers.h"
#include "imageTexture.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_scan.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


__host__ Texture::Texture(const std::filesystem::path &imagePath) noexcept {
    assert(!imagePath.string().empty());
    width = height = dim = 0;// make compiler not issue warnings
    printf("\tLoading texture %s\n", imagePath.c_str());
    auto *hostTexture = (Vector3f *) stbi_loadf(imagePath.c_str(), &width, &height, &dim, 3);

    std::cout << imagePath << " texture has first value " << hostTexture[0] << '\n';


    printf("Size of the image is %i / %i\n", width, height);


    checkCudaErrors(cudaMalloc(&deviceTexture, width * height * sizeof(Vector3f)));
    checkCudaErrors(cudaMalloc(&deviceCDF, width * height * sizeof(float)));

    checkCudaErrors(cudaMemcpy(deviceTexture, hostTexture, width * height * sizeof(Vector3f), cudaMemcpyHostToDevice));


    //TODO is this actually called Radiance?
    ColorToRadiance colorToRadiance(deviceTexture, width, height);

    thrust::device_ptr<Vector3f> deviceTexturePtr{deviceTexture};
    float totalSum = thrust::transform_reduce(deviceTexturePtr, deviceTexturePtr + width*height,
                                                    colorToRadiance, 0.f,  thrust::plus<float>());

    //TODO remove
    std::cout << "Total texture sum is " << totalSum << '\n';


    ColorToCDF colorToCdf{deviceTexture, width, height, totalSum};

    thrust::transform_inclusive_scan(deviceTexturePtr, deviceTexturePtr + width*height,
                                     deviceCDF, colorToCdf, thrust::plus<float>());


    assert(dim == 3);
}