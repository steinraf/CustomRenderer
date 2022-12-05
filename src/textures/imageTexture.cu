//
// Created by steinraf on 02/12/22.
//

#include "../cudaHelpers.h"
#include "imageTexture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


__host__ Texture::Texture(const std::filesystem::path &imagePath) noexcept {
    assert(!imagePath.string().empty());
    width = height = dim = 0;// make compiler not issue warnings
    printf("\tLoading texture %s\n", imagePath.c_str());
    auto *hostTexture = (Vector3f *) stbi_loadf(imagePath.c_str(), &width, &height, &dim, 3);


    printf("Size of the image is %i / %i\n", width, height);


    checkCudaErrors(cudaMalloc(&deviceTexture, width * height * sizeof(Vector3f)));
    checkCudaErrors(cudaMemcpy(deviceTexture, hostTexture, width * height * sizeof(Vector3f), cudaMemcpyHostToDevice));


    assert(dim == 3);
}