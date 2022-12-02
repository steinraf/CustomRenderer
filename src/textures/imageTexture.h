//
// Created by steinraf on 02/12/22.
//

#pragma once


#include "../utility/vector.h"
#include "../cudaHelpers.h"
#include "../utility/sampler.h"
#include "../scene/sceneLoader.h"

#include "stb_image.h"


#include <filesystem>


class ImageTexture{
    Vector3f *deviceTexture;
    int width, height;
    int dim;

    __host__ explicit ImageTexture(const std::filesystem::path &imagePath) noexcept{
        assert(imagePath.string() != "");
        width = height = dim = 0; // make compiler not issue warnings
        deviceTexture = (Vector3f *)stbi_loadf(imagePath.c_str(), &width, &height, &dim, 3);

        assert(dim == 3);


//        checkCudaErrors(cudaMall)
    }

//    __device__



};
