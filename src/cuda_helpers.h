//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <curand_kernel.h>

#include "utility/vector.h"
#include "utility/ray.h"
#include "hittableList.h"
#include "camera.h"

#define checkCudaErrors(val) cuda_helpers::check_cuda( (val), #val, __FILE__, __LINE__ )


namespace cuda_helpers {

    __host__ void check_cuda(cudaError_t result, char const *func, const char *file, int line);


    __global__ void initRng(int width, int height, curandState *randState);

    // Pointer-pointers are used because cuda has problems with passing pointers to objects with virtual functions to global kernels
    __global__ void initVariables(Hittable **hittables, HittableList **hittableList, size_t numHittables);

    __global__ void freeVariables(int width, int height);

    __device__ Color getColor(const Ray &r, HittableList **hittableList, int maxRayDepth, Sampler &sampler);


    __global__ void render(Vector3f *output, Camera cam, HittableList **hittableList, int width, int height,
                           curandState *globalRandState);

    __global__ void denoise(Vector3f *input, Vector3f *output, int width, int height);


    __device__ bool inline initIndices(int &i, int &j, int &pixelIndex, const int width, const int height) {
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i >= width) || (j >= height)) return false;

        pixelIndex = j * width + i;

        return true;
    }

};


