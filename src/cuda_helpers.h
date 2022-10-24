//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <curand_kernel.h>

#include "utility/vector.h"
#include "utility/ray.h"
#include "hittableList.h"
#include "camera.h"
#include "utility/meshLoader.h"
#include "Acceleration/bvh.h"
#include "constants.h"
#include <cuda/std/limits>


#define checkCudaErrors(val) cuda_helpers::check_cuda( (val), #val, __FILE__, __LINE__ )


namespace cuda_helpers{

    __device__ bool inline initIndices(int &i, int &j, int &pixelIndex, const int width, const int height){
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;

        if((i >= width) || (j >= height)) return false;

        pixelIndex = j * width + i;

        return true;
    }

    __host__ void check_cuda(cudaError_t result, char const *func, const char *file, int line);


    __global__ void initRng(int width, int height, curandState *randState);

    // Pointer-pointers are used because cuda has problems with passing pointers to objects with virtual functions to global kernels
//    __global__ void
//    initVariables(Hittable **hittables, HittableList **hittableList, size_t numHittables);

//    template<typename Primitive>
//    __global__ void initVariables(BVH<Primitive> *bvh, Primitive *primitives, int numPrimitives);

    template<typename Primitive>
    __global__ void initVariables(BVH<Primitive> *bvh, Primitive *primitives, int numPrimitives){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, 1, 1)) return;

        *bvh = BVH<Primitive>(primitives, numPrimitives);
    }

    __global__ void freeVariables(int width, int height);

//    __device__ Color getColor(const Ray &r, HittableList **hittableList, int maxRayDepth, Sampler &sampler);
//    template<typename Primitive>
//    __device__ Color getColor(const Ray &r, BVH<Primitive> *bvh, int maxRayDepth, Sampler &sampler);

    template<typename Primitive>
    __device__ Color getColor(const Ray &r, BVH<Primitive> *bvh, int maxRayDepth, Sampler &sampler){

        HitRecord record;

        Ray currentRay = r;

        Ray scattered;
        Color attenuation;

        Color currentAttenuation{1.f};

        for(int depth = 0; depth < maxRayDepth; ++depth){
            if(bvh->hit(currentRay, 1e-4, cuda::std::numeric_limits<float>::infinity(), record)){
                if(record.triangle->bsdf.scatter(currentRay, record, attenuation, scattered, sampler)){
                    currentRay = scattered;
                    currentAttenuation *= attenuation;
                }else{
                    return Color{0.f};
                }

            }else{
                float t = 0.5f * (r.getDirection().normalized()[1] + 1.f);
                Color c = (1 - t) * Vector3f{1.f} + t * Color{0.5f, 0.7f, 1.0f};
                return currentAttenuation * c;
            }
        }

        return Vector3f{0.f};

    }

    __global__ void denoise(Vector3f *input, Vector3f *output, int width, int height);


//    __global__ void render(Vector3f *output, Camera cam, HittableList **hittableList, int width, int height,
//                           curandState *globalRandState);
//    template<typename Primitive>
//    __global__ void render(Vector3f *output, Camera cam, BVH<Primitive> *bvh, int width, int height,
//                           curandState *globalRandState);

    template<typename Primitive>
    __global__ void render(Vector3f *output, Camera cam, BVH<Primitive> *bvh, int width, int height,
                           curandState *globalRandState){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;


        auto sampler = Sampler(&globalRandState[pixelIndex]);

        const auto iFloat = static_cast<float>(i);
        const auto jFloat = static_cast<float>(j);

        const auto widthFloat = static_cast<float>(width);
        const auto heightFloat = static_cast<float>(height);

        const int maxRayDepth = customRenderer::getMaxRayDepth();
        const int numSubsamples = customRenderer::getNumSubsamples();

        Color col{0.0f};

        for(int subSamples = 0; subSamples < numSubsamples; ++subSamples){
            const float s = (iFloat + sampler.getSample1D()) / (widthFloat - 1);
            const float t = (jFloat + sampler.getSample1D()) / (heightFloat - 1);

            const auto ray = cam.getRay(s, t, sampler);

            col += getColor(ray, bvh, maxRayDepth, sampler);
        }

        constexpr float scale = 1.f / numSubsamples;

        col = {
                sqrt(col[0] * scale),
                sqrt(col[1] * scale),
                sqrt(col[2] * scale)
        };
        output[pixelIndex] = col;

    }


};


