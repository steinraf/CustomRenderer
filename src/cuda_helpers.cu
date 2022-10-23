//
// Created by steinraf on 19/08/22.
//

#include "cuda_helpers.h"
#include "utility/ray.h"
#include "shapes/sphere.h"
#include "hittableList.h"
#include "material.h"
#include "constants.h"
#include "shapes/triangle.h"

#include <iostream>
#include <cuda/std/limits>


namespace cuda_helpers {

    __host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int line) {
        if (result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                      file << ":" << line << " '" << func << "' \n";
            cudaDeviceReset();
            exit(99);
        }
    }


    __global__ void initRng(int width, int height, curandState *randState) {
        int i, j, pixelIndex;
        if (!initIndices(i, j, pixelIndex, width, height)) return;

        curand_init(42, pixelIndex, 0, &randState[pixelIndex]);
    }

    __global__ void
    initVariables(Hittable **hittables, HittableList **hittableList, size_t numHittables) {
        int i, j, pixelIndex;
        if (!initIndices(i, j, pixelIndex, 1, 1)) return;

        hittableList[0] = new HittableList(hittables, numHittables);

        hittableList[0]->add(new Sphere({0, 0, -1}, 0.05, new Lambertian{{1.f, 0.0f, 0.83f}}));
//        hittableList[0]->add(new Sphere({0, -100.5, -1}, 100, new Lambertian{{0.f, 0.8f, 0.f}}));
        hittableList[0]->add(new Triangle({0, 0, -2},{2, 0, -2},{2, 2, -2}, new Lambertian{{0.f, 0.f, 0.83f}}));
    }

    __global__ void freeVariables(int width, int height) {
        int i, j, pixelIndex;
        if (!initIndices(i, j, pixelIndex, 1, 1)) return;


    }

    __device__ Color getColor(const Ray &r, HittableList **hittableList, int maxRayDepth, Sampler &sampler) {

        HitRecord record;

        Ray currentRay = r;

        Ray scattered;
        Color attenuation;

        Color currentAttenuation{1.f};

        for (int depth = 0; depth < maxRayDepth; ++depth) {
            if (hittableList[0]->hit(currentRay, 1e-4, cuda::std::numeric_limits<float>::infinity(), record)) {
                if (record.material->scatter(currentRay, record, attenuation, scattered, sampler)) {
                    currentRay = scattered;
                    currentAttenuation *= attenuation;
                } else {
                    return Color{0.f};
                }

            } else {
                float t = 0.5f * (r.getDirection().normalized()[1] + 1.f);
                Color c = (1 - t) * Vector3f{1.f} + t * Color{0.5f, 0.7f, 1.0f};
                return currentAttenuation * c;
            }
        }

        return Vector3f{0.f};

    }


    __global__ void render(Vector3f *output, Camera cam, HittableList **hittableList, int width, int height,
                           curandState *globalRandState) {
        int i, j, pixelIndex;
        if (!initIndices(i, j, pixelIndex, width, height)) return;


        auto sampler = Sampler(&globalRandState[pixelIndex]);

        const auto iFloat = static_cast<float>(i);
        const auto jFloat = static_cast<float>(j);

        const auto widthFloat = static_cast<float>(width);
        const auto heightFloat = static_cast<float>(height);

        const int maxRayDepth = customRenderer::getMaxRayDepth();
        const int numSubsamples = customRenderer::getNumSubsamples();

        Color col{0.0f};

        for (int subSamples = 0; subSamples < numSubsamples; ++subSamples) {
            const float s = (iFloat + sampler.getSample1D()) / (widthFloat - 1);
            const float t = (jFloat + sampler.getSample1D()) / (heightFloat - 1);

            const auto ray = cam.getRay(s, t, sampler);

            col += getColor(ray, hittableList, maxRayDepth, sampler);
        }

        constexpr float scale = 1.f / numSubsamples;

        col = {
                sqrt(col[0] * scale),
                sqrt(col[1] * scale),
                sqrt(col[2] * scale)
        };
        output[pixelIndex] = col;

    }

    __global__ void denoise(Vector3f *input, Vector3f *output, int width, int height) {
        int i, j, pixelIndex;
        if (!initIndices(i, j, pixelIndex, width, height)) return;

//        output[pixelIndex] = input[pixelIndex] > ;

        Vector3f tmp{0.f};
        int count = 0;

        auto between = [](int val, int low, int high){return val >= low && val < high;};

        const int range = 2;

        float filter[5][5] = {
                {0, 0, 0, 0, 0},
                {0, 0, 0.125f, 0, 0},
                {0, 0.125f, 0.5f, 0.125f, },
                {0, 0, 0.125f, 0, 0},
                {0, 0, 0, 0, 0},
        };

        for(int a = -range; a <= range; ++a){
            for(int b = -range; b <= range; ++b) {
                if(between(i+a, 0, width) && between(j+b, 0, height)){
                    ++count;
                    tmp += filter[range+a][range+b] * input[(j+b)*width + i + a];
                }
            }
        }

        output[pixelIndex] = tmp.clamp(0.f, 1.f); //(input[pixelIndex] + tmp/(count))/2.0;
    }
}
