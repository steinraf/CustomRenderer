//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "utility/vector.h"
#include "utility/ray.h"
#include "utility/sampler.h"


class Camera{
public:
    __host__ __device__ Camera(Vector3f origin, Vector3f lookAt, Vector3f up, float vFOV, float aspectRatio,
                               float aperture,
                               float focusDist);

    __device__ Ray getRay(float s, float t, Sampler &sampler) const;

private:
    Vector3f origin;
    Vector3f lowerLeftCorner;
    Vector3f horizontal, vertical;
    Vector3f u, v, w;

    float lensRadius;

};
