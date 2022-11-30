//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "utility/vector.h"
#include "utility/ray.h"
#include "utility/sampler.h"


class Camera{
public:
    __host__ __device__ Camera(Vector3f origin, Vector3f lookAt, Vector3f _up, float vFOV, float aspectRatio,
                               float aperture,
                               float focusDist);

    __device__ Ray3f getRay(float s, float t, Sampler &sampler) const;

private:
    Vector3f origin;
    Vector3f upperLeft;
    Vector3f horizontal, vertical;
    Vector3f right, up, front;

    float lensRadius;

};
