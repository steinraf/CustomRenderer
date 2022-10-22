//
// Created by steinraf on 19/08/22.
//

#pragma once

#include "vector.h"


class Ray {
public:
    __device__ explicit Ray();

    __device__ Ray(const Vector3f &a, const Vector3f &b);

    __device__ Vector3f atTime(float t) const;

    __device__ Vector3f getOrigin() const { return origin; }

    __device__ Vector3f getDirection() const { return dir; }


private:
    Vector3f origin;
    Vector3f dir;
};

