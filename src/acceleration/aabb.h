//
// Created by steinraf on 24/10/22.
//

#pragma once

#include "../utility/vector.h"
#include "../utility/ray.h"

#include <cuda/std/limits>
#include <thrust/extrema.h>


struct AABB{

    __device__ AABB(const Vector3f &min, const Vector3f max) : min(min), max(max){}

    __device__ AABB(const Vector3f &a, const Vector3f &b, const Vector3f &c);

    //Nori BoundingBox RayIntersect
    __device__ bool rayIntersect(const Ray &ray, float &nearT, float &farT) const;

    __device__ AABB operator+(const AABB &other);

private:
    Vector3f min, max;
};
