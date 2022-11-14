//
// Created by steinraf on 24/10/22.
//

#pragma once

#include "../utility/vector.h"
#include "../utility/ray.h"

#include <cuda/std/limits>
#include <thrust/extrema.h>


struct AABB{

    __host__ __device__ AABB() : min(INFINITY), max(-INFINITY){}

    __host__ __device__ AABB(const Vector3f &min, const Vector3f max) : min(min), max(max){}

    __host__ __device__ AABB(const Vector3f &a, const Vector3f &b, const Vector3f &c);

    //Nori BoundingBox RayIntersect
    __device__ bool rayIntersect(const Ray &ray, float &nearT, float &farT) const;

    __device__ __host__ Vector3f getCenter() const{
        return 0.5f * (min + max);
    }

    [[nodiscard]] __device__ bool isEmpty() const noexcept{
        return min == Vector3f(INFINITY) && max == Vector3f(-INFINITY);
    }

    __device__ AABB operator+(const AABB &other) const;

//private:
    Vector3f min, max;
};
