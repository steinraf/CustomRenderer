//
// Created by steinraf on 19/08/22.
//

#pragma once

#include "vector.h"
#include <cuda/std/limits>


class Ray{
public:
    [[nodiscard]] __device__ constexpr Ray()
        : origin{0.f, 0.f, 0.f}, dir{1.f, 0.f, 0.f}
        , minDist(FLT_EPSILON), maxDist(cuda::std::numeric_limits<float>::infinity()){

    }

    [[nodiscard]] __device__ constexpr Ray(const Vector3f &origin, const Vector3f &direction
                                           , float minDist = FLT_EPSILON,
                                           float maxDist = cuda::std::numeric_limits<float>::infinity()) noexcept
        : origin(origin), dir(direction), minDist(minDist), maxDist(maxDist){

    }

    [[nodiscard]] __device__ constexpr Vector3f atTime(float t) const noexcept {return origin + t * dir;}

    [[nodiscard]] __device__ constexpr Vector3f getOrigin() const noexcept { return origin; }

    [[nodiscard]] __device__ constexpr Vector3f getDirection() const noexcept { return dir; }


//private:
    Vector3f origin;
    Vector3f dir;
    float minDist;
    float maxDist;
};

