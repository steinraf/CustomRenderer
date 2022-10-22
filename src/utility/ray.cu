//
// Created by steinraf on 19/08/22.
//

#include "ray.h"


__device__ Ray::Ray()
        : origin{1.0, 0.0, 0.0}, dir{1.0, 0.0, 0.0} {

}

__device__ Ray::Ray(const Vector3f &a, const Vector3f &b) {
    origin = a;
    dir = b;
}

__device__ Vector3f Ray::atTime(float t) const {
    return origin + t * dir;
}

