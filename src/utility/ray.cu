//
// Created by steinraf on 19/08/22.
//

#include "ray.h"


__device__ Ray::Ray()
        : origin{0.f, 0.f, 0.f}, dir{1.f, 0.f, 0.f}{

}

__device__ Ray::Ray(const Vector3f &origin, const Vector3f &direction)
    :origin(origin), dir(direction){

}

__device__ Vector3f Ray::atTime(float t) const{
    return origin + t * dir;
}

