//
// Created by steinraf on 23/10/22.
//

#pragma once

#include "../utility/ray.h"
#include "../hittable.h"
#include "../bsdf.h"


class Triangle/* : public Hittable*/{
public:
    Triangle() = delete;

//    __host__ Triangle(Triangle &&) = default;

    __device__ __host__ Triangle(const Vector3f &p0, const Vector3f &p1, const Vector3f &p2, BSDF bsdf)
            : p0(p0), p1(p1), p2(p2), bsdf(bsdf){}

    __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const/* override*/;

    BSDF bsdf;
private:
    Vector3f p0, p1, p2;
};
