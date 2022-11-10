//
// Created by steinraf on 23/10/22.
//

#pragma once

#include "../utility/ray.h"
#include "../hittable.h"
#include "../bsdf.h"
#include "../acceleration/aabb.h"


class Triangle/* : public Hittable*/{
public:
    Triangle() = delete;

//    __host__ Triangle(Triangle &&) = default;

    __device__ __host__ Triangle(
            const Vector3f &p0, const Vector3f &p1, const Vector3f &p2,
            const Vector2f &uv0, const Vector2f &uv1, const Vector2f &uv2,
            const Vector3f &n0, const Vector3f &n1, const Vector3f &n2,
            const BSDF &bsdf)
            : p0(p0), p1(p1), p2(p2),
              uv0(uv0), uv1(uv1), uv2(uv2),
              n0(n0), n1(n1), n2(n2),
              bsdf(bsdf), boundingBox(p0, p1, p2){

//        printf("Initializing triangle with coordinates\n"
//               "(%f, %f, %f) \n"
//               "(%f, %f, %f) \n"
//               "(%f, %f, %f) \n",
//               p0[0], p0[1], p0[2],
//               p1[0], p1[1], p1[2],
//               p2[0], p2[1], p2[2]);
    }

    __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const;

    BSDF bsdf;
//private:
    Vector3f p0, p1, p2;
    Vector2f uv0, uv1, uv2;
    Vector3f n0, n1, n2;

    AABB boundingBox;

};

