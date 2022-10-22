//
// Created by steinraf on 20/08/22.
//

#pragma once

#include "utility/ray.h"
#include "hittable.h"


class Material {
public:
    __device__ virtual bool scatter(const Ray &rIn, const HitRecord &rec, Vector3f &attenuation, Ray &scattered,
                                    curandState *local_rand_state) const = 0;
};

class Lambertian : public Material {
public:
    __device__ explicit Lambertian(const Vector3f &a) : albedo(a) {}

    __device__ bool scatter(const Ray &rIn, const HitRecord &rec, Vector3f &attenuation, Ray &scattered,
                            curandState *localRandState) const override;

    Vector3f albedo;
};