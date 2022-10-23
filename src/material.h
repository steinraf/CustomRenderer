//
// Created by steinraf on 20/08/22.
//

#pragma once

#include "utility/ray.h"
#include "hittable.h"
#include "utility/vector.h"
#include "utility/sampler.h"


//class BSDF {
//    __device__ Color sample(BSDFRecord &rec, )
//};


class Material {
public:
    __device__ virtual bool scatter(const Ray &rIn, const HitRecord &rec, Vector3f &attenuation, Ray &scattered,
                                    Sampler &sampler) const = 0;
};

class Lambertian : public Material {
public:
    __device__ explicit Lambertian(const Vector3f &a) : albedo(a) {}

    __device__ bool scatter(const Ray &rIn, const HitRecord &rec, Vector3f &attenuation, Ray &scattered,
                            Sampler &sampler) const override;

    Vector3f albedo;
};