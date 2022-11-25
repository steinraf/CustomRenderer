//
// Created by steinraf on 24/10/22.
//

#pragma once

#include "utility/ray.h"
#include "hittable.h"
#include "utility/vector.h"
#include "utility/sampler.h"
#include "utility/warp.h"


enum class Material{
    DIFFUSE,
};


class BSDF{
public:

    __device__ __host__ constexpr BSDF() noexcept
        : material(Material::DIFFUSE), albedo(1.f){

    }

    __device__ __host__ constexpr BSDF(Material mat, Color3f albedo=Color3f{1.5f}) noexcept
        : material(mat), albedo(albedo){

    }

    __device__ bool scatter(const Ray &rIn, const Intersection &rec, Vector3f &attenuation, Ray &scattered,
                            Sampler &sampler) const;

//private:
    Material material;
    Color3f albedo;

};
