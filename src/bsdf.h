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
        : material(Material::DIFFUSE){

    }

    __device__ __host__ constexpr BSDF(Material mat) noexcept
        : material(mat){

    }

    __device__ bool scatter(const Ray &rIn, const HitRecord &rec, Vector3f &attenuation, Ray &scattered,
                            Sampler &sampler) const;

private:
    Material material;
    Color3f albedo{0.5f};

};
