//
// Created by steinraf on 24/10/22.
//

#pragma once

#include "utility/ray.h"
#include "hittable.h"
#include "utility/vector.h"
#include "utility/sampler.h"
#include "utility/warp.h"

enum EMeasure {
    EUnknownMeasure = 0,
    ESolidAngle,
    EDiscrete
};

struct BSDFQueryRecord{
    Vector3f wi;
    Vector3f wo;

    Vector3f p;
    Vector2f uv;

    float eta;

    EMeasure measure;

    __device__ constexpr explicit BSDFQueryRecord(const Vector3f &wi) noexcept
            : wi(wi), measure(EUnknownMeasure), wo(0.f), uv(0.f), eta(1.0) {}

    __device__ constexpr BSDFQueryRecord(const Vector3f &wi,
                    const Vector3f &wo, EMeasure measure) noexcept
            : wi(wi), wo(wo), measure(measure), uv(0.f), eta(1.0) {}



};


enum class Material{
    DIFFUSE,
    MIRROR
};


class BSDF{
public:

    __device__ __host__ constexpr BSDF() noexcept
        : material(Material::DIFFUSE), albedo(1.f){

    }

    __device__ __host__ constexpr BSDF(Material mat, Color3f albedo) noexcept
        : material(mat), albedo(albedo){

    }

    __device__ constexpr Color3f sample(BSDFQueryRecord &bsdfQueryRecord, const Vector2f &sample) const noexcept{
        switch(material){
            case Material::DIFFUSE:
                if (Frame::cosTheta(bsdfQueryRecord.wi) <= 0)
                    return Color3f(0.0f);

                bsdfQueryRecord.measure = ESolidAngle;

                bsdfQueryRecord.wo = Warp::squareToCosineHemisphere(sample);

                bsdfQueryRecord.eta = 1.0f;

                //TODO add support for textures
                return albedo;

        }

    }

//private:
    Material material;
    Color3f albedo;

};
