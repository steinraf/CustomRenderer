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
            : wi(wi), wo(0.f), uv(0.f), eta(1.0), measure(EUnknownMeasure) {}

    __device__ constexpr BSDFQueryRecord(const Vector3f &wi,
                    const Vector3f &wo, EMeasure measure) noexcept
            : wi(wi), wo(wo), uv(0.f), eta(1.0), measure(measure) {}



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

    [[nodiscard]] __device__ constexpr Color3f sample(BSDFQueryRecord &bsdfQueryRecord, const Vector2f &sample) const noexcept{
        switch(material){
            case Material::DIFFUSE:
                if (Frame::cosTheta(bsdfQueryRecord.wi) <= 0)
                    return Color3f(0.0f);

                bsdfQueryRecord.measure = ESolidAngle;

                bsdfQueryRecord.wo = Warp::squareToCosineHemisphere(sample);

                bsdfQueryRecord.eta = 1.0f;

                //TODO add support for textures
                return albedo;
            case Material::MIRROR:
                assert(false);

        }

    }

    [[nodiscard]] __device__ constexpr float pdf(const BSDFQueryRecord &bsdfQueryRecord) const noexcept{
        switch(material){
            case Material::DIFFUSE:
                if (bsdfQueryRecord.measure != ESolidAngle
                    || Frame::cosTheta(bsdfQueryRecord.wi) <= 0
                    || Frame::cosTheta(bsdfQueryRecord.wo) <= 0)
                    return 0.0f;

                return M_1_PIf * Frame::cosTheta(bsdfQueryRecord.wo);
            case Material::MIRROR:
                assert(false);
        }
    }

    [[nodiscard]] __device__ constexpr Color3f eval(const BSDFQueryRecord &bsdfQueryRecord) const noexcept{
        switch(material){
            case Material::DIFFUSE:
                if (bsdfQueryRecord.measure != ESolidAngle
                    || Frame::cosTheta(bsdfQueryRecord.wi) <= 0
                    || Frame::cosTheta(bsdfQueryRecord.wo) <= 0)
                    return Color3f(0.0f);

                return albedo * M_1_PIf;
            case Material::MIRROR:
                assert(false);
        }

    }

//private:
    Material material;
    Color3f albedo;

};
