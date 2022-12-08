//
// Created by steinraf on 07/12/22.
//

#include "../shapes/triangle.h"
#include "../utility/ray.h"
#include "../utility/vector.h"
#include "areaLight.h"

#pragma once


class EnvironmentEmitter {

private:
    Texture texture;

public:
    explicit __host__ __device__ constexpr EnvironmentEmitter(Texture texture) noexcept
        : texture(texture) {

    }
    
    [[nodiscard]] __device__ constexpr Color3f eval(const Ray3f &ray) const noexcept {

        assert(ray.getDirection().norm() != 0.f);
        const Vector3f dir = ray.getDirection().normalized();

//        const float u = CustomRenderer::clamp(acos(dir[2]) * M_1_PIf, -1.f, 1.f);
//        const float v = atan2(dir[1], dir[0]) * 0.5f * M_1_PIf;

        const float u = atan2(dir[0], -dir[2]) * 0.5f * M_1_PIf;
        const float v = CustomRenderer::clamp(acos(-dir[1]) * M_1_PIf, -1.f, 1.f);
//        printf("UV: (%f, %f)\n", u, v);
        if(!::isfinite(u) || !::isfinite(v)){
#ifndef NDEBUG
            printf("UV infinite for dir (%f, %f, %f)\n", ray.getDirection()[0], ray.getDirection()[1], ray.getDirection()[2]);
#endif
            return Vector3f{0.f};
        }
        assert(isfinite(u) && ::isfinite(v));
        return texture.eval(Vector2f{(u < 0) ? (u + 1) : u, v});
    }

    [[nodiscard]] __device__ constexpr float pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept{
        return 1.f;
    }


    [[nodiscard]] __device__ Color3f constexpr sample(EmitterQueryRecord &emitterQueryRecord, const Vector2f &sample) const noexcept{

        //TODO importance sample as emitter
//        Vector3f ref;
//        Vector3f p;
//        Vector3f n;
//        Vector3f wi;
//        float pdf;
//        Ray3f shadowRay;

        //sampleSurface

        const Vector3f dirSample = Warp::squareToUniformSphere(sample);

        Vector2f s = sample;

        assert(texture.deviceCDF);
        const size_t idx = Warp::sampleCDF(s[0], &texture.deviceCDF[0], &texture.deviceCDF[texture.width * texture.height - 1]);

        const float u = idx % texture.width;
        const float v = idx / texture.width;

        return texture.eval(Vector2f{u, v});

//        emitterQueryRecord.p = sRec.p;
//        emitterQueryRecord.wi = (emitterQueryRecord.p - emitterQueryRecord.ref).normalized();
//        emitterQueryRecord.shadowRay = {
//                emitterQueryRecord.ref,
//                emitterQueryRecord.wi,
//                EPSILON,
//                (emitterQueryRecord.p - emitterQueryRecord.ref).norm() - EPSILON};
//
//
//        emitterQueryRecord.n = sRec.n.normalized();
//        emitterQueryRecord.pdf = pdf(emitterQueryRecord);
//
//        return eval(emitterQueryRecord) / emitterQueryRecord.pdf;

        return Color3f{0.f};
    }

};
