//
// Created by steinraf on 28/11/22.
//

#pragma once

#include "../shapes/triangle.h"
#include "../utility/ray.h"
#include "../utility/vector.h"

struct EmitterQueryRecord {
    Vector3f ref;
    Vector3f p;
    Vector3f n;
    Vector3f wi;
    float pdf;
    Ray3f shadowRay;

    __device__ constexpr explicit EmitterQueryRecord(const Vector3f &ref) noexcept
        : ref(ref), p(), n(), wi(), pdf(), shadowRay() {
    }

    __device__ constexpr EmitterQueryRecord(const Vector3f &ref, const Vector3f &p, const Vector3f &n) noexcept
        : ref(ref), p(p), n(n), wi((p - ref).normalized()), pdf(), shadowRay() {
    }
};


class AreaLight {
public:
    explicit __host__ __device__ constexpr AreaLight(const Color3f &radiance) noexcept
        : radiance(radiance), blas(nullptr) {
        //        assert(blas);
        //        printf("Initialized Area Light with radiance (%f, %f, %f)\n", blas->radiance[0], blas->radiance[1], blas->radiance[2]);
        //        printf("THIS: %p\n", this);
    }

    AreaLight() = default;

    __device__ void constexpr setBlas(const class BLAS *newBlas) {
        blas = newBlas;
    }


    [[nodiscard]] __device__ constexpr Color3f eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept {
        //        assert(blas);
        if(emitterQueryRecord.n.dot(emitterQueryRecord.wi) >= 0)
            return Color3f{0.f};

        return radiance;
    }

    [[nodiscard]] __device__ float pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept;


    [[nodiscard]] __device__ Color3f
    sample(EmitterQueryRecord &emitterQueryRecord, const Vector3f &sample) const noexcept;


    [[nodiscard]] __device__ constexpr bool isEmitter() const noexcept {
        return !radiance.isZero();
    }

    //    [[nodiscard]] __device__ Color3f sample(EmitterQueryRecord &emitterQueryRecord, const Vector2f &sample) const noexcept{
    //
    //    }

    //private:
    Color3f radiance;
    const BLAS *blas;
};