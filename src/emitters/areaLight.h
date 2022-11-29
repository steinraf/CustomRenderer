//
// Created by steinraf on 28/11/22.
//

#pragma once

//#include "../acceleration/bvh.h"
#include "../utility/vector.h"
#include "../utility/ray.h"
#include "../shapes/triangle.h"

struct EmitterQueryRecord{
    Vector3f ref;
    Vector3f p;
    Vector3f n;
    Vector3f wi;
    float pdf;
    Ray shadowRay;

    __device__ constexpr explicit EmitterQueryRecord(const Vector3f &ref) noexcept
        :ref(ref), p(), n(), wi(), pdf(), shadowRay(){

    }

    __device__ constexpr EmitterQueryRecord(const Vector3f &ref, const Vector3f &p, const Vector3f &n) noexcept
        :ref(ref), p(p), n(n), wi((p-ref).normalized()), pdf(), shadowRay(){

    }
};

template <typename Primitive>
class BLAS;

class AreaLight{
public:
    explicit __host__ __device__ AreaLight(const Color3f &radiance) noexcept;

    AreaLight() = default;

    __device__ void constexpr setBlas(const class BLAS<Triangle> *newBlas){
        blas = newBlas;
    }


    [[nodiscard]] __device__ Color3f eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept;

    [[nodiscard]] __device__ constexpr bool isEmitter() const noexcept{
        return !radiance.isZero();
    }

//    [[nodiscard]] __device__ Color3f sample(EmitterQueryRecord &emitterQueryRecord, const Vector2f &sample) const noexcept{
//
//    }

//private:
    Color3f radiance;
    const BLAS<Triangle> *blas;
};