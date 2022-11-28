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
    explicit __device__ AreaLight(const class BLAS<Triangle> *blas)
        :blas(blas){
        assert(blas);

    }

    [[nodiscard]] __device__ Color3f eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept;


//    [[nodiscard]] __device__ Color3f sample(EmitterQueryRecord &emitterQueryRecord, const Vector2f &sample) const noexcept{
//
//    }

private:
    const BLAS<Triangle> *blas;
};