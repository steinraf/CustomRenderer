//
// Created by steinraf on 28/11/22.
//

#pragma once

#include "../utility/vector.h"
#include "../utility/ray.h"
#include "../shapes/triangle.h"
#include "../acceleration/bvh.h"

struct EmitterQueryRecord{
    Vector3f ref;
    Vector3f p;
    Vector3f n;
    Vector3f wi;
    float pdf;
    Ray shadowRay;

    EmitterQueryRecord(const Vector3f &ref)
        :ref(ref){

    }

    EmitterQueryRecord(const Vector3f &ref, const Vector3f &p, const Vector3f &n)
        :ref(ref), p(p), n(n), wi((p-ref).normalized()){

    }
};


class AreaLight{
public:
    explicit AreaLight(const BLAS<Triangle> *blas)
        :blas(blas){


    }

    [[nodiscard]] Color3f eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept{

        if(emitterQueryRecord.n.dot(emitterQueryRecord.wi) > 0)
            return Color3f{0.f};

        return blas->radiance;
    }

    [[nodiscard]] Color3f sample(EmitterQueryRecord &emitterQueryRecord, const Vector2f &sample) const noexcept{

    }

private:
    BLAS<Triangle> *blas;
};