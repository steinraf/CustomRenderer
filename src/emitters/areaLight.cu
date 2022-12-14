//
// Created by steinraf on 28/11/22.
//

#include "../acceleration/bvh.h"
#include "../shapes/triangle.h"
#include "areaLight.h"


__device__ float AreaLight::pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept {
    assert(blas);
    ShapeQueryRecord sRec{
            emitterQueryRecord.ref,
            emitterQueryRecord.p};

    //TODO take into account feedback received in exercise

    return (emitterQueryRecord.ref - emitterQueryRecord.p).squaredNorm() * blas->pdfSurface(sRec) / abs(emitterQueryRecord.n.dot(-emitterQueryRecord.wi) + EPSILON);
}

__device__ Color3f AreaLight::sample(EmitterQueryRecord &emitterQueryRecord, const Vector3f &sample) const noexcept {

    assert(isEmitter());


    ShapeQueryRecord sRec{
            emitterQueryRecord.ref};

    assert(blas);
    blas->sampleSurface(sRec, sample);


    emitterQueryRecord.p = sRec.p;
    emitterQueryRecord.wi = (emitterQueryRecord.p - emitterQueryRecord.ref).normalized();
    emitterQueryRecord.shadowRay = {
            emitterQueryRecord.ref,
            emitterQueryRecord.wi,
            EPSILON,
            (emitterQueryRecord.p - emitterQueryRecord.ref).norm() - EPSILON};


    emitterQueryRecord.n = sRec.n.normalized();
    emitterQueryRecord.pdf = pdf(emitterQueryRecord);

    return eval(emitterQueryRecord) / emitterQueryRecord.pdf;
}

//__device__ Color3f AreaLight::eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept
