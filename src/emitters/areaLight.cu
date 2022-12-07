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

__device__ Color3f AreaLight::sample(EmitterQueryRecord &emitterQueryRecord, const Vector2f &sample) const noexcept {

    //    printf("AreaEmitter BLAS NumPrimitives is %lu\n", blas->numPrimitives);


    assert(isEmitter());


    ShapeQueryRecord sRec{
            emitterQueryRecord.ref};

    assert(blas);
    blas->sampleSurface(sRec, sample);

    //    printf("Sampled surface\n");

    //    assert(emitterQueryRecord.p != emitterQueryRecord.ref);

    //    printf("p(%f, %f, %f), ref(%f, %f, %f)\n",
    //           sRec.p[0], sRec.p[1], sRec.p[2],
    //           emitterQueryRecord.ref[0], emitterQueryRecord.ref[1], emitterQueryRecord.ref[2]
    //    );


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
