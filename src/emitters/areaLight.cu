//
// Created by steinraf on 28/11/22.
//

#include "areaLight.h"
#include "../acceleration/bvh.h"

__host__ __device__ AreaLight::AreaLight(const Color3f &radiance) noexcept
    :radiance(radiance), blas(nullptr){
//        assert(blas);
//        printf("Initialized Area Light with radiance (%f, %f, %f)\n", blas->radiance[0], blas->radiance[1], blas->radiance[2]);
//        printf("THIS: %p\n", this);
}

__device__ Color3f AreaLight::eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept{

    assert(isEmitter());

    assert(blas);

//    assert(blas->isEmitter());
//    return Color3f{0.f};

//    return Color3f{1.f};
//
    if(emitterQueryRecord.n.dot(emitterQueryRecord.wi) > 0)
        return Color3f{0.f};


//    printf("Result was %f\n", emitterQueryRecord.n.dot(emitterQueryRecord.wi));


//    if(isEmitter()){
//        printf("THIS: %p\n", this);
//        printf("Radiance is (%e, %e, %e)\n", radiance[0], radiance[1],radiance[2]);
//    }




    return radiance;
}
