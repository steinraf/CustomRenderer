//
// Created by steinraf on 28/11/22.
//

#include "areaLight.h"
#include "../acceleration/bvh.h"

__device__ Color3f AreaLight::eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept{

    return Color3f{1.f};

//    if(emitterQueryRecord.n.dot(emitterQueryRecord.wi) > 0)
//        return Color3f{0.f};

    return blas->radiance;
}
