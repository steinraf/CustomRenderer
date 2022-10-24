//
// Created by steinraf on 23/10/22.
//

#include "bvh.h"


template<typename Primitive>
__device__
BVH<Primitive>::BVH(Primitive *primitives, int numPrimitives):primitives(primitives), numPrimitives(numPrimitives){

}

template<typename Primitive>
__device__ bool BVH<Primitive>::hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const{
    HitRecord record;
    bool hasHit = false;
    float tClosest = tMax;


    for(int i = 0; i < numPrimitives; ++i){
        if(primitives[i]->hit(r, tMin, tClosest, record)){
            hasHit = true;
            tClosest = record.t;
            rec = record;
        }
    }

    return hasHit;
}

