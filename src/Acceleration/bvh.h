//
// Created by steinraf on 23/10/22.
//

#pragma once


#include "../utility/ray.h"
#include "../hittable.h"

template<typename Primitive>
struct BVHNode{
    __device__ explicit BVHNode(const Primitive &primitive) noexcept
            : left(nullptr), right(nullptr),
              countLeft(0), countRight(0),
              primitive(primitive){

    }

    BVHNode *left;
    BVHNode *right;
    int countLeft;
    int countRight;
    const Primitive &primitive;
};


template<typename Primitive>
class BVH{
public:
    __device__ explicit BVH(Primitive *primitives, int numPrimitives);

    __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const;

private:
    Primitive *primitives;
    int numPrimitives;
//    BVHNode<Primitive> *root;
};



