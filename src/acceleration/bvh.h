//
// Created by steinraf on 23/10/22.
//

#pragma once

#include "aabb.h"

#include "../utility/ray.h"
#include "../hittable.h"

template<typename Primitive>
struct BVHNode{
    BVHNode() noexcept = default;

    __device__ __host__ BVHNode(BVHNode *left, BVHNode *right, Primitive *primitive, AABB boundingBox, bool isLeaf)
        : left(left), right(right), primitive(primitive), boundingBox(boundingBox), isLeaf(isLeaf){
        assert(left != this && right != this);
    }


    BVHNode *left;
    BVHNode *right;
    Primitive *primitive;
    AABB boundingBox; // Only needs to be set if not leaf
    bool isLeaf;
};


template<typename Primitive>
class BVH{
public:
    __device__ explicit BVH(Primitive *primitives, int numPrimitives) : primitives(primitives),
                                                                        numPrimitives(numPrimitives){
//        for(int i = 0; i < numPrimitives; ++i)
//            printf("Initializing BVH Primitive with coordinates\n"
//                   "(%f, %f, %f) \n"
//                   "(%f, %f, %f) \n"
//                   "(%f, %f, %f) \n",
//                   primitives[i].p0[0], primitives[i].p0[1], primitives[i].p0[2],
//                   primitives[i].p1[0], primitives[i].p1[1], primitives[i].p1[2],
//                   primitives[i].p2[0], primitives[i].p2[1], primitives[i].p2[2]);




    }

//    __device__ bool hit(const primitives[i].Ray &r, float tMin, float tMax, HitRecord &rec) const;

    __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const{
        HitRecord record;
        bool hasHit = false;
        float tClosest = tMax;


        for(int i = 0; i < numPrimitives; ++i){
//            printf("Trying to hit Primitive %i with coordinates\n"
//                   "(%f, %f, %f) \n"
//                   "(%f, %f, %f) \n"
//                   "(%f, %f, %f) \n", i,
//                   primitives[i].p0[0], primitives[i].p0[1], primitives[i].p0[2],
//                   primitives[i].p1[0], primitives[i].p1[1], primitives[i].p1[2],
//                   primitives[i].p2[0], primitives[i].p2[1], primitives[i].p2[2]);

            if(primitives[i].hit(r, tMin, tClosest, record)){
                hasHit = true;
                tClosest = record.t;
                rec = record;
            }
        }

        return hasHit;
    }

private:
    Primitive *primitives;
    int numPrimitives;
//    BVHNode<Primitive> *root;
};



