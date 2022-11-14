//
// Created by steinraf on 23/10/22.
//

#pragma once

#include "aabb.h"

#include "../utility/ray.h"
#include "../hittable.h"

template<typename Primitive>
struct BVHNode{
    __device__ __host__ BVHNode() noexcept
        :left(nullptr), right(nullptr), primitive(nullptr), boundingBox(AABB{}), isLeaf(false){
    };

    __device__ __host__ BVHNode(BVHNode *left, BVHNode *right, Primitive *primitive, AABB boundingBox, bool isLeaf)
        : left(left), right(right), primitive(primitive), boundingBox(boundingBox), isLeaf(isLeaf){

        assert(left != this && right != this);

        if(!isLeaf){
            assert(left != nullptr);
            assert(right != nullptr);
        }


    }

    [[nodiscard]] __device__ bool hasBoundingBox() const noexcept;


    BVHNode *left;
    BVHNode *right;
    Primitive *primitive;
    AABB boundingBox; // Only needs to be set if not leaf
    bool isLeaf=false;

    bool isPointedTo=false;
};

template<typename Primitive>
__device__ bool BVHNode<Primitive>::hasBoundingBox() const noexcept{
    return isLeaf || !boundingBox.isEmpty();
}


template<typename Primitive>
class BVH{
private:
    typedef BVHNode<Primitive> Node;
    typedef Node* NodePtr;
public:
//    __device__ explicit BVH(Primitive *primitives, int numPrimitives) : primitives(primitives),
//                                                                        numPrimitives(numPrimitives){
//        for(int i = 0; i < numPrimitives; ++i)
//            printf("Initializing BVH Primitive with coordinates\n"
//                   "(%f, %f, %f) \n"
//                   "(%f, %f, %f) \n"
//                   "(%f, %f, %f) \n",
//                   primitives[i].p0[0], primitives[i].p0[1], primitives[i].p0[2],
//                   primitives[i].p1[0], primitives[i].p1[1], primitives[i].p1[2],
//                   primitives[i].p2[0], primitives[i].p2[1], primitives[i].p2[2]);




//    }

    __device__ explicit BVH(BVHNode<Primitive> *bvhTotalNodes)
        :root(bvhTotalNodes){

    }

    __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const{
        HitRecord record;
        bool hasHit = false;
        float tClosest = tMax;

        constexpr int stackSize = 256;
        NodePtr stack[stackSize];
        int idx = 0;

        NodePtr currentNode = root;

        do {
            assert(idx < stackSize);
//            printf("Visiting %p ", currentNode);
            if(currentNode->isLeaf){
//                printf(", which is leaf");
                if(currentNode->primitive->hit(r, tMin, tClosest, record)){
//                    printf(", and did hit");
                    hasHit = true;
                    tClosest = record.t;
                    rec = record;
                }
//                printf("\n");
                currentNode = stack[--idx];

            } else {
//                printf(", which is not leaf");
                NodePtr left = currentNode->left;
                NodePtr right = currentNode->right;
                float tMinimum, tMaximum; //TODO check if the intersect is actually viable
                bool continueLeft = left->boundingBox.rayIntersect(r, tMinimum, tMaximum);// && !left->isLeaf;
                bool continueRight = right->boundingBox.rayIntersect(r, tMinimum, tMaximum);// && !right->isLeaf;

//                printf(" left: %d, right: %d\n", continueLeft, continueRight);

//                printf("Ray: (%f, %f, %f) -> (%f, %f, %f)\n",
//                       r.getOrigin()[0], r.getOrigin()[1], r.getOrigin()[2],
//                       r.getDirection()[0], r.getDirection()[1], r.getDirection()[2]
//                       );

//                printf("Left BB: (%f, %f, %f) -> (%f, %f, %f)\n",
//                       left->boundingBox.min[0], left->boundingBox.min[1], left->boundingBox.min[2],
//                       left->boundingBox.max[0], left->boundingBox.max[1], left->boundingBox.max[2]);
//
//                printf("Right BB: (%f, %f, %f) -> (%f, %f, %f)\n",
//                       right->boundingBox.min[0], right->boundingBox.min[1], right->boundingBox.min[2],
//                       right->boundingBox.max[0], right->boundingBox.max[1], right->boundingBox.max[2]);

                if(!continueLeft && !continueRight){
                    currentNode = stack[--idx];// Pop stack
                } else {
                    currentNode = continueLeft ? left : right;

                    if (continueLeft && continueRight){
                        stack[idx++] = right;
                    }
                }

            }
        } while(idx > 0);

//        printf("Ended hit check %d\n", hasHit);


        return hasHit;
    }

//    __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const{
//        HitRecord record;
//        bool hasHit = false;
//        float tClosest = tMax;
//
//
//        for(int i = 0; i < numPrimitives; ++i){
////            printf("Trying to hit Primitive %i with coordinates\n"
////                   "(%f, %f, %f) \n"
////                   "(%f, %f, %f) \n"
////                   "(%f, %f, %f) \n", i,
////                   primitives[i].p0[0], primitives[i].p0[1], primitives[i].p0[2],
////                   primitives[i].p1[0], primitives[i].p1[1], primitives[i].p1[2],
////                   primitives[i].p2[0], primitives[i].p2[1], primitives[i].p2[2]);
//
//            if(primitives[i].hit(r, tMin, tClosest, record)){
//                hasHit = true;
//                tClosest = record.t;
//                rec = record;
//            }
//        }
//
//        return hasHit;
//    }

private:
    NodePtr root;

};



