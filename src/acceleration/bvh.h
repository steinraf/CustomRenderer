//
// Created by steinraf on 23/10/22.
//

#pragma once

#include "aabb.h"

#include "../utility/ray.h"
#include "../hittable.h"

template<typename Primitive>
struct AccelerationNode{
    __device__ __host__ constexpr AccelerationNode() noexcept
            : left(nullptr), right(nullptr), primitive(nullptr), boundingBox(AABB{}), isLeaf(false){
    };

    __device__ __host__ constexpr AccelerationNode(AccelerationNode *left, AccelerationNode *right, Primitive *primitive, AABB boundingBox, bool isLeaf) noexcept
            : left(left), right(right), primitive(primitive), boundingBox(boundingBox), isLeaf(isLeaf){

        assert(left != this && right != this);

        if(!isLeaf){
            assert(left != nullptr);
            assert(right != nullptr);
        }


    }

    [[nodiscard]] __device__ inline constexpr bool hasBoundingBox() const noexcept{
        return isLeaf || !boundingBox.isEmpty();
    }


    AccelerationNode *left;
    AccelerationNode *right;
    Primitive *primitive;
    AABB boundingBox; // Only needs to be set if not leaf
    bool isLeaf = false;
};

//Bottom Layer Acceleration structure, holds primitives
template<typename Primitive>
class BLAS{
private:
    typedef AccelerationNode<Primitive> Node;
    typedef Node *NodePtr;
public:

    __device__ constexpr explicit BLAS(AccelerationNode<Primitive> *bvhTotalNodes) noexcept
            : root(bvhTotalNodes){

    }

//TODO make generic over primitives
//TODO add shadow Rays
    [[nodiscard]] __device__ bool hit(const Ray &r, HitRecord &rec) const noexcept{
        HitRecord record;
        bool hasHit = false;
        float tClosest = r.maxDist;

        constexpr int stackSize = 256;
        NodePtr stack[stackSize];
        int idx = 0;

        NodePtr currentNode = root;

        float tMinimum, tMaximum; //TODO check if the intersect is actually viable

        if(!root->boundingBox.rayIntersect(r, tMinimum, tMaximum))
            return false;

        do{
            assert(idx < stackSize);

            if(currentNode->isLeaf){
                if(currentNode->primitive->hit(r, r.minDist, tClosest, record)){
                    hasHit = true;
                    tClosest = record.t;
                    rec = record;
                    record.triangle = currentNode->primitive;
                }
                currentNode = stack[--idx];
            }else{
                assert(currentNode->left && currentNode->right);
                NodePtr left = currentNode->left;
                NodePtr right = currentNode->right;
                bool continueLeft = left->boundingBox.rayIntersect(r, tMinimum, tMaximum);// && !left->isLeaf;
                bool continueRight = right->boundingBox.rayIntersect(r, tMinimum, tMaximum);// && !left->isLeaf;

                if(!continueLeft && !continueRight){
                    currentNode = stack[--idx];// Pop stack
                }else{
                    currentNode = continueLeft ? left : right;

                    if(continueLeft && continueRight){
                        stack[idx++] = right;// Push stack
                    }
                }
            }
        }while(idx >= 0);

        if(hasHit)
            record.triangle->setHitInformation(r, rec);


        return hasHit;
    }

private:
    NodePtr root;

};



