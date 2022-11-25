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

    NodePtr root;

    float *cdf;
    size_t numPrimitives;

public:

    __device__ constexpr explicit BLAS(AccelerationNode<Primitive> *bvhTotalNodes, float *cdf, size_t numPrimitives) noexcept
            : root(bvhTotalNodes), cdf(cdf), numPrimitives(numPrimitives){

    }

//TODO make generic over primitives
//TODO add shadow Rays
    [[nodiscard]] __device__ bool rayIntersect(const Ray &_r, Intersection &itsOut) const noexcept{
        Ray r = _r;
        Intersection itsTmp;
        bool hasHit = false;

        constexpr int stackSize = 256;
        NodePtr stack[stackSize];
        int idx = 0;

        NodePtr currentNode = root;

        if(!root->boundingBox.rayIntersect(r))
            return false;

        do{
            assert(idx < stackSize);

            if(currentNode->isLeaf){
                if(currentNode->primitive->rayIntersect(r, itsTmp)){
                    hasHit = true;
                    r.maxDist = itsTmp.t;
                    itsOut = itsTmp;
                    itsTmp.triangle = currentNode->primitive;
                }
                currentNode = stack[--idx];
            }else{
                assert(currentNode->left && currentNode->right);
                NodePtr left = currentNode->left;
                NodePtr right = currentNode->right;
                bool continueLeft = left->boundingBox.rayIntersect(r);// && !right->isLeaf;
                bool continueRight = right->boundingBox.rayIntersect(r);// && !right->isLeaf;

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
            itsTmp.triangle->setHitInformation(r, itsOut);


        return hasHit;
    }



};

//TODO make logarithmic traversal as well
//Top Layer Acceleration structure, holds BLAS<Primitive>
template<typename Primitive>
class TLAS{
private:
    BLAS<Primitive> **blasArr;
    int numBLAS;
public:
    __device__ constexpr TLAS(BLAS<Primitive> **blasArr, int numBLAS) noexcept
        : blasArr(blasArr), numBLAS(numBLAS){


    }

    [[nodiscard]] __device__ bool rayIntersect(const Ray &_r, Intersection &rec) const noexcept{
        Ray r = _r;
        Intersection record;
        bool hasHit = false;

        for(int i = 0; i < numBLAS; ++i){
            if(blasArr[i]->rayIntersect(r, record)){
                hasHit = true;
                r.maxDist = record.t;
            }
        }

        rec = record;
        return hasHit;

    }


};



