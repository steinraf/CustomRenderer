//
// Created by steinraf on 23/10/22.
//

#pragma once

#include "aabb.h"

#include "../utility/ray.h"
#include "../hittable.h"
#include "../emitters/areaLight.h"

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

    const float *cdf;
    size_t numPrimitives;




public:

    AreaLight *emitter;


    __device__ constexpr explicit BLAS(AccelerationNode<Primitive> *bvhTotalNodes, const float *cdf, size_t numPrimitives, AreaLight *emitter) noexcept
            : root(bvhTotalNodes), cdf(cdf), numPrimitives(numPrimitives), emitter(emitter){

        if(emitter)
            emitter->setBlas(this);
//        emitter = new AreaLight(this, radiance);

//        if(radiance.isZero()){
//            emitter = nullptr;
//        }else{
//            printf("Initialized emitter BLAS with radiance (%f, %f, %f)\n", radiance[0], radiance[1], radiance[2]);
//            emitter = new AreaLight(this, radiance);
//        }

//        printf("BVH LOG NUM: %f\n", log(numPrimitives));
    }

//TODO make generic over primitives
//TODO add shadow Rays
    [[nodiscard]] __device__ bool rayIntersect(const Ray &_r, Intersection &itsOut) const noexcept{
        Ray r = _r;
//        Intersection itsTmp;
        bool hasHit = false;

        constexpr int stackSize = 64;
        NodePtr stack[stackSize];
        int idx = 0;

        NodePtr currentNode = root;

        if(!root->boundingBox.rayIntersect(r))
            return false;

        do{
            assert(idx < stackSize);

            if(currentNode->isLeaf){
                if(currentNode->primitive->rayIntersect(r, itsOut)){
                    hasHit = true;
                    r.maxDist = itsOut.t;
                    itsOut.triangle = currentNode->primitive;

                    itsOut.emitter = this->emitter;
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

//        itsOut = itsTmp;
        if(hasHit)
            itsOut.triangle->setHitInformation(r, itsOut);



        return hasHit;
    }

    [[nodiscard]] __device__ constexpr bool isEmitter() const noexcept{
        return emitter && emitter->isEmitter();
    }

    [[nodiscard]] __device__ Triangle *sample() const noexcept{
        return nullptr;
    }

};

//TODO make logarithmic traversal as well
//Top Layer Acceleration structure, holds BLAS<Primitive>
template<typename Primitive>
class TLAS{
//private:
public:
    BLAS<Primitive> **meshBlasArr;
    int numMeshes;

    BLAS<Primitive> **emitterBlasArr;
    Vector3f **emitterRadiance;
    int numEmitters;

public:
    __device__ constexpr TLAS(BLAS<Primitive> **meshBlasArr, int numBLAS,
                              BLAS<Primitive> **emitterBlasArr, Vector3f **emitterRadiance, int numEmitters) noexcept
        : meshBlasArr(meshBlasArr), numMeshes(numBLAS),
        emitterBlasArr(emitterBlasArr), emitterRadiance(emitterRadiance), numEmitters(numEmitters){
        printf("TLAS contains %i meshes and %i emitters.\n", numMeshes, numEmitters);

    }

    [[nodiscard]] __device__ bool rayIntersect(const Ray &_r, Intersection &rec) const noexcept{
        Ray r = _r;
        Intersection record;
        bool hasHit = false;

        for(int i = 0; i < numMeshes; ++i){
            if(meshBlasArr[i]->rayIntersect(r, record)){
                hasHit = true;
                r.maxDist = record.t;
//                assert(!record.emitter);
            }
        }

        for(int i = 0; i < numEmitters; ++i){
            if(emitterBlasArr[i]->rayIntersect(r, record)){
                hasHit = true;
                r.maxDist = record.t;
//                assert(record.emitter);
            }
        }

        rec = record;
        return hasHit;

    }


};



