//
// Created by steinraf on 23/10/22.
//

#pragma once

#include "aabb.h"

#include "../bsdf.h"
#include "../emitters/areaLight.h"
#include "../hittable.h"
#include "../utility/ray.h"

template<typename Primitive>
struct AccelerationNode {
    __device__ __host__ constexpr AccelerationNode() noexcept
        : left(nullptr), right(nullptr), primitive(nullptr), boundingBox(AABB{}), isLeaf(false){};

    __device__ __host__ constexpr AccelerationNode(AccelerationNode *left, AccelerationNode *right,
                                                   Primitive *primitive, AABB boundingBox, bool isLeaf) noexcept
        : left(left), right(right), primitive(primitive), boundingBox(boundingBox), isLeaf(isLeaf) {

        assert(left != this && right != this);

        if(!isLeaf) {
            assert(left != nullptr);
            assert(right != nullptr);
        }
    }

    [[nodiscard]] __device__ inline constexpr bool hasBoundingBox() const noexcept {
        return isLeaf || !boundingBox.isEmpty();
    }


    AccelerationNode *left;
    AccelerationNode *right;
    Primitive *primitive;
    AABB boundingBox;// Only needs to be set if not leaf
    bool isLeaf = false;
};


//Bottom Layer Acceleration structure, holds primitives
template<typename Primitive>
class BLAS {
private:
    typedef AccelerationNode<Primitive> *NodePtr;
    //    typedef Node *NodePtr;

    NodePtr root;


public:
    const float *cdf;
    size_t numPrimitives;


    AreaLight *emitter;
    BSDF *bsdf;

    Triangle *firstTriangle;


public:
    float totalArea;

    __device__ constexpr explicit BLAS(AccelerationNode<Primitive> *bvhTotalNodes, float totalArea, const float *cdf,
                                       const size_t _numPrimitives, AreaLight *emitter, BSDF *bsdf) noexcept
        : root(bvhTotalNodes), cdf(cdf), numPrimitives(_numPrimitives), bsdf(bsdf), emitter(emitter),
          totalArea(totalArea) {

        //IF THINGS GET CHANGED HERE, REMEMBER TO CHANGE IN COPY CONSTRUCTOR AS WELL
        numPrimitives = _numPrimitives;


        assert(!bvhTotalNodes[numPrimitives - 2].isLeaf);
        assert(bvhTotalNodes[numPrimitives - 1].isLeaf);

        firstTriangle = bvhTotalNodes[numPrimitives - 1].primitive;

        //        printf("Initializing blas %p with numPrimitives %lu\n", this, numPrimitives);


        //        printf("BVH LOG NUM: %f\n", log(numPrimitives));
    }


    __device__ constexpr BLAS<Primitive> &operator=(const BLAS<Primitive> &blas) noexcept {

        root = blas.root;
        cdf = blas.cdf;

        numPrimitives = blas.numPrimitives;

        //        printf("NumPrimitive after CC is %lu\n", numPrimitives);

        emitter = blas.emitter;
        bsdf = blas.bsdf;

        firstTriangle = blas.firstTriangle;

        totalArea = blas.totalArea;

        if(emitter) {
            //            printf("Setting blas %p for emitter %p for %p primitives\n", this, emitter, &numPrimitives);
            emitter->setBlas(this);
        }

        return *this;
    }


    [[nodiscard]] __device__ constexpr Primitive *sample(float &sampleValue) const noexcept {


        //        printf("Sampling blas %p with numPrimitives %lu\n", this, numPrimitives);
        //        printf("numPrimitives %p -> %lu\n", this, numPrimitives);
        //
        //        printf("Triangle area is %f\n", firstTriangle->getArea());


        //        assert(false);

        const float *begin = &cdf[0], *end = &cdf[numPrimitives];
        size_t count = end - begin, step = 0;


        //        printf("Starting: Begin is %p, end is %p\n", begin, end);

        const float *it = nullptr;
        while(count > 0) {
            //            printf("Begin is %p, end is %p\n", begin, end);
            it = begin;
            step = count / 2;
            it += step;
            if(*it < sampleValue) {
                begin = ++it;
                count -= step + 1;
            } else {
                count = step;
            }
        }

        const size_t idx = begin - &cdf[0];

        //        printf("Index is %lu\n", idx);

        assert(numPrimitives >= 2);

        assert((*(begin + 1) - *(begin)) != 0);

        if(begin == &cdf[0]) {
            sampleValue = (sampleValue) / (*(begin + 1) - *(begin));
        } else {
            sampleValue = (sampleValue - (*(begin - 1))) / (*begin - *(begin - 1));
        }

        assert(sampleValue >= 0 && sampleValue <= 1);

        return firstTriangle + idx;
    }

    //TODO make generic over primitives
    //TODO add shadow Rays
    [[nodiscard]] __device__ bool
    rayIntersect(const Ray3f &_r, Intersection &itsOut, bool isShadowRay = false) const noexcept {
        Ray3f r = _r;
        //        Intersection itsTmp;
        bool hasHit = false;

        constexpr int stackSize = 64;
        NodePtr stack[stackSize];
        int idx = 0;

        Triangle *hitTriangle = nullptr;

        NodePtr currentNode = root;

        if(!root->boundingBox.rayIntersect(r))
            return false;

        do {
            assert(idx < stackSize);

            if(currentNode->isLeaf) {
                if(currentNode->primitive->rayIntersect(r, itsOut)) {
                    if(isShadowRay)
                        return true;
                    hasHit = true;
                    r.maxDist = itsOut.t;
                    hitTriangle = currentNode->primitive;
                    itsOut.mesh = this;
                }
                currentNode = stack[--idx];
            } else {
                assert(currentNode->left && currentNode->right);
                NodePtr left = currentNode->left;
                NodePtr right = currentNode->right;
                bool continueLeft = left->boundingBox.rayIntersect(r);  // && !right->isLeaf;
                bool continueRight = right->boundingBox.rayIntersect(r);// && !right->isLeaf;

                if(!continueLeft && !continueRight) {
                    currentNode = stack[--idx];// Pop stack
                } else {
                    currentNode = continueLeft ? left : right;

                    if(continueLeft && continueRight) {
                        stack[idx++] = right;// Push stack
                    }
                }
            }
        } while(idx >= 0);

        //        itsOut = itsTmp;
        if(hasHit)
            hitTriangle->setHitInformation(r, itsOut);


        return hasHit;
    }

    [[nodiscard]] __device__ constexpr BSDF *getBSDF() const noexcept {
        return bsdf;
    }

    [[nodiscard]] __device__ constexpr AreaLight *getEmitter() const noexcept {
        assert(emitter);
        return emitter;
    }

    [[nodiscard]] __device__ constexpr float pdfSurface(const ShapeQueryRecord &sRec) const noexcept {
        return 1.f / totalArea;
    }

    __device__ constexpr void
    sampleSurface(ShapeQueryRecord &shapeQueryRecord, const Vector2f &pointSample) const noexcept {
        Vector2f s = pointSample;

        const auto triangle = sample(s[0]);

        const Vector3f bc = Warp::squareToUniformTriangle(s);


        shapeQueryRecord.p = triangle->getCoordinate(bc);
        shapeQueryRecord.n = triangle->getNormal(bc);

        shapeQueryRecord.pdf = 1.f / totalArea;
    }

    [[nodiscard]] __device__ constexpr bool isEmitter() const noexcept {
        return emitter && emitter->isEmitter();
    }
};

//TODO make logarithmic traversal as well
//Top Layer Acceleration structure, holds BLAS<Primitive>
template<typename Primitive>
class TLAS {
    //private:
public:
    BLAS<Primitive> **meshBlasArr;
    int numMeshes;

    BLAS<Primitive> **emitterBlasArr;
    int numEmitters;

public:
    __device__ constexpr TLAS(BLAS<Primitive> **meshBlasArr, int numBLAS,
                              BLAS<Primitive> **emitterBlasArr, int numEmitters) noexcept
        : meshBlasArr(meshBlasArr), numMeshes(numBLAS),
          emitterBlasArr(emitterBlasArr), numEmitters(numEmitters) {
        printf("TLAS contains %i meshes and %i emitters.\n", numMeshes, numEmitters);
    }

    [[nodiscard]] __device__ bool
    rayIntersect(const Ray3f &_r, Intersection &rec, bool isShadowRay = false) const noexcept {
        Ray3f r = _r;
        Intersection record;
        bool hasHit = false;

        //TODO make shadow ray work

        for(int i = 0; i < numMeshes; ++i) {
            if(meshBlasArr[i]->rayIntersect(r, record, isShadowRay)) {
                if(isShadowRay)
                    return true;
                hasHit = true;
                r.maxDist = record.t;
                //                assert(!record.emitter);
            }
        }

        for(int i = 0; i < numEmitters; ++i) {
            if(emitterBlasArr[i]->rayIntersect(r, record, isShadowRay)) {
                if(isShadowRay)
                    return true;
                hasHit = true;
                r.maxDist = record.t;
                //                assert(record.emitter);
            }
        }

        rec = record;
        return hasHit;
    }

    [[nodiscard]] __device__ bool rayIntersect(const Ray3f &_r) const noexcept {
        Intersection its;
        return rayIntersect(_r, its, true);
    }

    [[nodiscard]] __device__ constexpr AreaLight *getRandomEmitter(float sample) const noexcept {
        assert(numEmitters > 0);


        //TODO maybe weigh by radiance?

        const size_t idx = CustomRenderer::min(static_cast<int>(floor(numEmitters * sample)), numEmitters - 1);

        //        printf("Idx is %lu\n", idx);

        assert(emitterBlasArr[idx]->isEmitter());
        return emitterBlasArr[idx]->getEmitter();
    }
};
