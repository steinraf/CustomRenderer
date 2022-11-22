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
public:

    __device__ constexpr explicit BLAS(AccelerationNode<Primitive> *bvhTotalNodes) noexcept
            : root(bvhTotalNodes){

//        printf("BLAS_CONSTRUCTOR: Total bounding box is (%f, %f, %f) -> (%f, %f, %f)\n",
//               root->boundingBox.min[0], root->boundingBox.min[1], root->boundingBox.min[2],
//               root->boundingBox.max[0], root->boundingBox.max[1], root->boundingBox.max[2]);

    }

    __device__ AABB getBoundingBox() const{
        return root->boundingBox;
    }

//TODO make generic over primitives
//TODO add shadow Rays
    [[nodiscard]] __device__ bool hit(const Ray &_r, HitRecord &rec) const noexcept{

        Ray ray = _r;

        HitRecord record;
        bool hasHit = false;

        constexpr int stackSize = 256;
        NodePtr stack[stackSize];
        int idx = 0;

        NodePtr currentNode = root;


        if(!root->boundingBox.rayIntersect(ray))
            return false;

//        printf("Hit BB\n");

        do{
            assert(idx < stackSize);

            if(currentNode->isLeaf){
                if(currentNode->primitive->hit(ray, record)){
                    hasHit = true;
                    ray.maxDist = record.t;
                    rec = record;
                    record.triangle = currentNode->primitive;
//                    printf("Hit BLAS\n");
                }
                currentNode = stack[--idx];
            }else{
                assert(currentNode->left && currentNode->right);
                NodePtr left = currentNode->left;
                NodePtr right = currentNode->right;
                bool continueLeft = left->boundingBox.rayIntersect(ray);
                bool continueRight = right->boundingBox.rayIntersect(ray);

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
            record.triangle->setHitInformation(ray, rec);


        return hasHit;
    }


};

//Top Layer Acceleration Structure, holds BLAS'
template<typename Primitive>
struct TLAS{
private:
    typedef AccelerationNode<BLAS<Primitive>> Node;
    typedef Node *NodePtr;

    NodePtr root;

    BLAS<Primitive> **blasArr;
    int numBlas;
public:

    __device__ constexpr explicit TLAS(BLAS<Primitive> **blasArr, int numBlas) noexcept
        :blasArr(blasArr), numBlas(numBlas){

    }

    __device__ constexpr explicit TLAS(AccelerationNode<BLAS<Primitive>> *bvhTotalNodes) noexcept
            : root(bvhTotalNodes){

    }

    //TODO make generic over primitives
    //TODO add shadow Rays
    [[nodiscard]] __device__ bool hit(const Ray &_r, HitRecord &rec) const noexcept{



        Ray ray = _r;
        HitRecord record;

        bool hasHit = false;


        for(int i = 0; i < numBlas; ++i){

//            const AABB &totalBoundingBox = blasArr[i].getBoundingBox();

//            printf("Total bounding box %i is (%f, %f, %f) -> (%f, %f, %f)\n", i,
//                   totalBoundingBox.min[0], totalBoundingBox.min[1], totalBoundingBox.min[2],
//                   totalBoundingBox.max[0], totalBoundingBox.max[1], totalBoundingBox.max[2]);
            if(blasArr[0]->hit(ray, record)){
                hasHit = true;
                ray.maxDist = record.t;
                //TODO check whether Triangle is set by BLAS
                rec = record;
//                printf("Hit TLAS\n");
            }
        }

        return hasHit;




//        Ray ray = _r;
//
//        HitRecord record;
//        bool hasHit = false;
//
//        //TODO make log(numTriangles)
//        constexpr int stackSize = 32;
//        NodePtr stack[stackSize];
//        int idx = 0;
//
//        NodePtr currentNode = root;
//
//
//        if(!root->boundingBox.rayIntersect(ray))
//            return false;
//
//        do{
//            assert(idx < stackSize);
//
//            if(currentNode->isLeaf){
//                if(currentNode->primitive->hit(ray, record)){
//                    hasHit = true;
//                    ray.maxDist = record.t;
//                    //TODO check whether Triangle is set by BLAS
//                    rec = record;
//                }
//                currentNode = stack[--idx];
//            }else{
//                assert(currentNode->left && currentNode->right);
//                NodePtr left = currentNode->left;
//                NodePtr right = currentNode->right;
//                assert(left);
//                assert(right);
//                bool continueLeft = left->boundingBox.rayIntersect(ray);
//                bool continueRight = right->boundingBox.rayIntersect(ray);
//
//                if(!continueLeft && !continueRight){
//                    currentNode = stack[--idx];// Pop stack
//                }else{
//                    currentNode = continueLeft ? left : right;
//
//                    if(continueLeft && continueRight){
//                        stack[idx++] = right;// Push stack
//                    }
//                }
//            }
//        }while(idx >= 0);
//
//        if(hasHit)
//            record.triangle->setHitInformation(ray, rec);
//
//
//        return hasHit;
    }


};


