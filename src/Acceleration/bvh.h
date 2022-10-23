//
// Created by steinraf on 23/10/22.
//

#pragma once



template<typename Primitive>
struct BVHNode{
    __device__ explicit BVHNode(const Primitive &primitive) noexcept
        :left(nullptr), right(nullptr),
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
    explicit BVH(Primitive *primitives, int numPrimitives);

private:
    Primitive *primitives;
    BVHNode<Primitive> *root;
};



