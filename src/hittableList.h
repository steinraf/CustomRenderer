//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "hittable.h"
#include "cuda/std/cassert"

class HittableList : public Hittable {
public:
    __device__ HittableList(Hittable **hittables, size_t size = 0);

    __device__ void add(Hittable *hittable);

    __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const override;


    size_t maxSize;

private:
    Hittable **hittables;

    size_t currentSize = 0;

};


