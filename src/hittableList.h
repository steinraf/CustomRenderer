////
//// Created by steinraf on 21/08/22.
////
//
//#pragma once
//
//#include "hittable.h"
//#include <cuda/std/cassert>
//
//#include <thrust/device_vector.h>
//
//class HittableList/* : public Hittable */{
//public:
//    __device__ explicit HittableList(Triangle *triangles, size_t size = 0);
//
//    __device__ void add(Triangle *triangles);
//
//    __device__ bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const override;
//
//
//    size_t maxSize;
//
//private:
//    Triangle *triangles;
//
////    thrust::device_vector<Hittable> hittableList;
//
//    size_t currentSize = 0;
//
//};
//
//
