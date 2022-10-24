////
//// Created by steinraf on 21/08/22.
////
//
//#include "hittableList.h"
//
//
//__device__ HittableList::HittableList(Hittable **hittables, size_t size) : hittables(hittables), maxSize(size) {
//    assert(hittables && "Tried to initialize HittableList with null pointer");
//}
//
//__device__ void HittableList::add(Hittable *hittable) {
//    assert(currentSize < maxSize && "Trying to add Hittable to full HittableList");
//    assert(hittable && "Found null pointer while adding Hittable to HittableList");
//    hittables[currentSize++] = hittable;
//}
//
//__device__ bool HittableList::hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const {
//
//    HitRecord record;
//    bool hasHit = false;
//    float tClosest = tMax;
//
//
//    for (int i = 0; i < currentSize; ++i) {
//        if (hittables[i]->hit(r, tMin, tClosest, record)) {
//            hasHit = true;
//            tClosest = record.t;
//            rec = record;
//        }
//    }
//
//    return hasHit;
//}
//
//
