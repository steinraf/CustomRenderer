//
// Created by steinraf on 24/10/22.
//

#include "aabb.h"

__host__ __device__ AABB::AABB(const Vector3f &a, const Vector3f &b, const Vector3f &c){
    min = {
            thrust::min(a[0], thrust::min(b[0], c[0])),
            thrust::min(a[1], thrust::min(b[1], c[1])),
            thrust::min(a[2], thrust::min(b[2], c[2])),
    };
    max = {
            thrust::max(a[0], thrust::max(b[0], c[0])),
            thrust::max(a[1], thrust::max(b[1], c[1])),
            thrust::max(a[2], thrust::max(b[2], c[2])),
    };

}

__device__ bool AABB::rayIntersect(const Ray &ray, float &nearT, float &farT) const{
    nearT = -cuda::std::numeric_limits<float>::infinity();
    farT = cuda::std::numeric_limits<float>::infinity();

    for(int i = 0; i < 3; i++){
        float origin = ray.getOrigin()[i];
        float minVal = min[i], maxVal = max[i];

        if(ray.getDirection()[i] == 0){
            if(origin < minVal || origin > maxVal)
                return false;
        }else{
            float t1 = (minVal - origin) / ray.getDirection()[i];
            float t2 = (maxVal - origin) / ray.getDirection()[i];

            if(t1 > t2){
                cuda::std::swap(t1, t2);
            }

            nearT = thrust::max(t1, nearT);
            farT = thrust::min(t2, farT);

            if(nearT > farT)
                return false;
        }
    }

    return true;
}

__device__ AABB AABB::operator+(const AABB &other) const{
    return {
            Vector3f{-FLT_EPSILON} + Vector3f{thrust::min(min[0], other.min[0]), thrust::min(min[1], other.min[1]),
                                              thrust::min(min[2], other.min[2])},
            Vector3f{FLT_EPSILON} + Vector3f{thrust::max(max[0], other.max[0]), thrust::max(max[1], other.max[1]),
                                             thrust::max(max[2], other.max[2])},
    };
}
