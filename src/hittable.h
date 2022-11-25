//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "utility/vector.h"
#include "utility/ray.h"



struct Intersection{
    Vector3f p;
    Vector3f n;

    class Triangle const *triangle;

    Vector2f uv;

    float t;
    bool frontFace;

//    __device__ inline void constexpr setFaceNormal(const Ray &r, const Vector3f &outwardNormal) noexcept{
//        frontFace = r.getDirection().dot(outwardNormal) < 0;
//        n = frontFace ? outwardNormal : -outwardNormal;
//    }



    //TODO rayIntersect records need to set in the shape class, and not just with ray.at(t) because of numerical
    //instabilities

    //TODO add textures?
};

