//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "utility/vector.h"
#include "utility/ray.h"



struct HitRecord{
    Vector3f position;
    Vector3f normal;

    class Triangle const *triangle;

    Vector2f uv;

    float t;
    bool frontFace;

    __device__ inline void setFaceNormal(const Ray &r, const Vector3f &outwardNormal){
        frontFace = r.getDirection().dot(outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }



    //TODO hit records need to set in the shape class, and not just with ray.at(t) because of numerical
    //instabilities

    //TODO add textures?
};

