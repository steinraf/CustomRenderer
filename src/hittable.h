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
    class AreaLight const *emitter;

    Vector2f uv;

    float t;


    //TODO rayIntersect records need to set in the shape class, and not just with ray.at(t) because of numerical
    //instabilities

    //TODO add textures?
};

