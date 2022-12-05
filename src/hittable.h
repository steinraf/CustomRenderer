//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "utility/frame.h"
#include "utility/ray.h"
#include "utility/vector.h"

template<typename Primitive>
class BLAS;

class Triangle;

struct Intersection {
    Vector3f p;
    Frame shFrame;
    //    Vector3f n;

    //    class Triangle const *triangle;
    //    class AreaLight const *emitter;
    BLAS<Triangle> const *mesh;

    Vector2f uv;

    float t;

    //TODO add textures?
};
