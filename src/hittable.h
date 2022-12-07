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

    BLAS<Triangle> const *mesh;

    Vector2f uv;

    float t;
};
