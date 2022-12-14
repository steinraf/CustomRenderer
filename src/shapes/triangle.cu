//
// Created by steinraf on 14/12/22.
//


#include "triangle.h"
#include "../acceleration/bvh.h"

__device__ void Triangle::setHitInformation(const Ray3f &ray, Intersection &its) const noexcept {

    const float u = its.uv[0];
    const float v = its.uv[1];

    const Vector3f bary{1 - u - v, u, v};

    Vector3f normal = bary[0] * n0 + bary[1] * n1 + bary[2] * n2;


    Vector3f tangent{1.f, 1.f, 1.f};
    Frame f{tangent, -normal.cross(tangent), normal};
    Vector3f nMap;
    if(its.mesh->bsdf.material == Material::DIFFUSE)
        nMap = (2.f*its.mesh->normalMap.eval(bary[0] * uv0+ bary[1] * uv1+ bary[2] * uv2) - Vector3f{1.f}).normalized();
    else
        nMap = Vector3f{0.f, 0.f, 1.f};



//    if(nMap != Vector3f{0.f, 0.f, 1.f})
//        printf("Normal is (%f, %f, %f)\n", nMap[0], nMap[1], nMap[2]);
    its.shFrame =       Frame{f.toWorld(nMap)};


    its.p =             bary[0] * p0 + bary[1] * p1 + bary[2] * p2;
    its.uv =            bary[0] * uv0+ bary[1] * uv1+ bary[2] * uv2;

//    printf("Local UV's are (%f, %f), global are (%f, %f)\n", u, v, its.uv[0], its.uv[1]);

}
