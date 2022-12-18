//
// Created by steinraf on 14/12/22.
//


#include "triangle.h"
#include "../acceleration/bvh.h"

__device__ void Triangle::setHitInformation(const Ray3f &ray, Intersection &its) const noexcept {

    const float u = its.uv[0];
    const float v = its.uv[1];

    const Vector3f bary{1 - u - v, u, v};

    its.p = getCoordinate(bary);
    its.uv = getUV(bary);

    its.uv = its.uv.clamp(0, 1);
    assert(its.uv[0] >= 0 && its.uv[0] <= 1 && its.uv[1] >= 0 && its.uv[1] <= 1);

    Vector3f normal = getNormal(bary);


    Vector3f edge1 = p1-p0, edge2 = p2-p0;
    Vector2f dUV1 = uv1-uv0, dUV2 = uv2-uv0;
    float uvMatDetInv = 1.f/(dUV1[0]*dUV2[1] - dUV1[1]*dUV2[0]);

    Vector3f tangent = (uvMatDetInv * (dUV2[1] * edge1 - dUV1[1] * edge2)).normalized();

//    Vector3f tangent{1.f, 1.f, 1.f};

    Frame f{tangent, -normal.cross(tangent), normal};
//    Vector3f nMap = its.mesh->normalMap.eval(its.uv);//
    Vector3f nMap = (2.f*its.mesh->normalMap.eval(its.uv) - Vector3f{1.f}).normalized();



//    if(nMap != Vector3f{0.f, 0.f, 1.f})
//        printf("Normal is (%f, %f, %f)\n", nMap[0], nMap[1], nMap[2]);
    its.shFrame =       Frame{f.toWorld(nMap)};


}