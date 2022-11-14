//
// Created by steinraf on 23/10/22.
//

#include "triangle.h"

//Nori Triangle Hit
__device__ bool Triangle::hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const{

//    printf("Trying to hit triangle %p\n", this);

//    printf("Intersecting triangle with coordinates\n"
//           "(%f, %f, %f) \n"
//           "(%f, %f, %f) \n"
//           "(%f, %f, %f) \n",
//           p0[0], p0[1], p0[2],
//           p1[0], p1[1], p1[2],
//           p2[0], p2[1], p2[2]);

    /* Find vectors for two edges sharing v[0] */
    Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

    /* Begin calculating determinant - also used to calculate U parameter */
    Vector3f pvec = r.getDirection().cross(edge2);

    /* If determinant is near zero, ray lies in plane of triangle */
    float det = edge1.dot(pvec);

//    printf("Edge is (%f,%f,%f), (%f,%f,%f)\n", edge1[0], edge1[1], edge1[2], edge2[0], edge2[1], edge2[2]);

//    printf("Det is %f\n", det);

    if(det > -1e-8f && det < 1e-8f){
//        printf("Determinant too small :skull: %f\n", det);
        return false;
    }

    float inv_det = 1.f / det;

    /* Calculate distance from v[0] to ray origin */
    Vector3f tvec = r.getOrigin() - p0;

    /* Calculate U parameter and test bounds */
    float u = tvec.dot(pvec) * inv_det;
    if(u < 0.f || u > 1.f){
//        printf("U outside :skull:\n");
        return false;
    }




    /* Prepare to test V parameter */
    Vector3f qvec = tvec.cross(edge1);

    /* Calculate V parameter and test bounds */
    float v = r.getDirection().dot(qvec) * inv_det;
    if(v < 0.f || u + v > 1.f){
//        printf("V outside :skull:\n");
        return false;
    }


    rec.t = edge2.dot(qvec) * inv_det;

//    printf("Would intersect at %f\n", rec.t);

    if(rec.t >= tMin && rec.t <= tMax){

        rec.position = r.atTime(rec.t);
        rec.triangle = this;
//        rec.bsdf = bsdf;
        rec.setFaceNormal(r, n0 * (1 - u - v) + n1 * u + n2 * v);
        return true;
    }

    // TODO add proper triangle test


    return false;
}
