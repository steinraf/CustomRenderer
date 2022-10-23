//
// Created by steinraf on 23/10/22.
//

#include "triangle.h"

__device__ bool Triangle::hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const {



    /* Find vectors for two edges sharing v[0] */
    Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

    /* Begin calculating determinant - also used to calculate U parameter */
    Vector3f pvec = r.getDirection().cross(edge2);

    /* If determinant is near zero, ray lies in plane of triangle */
    float det = edge1.dot(pvec);

//    printf("Edge is (%f,%f,%f), (%f,%f,%f)\n",edge1[0],edge1[1],edge1[2],edge2[0],edge2[1],edge2[2]);

//    printf("Det is %f\n", det);

    if (det > -1e-8f && det < 1e-8f)
        return false;
    float inv_det = 1.f / det;

//    printf("Hitting triangle with good det :bustingood:\n");

    /* Calculate distance from v[0] to ray origin */
    Vector3f tvec = r.getOrigin() - p0;

    /* Calculate U parameter and test bounds */
    float u = tvec.dot(pvec) * inv_det;
    if (u < 0.f || u > 1.f)
        return false;



    /* Prepare to test V parameter */
    Vector3f qvec = tvec.cross(edge1);

    /* Calculate V parameter and test bounds */
    float v = r.getDirection().dot(qvec) * inv_det;
    if (v < 0.f || u + v > 1.f)
        return false;


    rec.t = edge2.dot(qvec) * inv_det;

//    printf("Would intersect at %f\n", rec.t);

    if(rec.t >= tMin && rec.t <= tMax){

        rec.position = r.atTime(rec.t);
        rec.material = material;
        rec.setFaceNormal(r, (p1 - p0).cross(p2 - p0));
//        printf("Hitting triangle :bustingood: \n");
        return true;
    }

    return false;
}
