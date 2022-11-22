//
// Created by steinraf on 23/10/22.
//

#pragma once


#include "../utility/ray.h"
#include "../hittable.h"
#include "../bsdf.h"
#include "../acceleration/aabb.h"


class Triangle{
public:

    constexpr Triangle() noexcept = default;

    __device__ __host__ constexpr Triangle(
            const Vector3f &p0, const Vector3f &p1, const Vector3f &p2,
            const Vector2f &uv0, const Vector2f &uv1, const Vector2f &uv2,
            const Vector3f &n0, const Vector3f &n1, const Vector3f &n2,
            const BSDF &bsdf) noexcept
            : p0(p0), p1(p1), p2(p2),
              uv0(uv0), uv1(uv1), uv2(uv2),
              n0(n0), n1(n1), n2(n2),
              bsdf(bsdf), boundingBox(p0, p1, p2){

    }

    //Nori Triangle Ray intersect
    __device__ constexpr bool hit(const Ray &r, HitRecord &rec) const noexcept {


        /* Find vectors for two edges sharing v[0] */
        Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

        /* Begin calculating determinant - also used to calculate U parameter */
        Vector3f pvec = r.getDirection().cross(edge2);

        /* If determinant is near zero, ray lies in plane of triangle */
        float det = edge1.dot(pvec);


        if(det > -1e-8f && det < 1e-8f){
            return false;
        }

        float inv_det = 1.f / det;

        /* Calculate distance from v[0] to ray origin */
        Vector3f tvec = r.getOrigin() - p0;

        /* Calculate U parameter and test bounds */
        float u = tvec.dot(pvec) * inv_det;
        if(u < 0.f || u > 1.f){
            return false;
        }

        /* Prepare to test V parameter */
        Vector3f qvec = tvec.cross(edge1);

        /* Calculate V parameter and test bounds */
        float v = r.getDirection().dot(qvec) * inv_det;
        if(v < 0.f || u + v > 1.f){
            return false;
        }


        rec.t = edge2.dot(qvec) * inv_det;


        if(rec.t >= r.minDist && rec.t <= r.maxDist){
            rec.position = p0*(1 - u - v) + p1*u + p2*v;//r.atTime(rec.t);
            rec.triangle = this;
//        rec.bsdf = bsdf;
            rec.setFaceNormal(r, n0 * (1 - u - v) + n1 * u + n2 * v);
            return true;
        }


        return false;
    }

    __device__ constexpr void setHitInformation(const Ray &ray, HitRecord rec) const {
        //TODO set precise hit Info here
//        const Vector3f barycentric = {1-rec.uv};
    }

    BSDF bsdf;
//private:
    Vector3f p0, p1, p2;
    Vector2f uv0, uv1, uv2;
    Vector3f n0, n1, n2;

    AABB boundingBox;

};

