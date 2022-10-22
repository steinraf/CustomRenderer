//
// Created by steinraf on 21/08/22.
//

#include "sphere.h"


__device__ bool Sphere::hit(const Ray &r, float tMin, float tMax, HitRecord &rec) const {

    auto oc = r.getOrigin() - center;
    const float a = r.getDirection().squaredNorm();
    const float halfB = oc.dot(r.getDirection());
    const float c = oc.squaredNorm() - radius * radius;

    const float discriminant = halfB * halfB - a * c;

    if (discriminant < 0) return false;

    const float discriminantSqrt = sqrt(discriminant);

    float root = (-halfB - discriminantSqrt) / a;

    if (root < tMin || tMax < root) {
        root = (-halfB + discriminantSqrt) / a;
        if (root < tMin || tMax < root)
            return false;
    }

    rec.t = root;
    rec.position = r.atTime(rec.t);
    rec.material = material;
    rec.setFaceNormal(r, (rec.position - center) / radius);


    return true;
}
