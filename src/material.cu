////
//// Created by steinraf on 20/08/22.
////
//
//#include "material.h"
//#include "utility/warp.h"
//
//
//__device__ bool Lambertian::scatter(const Ray &rayIn, const HitRecord &rec, Vector3f &attenuation, Ray &scattered,
//                                    Sampler &sampler) const {
//
//    Vector3f scatter = rec.normal + Warp::RandomInUnitSphere(sampler);
//    if (scatter.squaredNorm() < 1e-5)
//        scatter = rec.normal;
//
//    const Vector3f target = rec.position + scatter;
//
//    scattered = Ray(rec.position, target - rec.position);
//    attenuation = albedo;
//
//    return true;
//}