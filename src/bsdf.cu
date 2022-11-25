//
// Created by steinraf on 24/10/22.
//

#include "bsdf.h"

__device__ bool
BSDF::scatter(const Ray &rIn, const Intersection &rec, Vector3f &attenuation, Ray &scattered, Sampler &sampler) const{

    switch(material){
        break;case Material::DIFFUSE:
            const Vector3f scatter = Warp::squareToCosineHemisphere(sampler.getSample2D());

            const Vector3f target = rec.p + scatter;

            scattered = Ray(rec.p, target - rec.p);
            attenuation = albedo;

            return true;
    }
}
