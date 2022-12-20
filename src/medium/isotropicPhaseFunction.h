//
// Created by steinraf on 20/12/22.
//

#pragma once

#include "../bsdf.h"
#include "../utility/vector.h"
struct PhaseFunctionQueryRecord {
    /// Incident direction
    Vector3f wi;

    /// Outgoing direction
    Vector3f wo;

    /// Relative refractive index in the sampled direction
    float eta;

    /// Measure associated with the sample
    EMeasure measure;

    /// Create a new record for sampling the PhaseFunction
    __device__ PhaseFunctionQueryRecord(const Vector3f &wi)
        : wi(wi), measure(EUnknownMeasure) {

    }

    /// Create a new record for querying the PhaseFunction
    __device__ PhaseFunctionQueryRecord(const Vector3f &wi,
                             const Vector3f &wo, EMeasure measure)
        : wi(wi), wo(wo), measure(measure) {

    }

    /// Point associated with the point
    Vector3f p;
};


class IsotropicPhaseFunction {

public:
    [[nodiscard]] static __host__ __device__ Color3f sample(PhaseFunctionQueryRecord &bRec, const Vector2f &sample) {
        bRec.wo = Warp::squareToUniformSphere(sample);
        bRec.measure = ESolidAngle;
        bRec.eta = 1.0f;
        return Color3f{1.0};
    };

    [[nodiscard]] static __host__ __device__ Color3f eval(const PhaseFunctionQueryRecord &bRec) {
        return 0.25 * Color3f{1.00} * M_1_PI;
    };

    [[nodiscard]] __host__ __device__ float pdf(const PhaseFunctionQueryRecord &bRec) const {
        return 0.25f * M_1_PIf;
    };

    //Added for interchangeable nori syntax
    __device__ IsotropicPhaseFunction* operator->(){
        return this;
    }

    __device__ const IsotropicPhaseFunction* operator->() const{
        return this;
    }


};
