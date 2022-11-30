//
// Created by steinraf on 21/10/22.
//



#pragma once

#include "vector.h"
#include "sampler.h"
#include <curand_kernel.h>


namespace Warp{

//    [[nodiscard]] __device__ static inline Vector3f RandomInUnitDisk(const Vector2f &sample){
//        const float r = sqrt(sample[0]);
//        const float phi = (2 * sample[1] - 1) * M_PIf;
//        return {r * sin(phi), r * cos(phi), 0};
//    }
//
//    [[nodiscard]] __device__ static inline Vector3f RandomInUnitSphere(const Vector2f &sample){
//
//        float cosT = 2 * sample[0] - 1;
//        float phi = 2 * M_PIf * sample[1];
//        float sTheta = sin(acos(cosT));
//        return {
//                sTheta * sin(phi),
//                sTheta * cos(phi),
//                cosT
//        };
//    }

    [[nodiscard]] __device__ inline Vector3f sampleUniformHemisphere(Sampler &sampler, const Vector3f &pole) {
        // Naive implementation using rejection sampling
        Vector3f v;
        do {
            v[0] = 1.f - 2.f * sampler.getSample1D();
            v[1] = 1.f - 2.f * sampler.getSample1D();
            v[2] = 1.f - 2.f * sampler.getSample1D();
        } while (v.squaredNorm() > 1.f);

        if (v.dot(pole) < 0.f)
            v = -v;
        v /= v.norm();

        return v;
    }

    [[nodiscard]] __device__ constexpr Vector2f squareToUniformSquare(const Vector2f &sample) {
        return sample;
    }

    [[nodiscard]] __device__ constexpr float squareToUniformSquarePdf(const Vector2f &sample) {
        return 1.f;
    }

    [[nodiscard]] __device__ constexpr Vector2f squareToUniformDisk(const Vector2f &sample) {
        const float r = sqrt(sample[0]);
        const float phi = (2 * sample[1] - 1) * M_PIf;
        return {r * sin(phi), r * cos(phi)};
    }

    [[nodiscard]] __device__ constexpr float squareToUniformDiskPdf(const Vector2f &p) {
        return static_cast<float>(p.squaredNorm() <= 1) * M_1_PIf;
    }

    [[nodiscard]] __device__ constexpr Vector3f squareToUniformSphereCap(const Vector2f &sample, float cosThetaMax) {
        const float cosT = sample[0] * (1 - cosThetaMax) + cosThetaMax;
        const float phi = 2 * M_PIf * sample[1];
        const float sTheta = sin(acos(cosT));
        return {
                sTheta * cos(phi),
                sTheta * sin(phi),
                cosT
        };
    }

    [[nodiscard]] __device__ constexpr float squareToUniformSphereCapPdf(const Vector3f &v, float cosThetaMax) {
        return static_cast<float>(v[2] >= cosThetaMax) * M_1_PIf / (2.f - 2.f*cosThetaMax);
    }

    [[nodiscard]] __device__ constexpr Vector3f squareToUniformSphere(const Vector2f &sample) {
        float cosT = 2 * sample[0] - 1;
        float phi = 2 * M_PIf * sample[1];
        float sTheta = sin(acos(cosT));
        return {
                sTheta * sin(phi),
                sTheta * cos(phi),
                cosT
        };
    }

    [[nodiscard]] __device__ constexpr float squareToUniformSpherePdf(const Vector3f &v) {
        return static_cast<float>((v.squaredNorm() - 1.f) < EPSILON) * M_1_PIf / 4.f;
    }

    [[nodiscard]] __device__ constexpr Vector3f squareToUniformHemisphere(const Vector2f &sample) {
        return squareToUniformSphereCap(sample, 0.f);
    }

    [[nodiscard]] __device__ constexpr float squareToUniformHemispherePdf(const Vector3f &v) {
        return squareToUniformSphereCapPdf(v, 0);
    }

    [[nodiscard]] __device__ constexpr Vector3f squareToCosineHemisphere(const Vector2f &sample) {
        auto p = squareToUniformDisk(sample);
        float z = sqrt(1 - p[0] * p[0] - p[1] * p[1]);
        return {p[0], p[1], z};
    }

    [[nodiscard]] __device__ constexpr float squareToCosineHemispherePdf(const Vector3f &v) {
        return v[2] < 0 ? 0.f : v[2] * M_1_PIf;
    }

    [[nodiscard]] __device__ constexpr Vector3f squareToBeckmann(const Vector2f &sample, float alpha) {
        const float cosTheta = sqrt(1.f / (1 - alpha * alpha * log(1 - sample[0])));
        const float phi = sample[1] * 2 * M_PIf;
        const float sTheta = sin(acos(cosTheta));
        return {
                sTheta * sin(phi),
                sTheta * cos(phi),
                cosTheta
        };
    }

    [[nodiscard]] __device__ constexpr float squareToBeckmannPdf(const Vector3f &m, float alpha) {
        const float cosT3 = m[2] * m[2] * m[2];
        if (cosT3 < FLT_EPSILON) return 0;
        const float alphaI2 = 1.f / (alpha * alpha);
        return exp((1.f - 1.f / (m[2] * m[2])) * alphaI2) * M_1_PIf * alphaI2 / (cosT3);
    }

    [[nodiscard]] __device__ constexpr Vector3f squareToUniformTriangle(const Vector2f &sample) {
        float su1 = sqrtf(sample[0]);
        float u = 1.f - su1, v = sample[1] * su1;
        return {u, v, 1.f - u - v};
    }
}