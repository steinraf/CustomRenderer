//
// Created by steinraf on 21/10/22.
//

#pragma once

#include "vector.h"
#include <curand_kernel.h>


namespace Warp {

    [[nodiscard]] __device__ static inline Vector3f  RandomInUnitDisk(Sampler &sampler) {
        const float r = sqrt(sampler.getSample1D());
        const float phi = (2 * sampler.getSample1D() - 1) * M_PIf;
        return {r * sin(phi), r * cos(phi), 0};
    }

    [[nodiscard]] __device__ static inline Vector3f  RandomInUnitSphere(Sampler &sampler) {

        float cosT = 2 * sampler.getSample1D() - 1;
        float phi = 2 * M_PIf * sampler.getSample1D();
        float sTheta = sin(acos(cosT));
        return {
                sTheta * sin(phi),
                sTheta * cos(phi),
                cosT
        };
    }


}