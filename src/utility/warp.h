//
// Created by steinraf on 21/10/22.
//

#pragma once

#include "vector.h"
#include <curand_kernel.h>


namespace Warp {
    [[nodiscard]] __device__ static inline Vector3f Random(curandState *local_rand_state) {
        return {
            curand_uniform(local_rand_state),
            curand_uniform(local_rand_state),
            curand_uniform(local_rand_state)
        };
    }

    [[nodiscard]] __device__ static inline Vector3f  RandomInUnitDisk(curandState *local_rand_state) {
        const float r = sqrt(curand_uniform(local_rand_state));
        const float phi = (2 * curand_uniform(local_rand_state) - 1) * M_PIf;
        return {r * sin(phi), r * cos(phi), 0};
    }

    [[nodiscard]] __device__ static inline Vector3f  RandomInUnitSphere(curandState *local_rand_state) {

        float cosT = 2 * curand_uniform(local_rand_state) - 1;
        float phi = 2 * M_PIf * curand_uniform(local_rand_state);
        float sTheta = sin(acos(cosT));
        return {
                sTheta * sin(phi),
                sTheta * cos(phi),
                cosT
        };
    }


}