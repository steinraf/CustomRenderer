//
// Created by steinraf on 21/10/22.
//

#pragma once

#include "utility/vector.h"
//    (-11.3362,-0.0301921,-25.9856) to (11.4425,36.1183,-8.58231)
namespace customRenderer{
    [[nodiscard]] __device__ __host__ static constexpr Vector3f getCameraOrigin() noexcept{
//                return {-20.f, 18.f, -10.f}; // Ajax Ref
        return {-65.6055, 47.5762, 24.3583}; //Nori Ajax
    }

    [[nodiscard]] __device__ __host__ static constexpr Vector3f getCameraLookAt() noexcept{
//        return {-0.f, 22.f, -18.f};
        return {-64.8161, 47.2211, 23.8576};
    }

    [[nodiscard]] __device__ __host__ static constexpr Vector3f getCameraUp() noexcept{
        return {0.299858, 0.934836, -0.190177};
    }

    [[nodiscard]] __device__ __host__ static constexpr float getCameraFOV() noexcept{ return 30.f; }//return 90.f; }

    [[nodiscard]] __device__ __host__ static constexpr float getCameraAperture() noexcept{ return 0.f; }

    [[nodiscard]] __device__ __host__ static constexpr int getNumSubsamples() noexcept{ return 32; }

    [[nodiscard]] __device__ __host__ static constexpr int getMaxRayDepth() noexcept{ return 1; }


}

