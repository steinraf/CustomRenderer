//
// Created by steinraf on 21/10/22.
//

#pragma once

#include "utility/vector.h"

namespace customRenderer{
    [[nodiscard]] __device__ __host__ static constexpr Vector3f getCameraOrigin() noexcept{ return {0.f, 0.f, 0.f}; }

    [[nodiscard]] __device__ __host__ static constexpr Vector3f getCameraLookAt() noexcept{ return {0.f, 0.f, -1.f}; }

    [[nodiscard]] __device__ __host__ static constexpr Vector3f getCameraUp() noexcept{ return {0.f, 1.f, 0.f}; }

    [[nodiscard]] __device__ __host__ static constexpr float getCameraFOV() noexcept{ return 90.f; }

    [[nodiscard]] __device__ __host__ static constexpr float getCameraAperture() noexcept{ return 0.0f; }

    [[nodiscard]] __device__ __host__ static constexpr int getNumSubsamples() noexcept{ return 256; }

    [[nodiscard]] __device__ __host__ static constexpr int getMaxRayDepth() noexcept{ return 100; }


}

