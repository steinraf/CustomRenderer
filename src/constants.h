//
// Created by steinraf on 21/10/22.
//

#pragma once

#include "utility/vector.h"
//    (-11.3362,-0.0301921,-25.9856) to (11.4425,36.1183,-8.58231)
namespace customRenderer{

//    [[nodiscard]] __device__ __host__ static constexpr float getCameraFOV() noexcept{ return 30.f; }//return 90.f; }

    [[nodiscard]] __device__ __host__ static constexpr float getCameraAperture() noexcept{ return 0.f; }

    [[nodiscard]] __device__ __host__ static constexpr int getMaxRayDepth() noexcept{ return 3; }


}

