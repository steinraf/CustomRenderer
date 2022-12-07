//
// Created by steinraf on 07/12/22.
//

#include "../shapes/triangle.h"
#include "../utility/ray.h"
#include "../utility/vector.h"
#include "areaLight.h"

#pragma once


class EnvironmentEmitter {

private:
    Texture texture;

public:
    explicit __host__ __device__ constexpr EnvironmentEmitter(Texture texture) noexcept
        : texture(texture) {

    }

    EnvironmentEmitter() = default;

    [[nodiscard]] __device__ constexpr Color3f eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept {


        assert(false);
        return Color3f{0.f};
    }

    [[nodiscard]] __device__ constexpr float pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept{
        return 1.f;
    }


    [[nodiscard]] __device__ Color3f constexpr sample(EmitterQueryRecord &emitterQueryRecord, const Vector2f &sample) const noexcept{
        return Color3f{0.f};
    }

};
