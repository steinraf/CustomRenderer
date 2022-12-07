//
// Created by steinraf on 02/12/22.
//

#pragma once


#include "../utility/vector.h"


#include <filesystem>


class Texture {
private:
    Vector3f *deviceTexture = nullptr;
    int width, height;
    int dim;
    Vector3f radiance;

public:
    __host__ explicit Texture(const std::filesystem::path &imagePath) noexcept;


    __host__ __device__ constexpr explicit Texture(const Vector3f &radiance) noexcept
        : width(0), height(0), dim(3), radiance(radiance) {
    }

    __host__ __device__ constexpr explicit Texture() noexcept
        : Texture(Vector3f{0.f}) {
    }

    [[nodiscard]] __device__ constexpr Color3f eval(const Vector2f &uv) const noexcept {
        if(deviceTexture) {

//            Checkerboard
                        Vector2f m_scale{0.1f, 0.1f}, m_delta{0.f, 0.f};
                        Color3f m_value1{1.f}, m_value2{0.f};

                        Vector2f p = uv / m_scale - m_delta;

                        auto a = static_cast<int>(floorf(p[0]));
                        auto b = static_cast<int>(floorf(p[1]));

                        auto mod = [](int a, int b)->int{
                            const int r = a % b;
                            return (r < 0) ? r + b : r;
                        };


                        if(mod(a + b, 2) == 0)
                            return m_value1;

                        return m_value2;

            //TODO mipmapping
            return deviceTexture[(height - 1 - static_cast<int>(uv[1] * height)) * width + static_cast<int>(uv[0] * width)];
        } else {
            return radiance;
        }
    }
};
