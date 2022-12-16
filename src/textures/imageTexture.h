//
// Created by steinraf on 02/12/22.
//

#pragma once


#include "../utility/vector.h"


#include <filesystem>


class Texture {
//private:
public:
    Vector3f *deviceTexture = nullptr;
    int width, height;
    int dim;
    Vector3f radiance;

//public:
    float *deviceCDF = nullptr;



    __host__ explicit Texture(const std::filesystem::path &imagePath) noexcept;


    __host__ __device__ constexpr explicit Texture(const Vector3f &radiance) noexcept
        : width(0), height(0), dim(3), radiance(radiance){
    }

    __host__ __device__ constexpr explicit Texture() noexcept
        : Texture(Vector3f{1.0f, 1.0f, 1.0f}) {
    }

    [[nodiscard]] __device__ constexpr float pdf(size_t idx) const noexcept {
        if(deviceTexture){
            if(idx == width*height - 1){
                return 1 - deviceCDF[idx];
            }else{
                assert((deviceCDF[idx+1] - deviceCDF[idx]) > EPSILON);
                return deviceCDF[idx+1] - deviceCDF[idx];
            }
        }else{
            return 1.f;
        }
    }


    [[nodiscard]] __device__ constexpr Color3f eval(const Vector2f &uv) const noexcept {
        if(deviceTexture) {

//            Checkerboard
//            Vector2f m_scale{0.1f, 0.1f}, m_delta{0.f, 0.f};
//            Color3f m_value1{1.f}, m_value2{0.f};
//
//            Vector2f p = uv / m_scale - m_delta;
//
//            auto a = static_cast<int>(floorf(p[0]));
//            auto b = static_cast<int>(floorf(p[1]));
//
//            auto mod = [](int a, int b)->int{
//                const int r = a % b;
//                return (r < 0) ? r + b : r;
//            };
//
//
//            if(mod(a + b, 2) == 0)
//                return m_value1;
//
//            return m_value2;

            //TODO add sin(theta) factor
//            printf("UVTexture: (%f, %f)\n", uv[0], uv[1]);
#ifndef NDEBUG
            if(!(uv[0] >= 0.f && uv[0] <= 1.f && uv[1] >= 0.f && uv[1] <= 1.f)){
                printf("Failed UV (%f, %f)\n", uv[0], uv[1]);
                //TODO find reason
//                assert(false);
            }
#endif
            const auto h = static_cast<float>(height);
            const auto w = static_cast<float>(width);


            const float x = CustomRenderer::clamp(uv[0] * (w-1.f), 0.f, w-1-0.01f);

            const float y = CustomRenderer::clamp(h - 2 - uv[1] * (h-1.f), 0.f, h-1-0.01f);

            const int x1 = std::floor(x), x2 = x1+1;
            const int y1 = std::floor(y), y2 = y1+1;
#ifndef NDEBUG
            if(x1 < 0 || y1 < 0 || x2 >= w || y2 >= h){
                printf("Wrong Coords (%f, %f) -> (%i, %i, %i, %i)\n", x, y, x1, x2, y1, y2);
            }
#endif

            const float w1 = (x2-x)*(y2-y), w2 = (x-x1)*(y2-y), w3 = (x2-x)*(y-y1), w4 = (x-x1)*(y-y1);

            assert(w1 >= 0 && w2 >= 0 && w3 >= 0 && w4 >= 0);

            return w1 * deviceTexture[y1*width + x1] +
                   w2 * deviceTexture[y1*width + x2] +
                   w3 * deviceTexture[y2*width + x1] +
                   w4 * deviceTexture[y2*width + x2];
        } else {
            return radiance;
        }
    }
};

struct ColorToRadiance{
    const Vector3f *const first;
    const int width, height;


    __host__ __device__ explicit ColorToRadiance(Vector3f *first, int width, int height) noexcept
        : first(first), width(width), height(height) {
    }


    __host__ __device__ constexpr float operator()(const Vector3f &vec) const noexcept {
        const size_t y = (&vec - first)%width;

        return CustomRenderer::clamp(vec.norm(), 0.f, 1000.f) * sin(y*1.f/height);
    }
};




struct ColorToCDF{
private:
    const Vector3f *const first;
    const int width, height;
    float totalArea;

public:
    __host__ __device__ explicit ColorToCDF(Vector3f *first, int width, int height, float totalArea) noexcept
        : first(first), width(width), height(height), totalArea(totalArea) {
    }

    __host__ __device__ constexpr float operator()(const Vector3f &vec) const noexcept {
        const size_t y = (&vec - first)%width;
        return CustomRenderer::clamp(vec.norm(), 0.f, 1000.f) * sin(y*1.f/height) / totalArea;
    }
};

class NormalMap{
    Texture texture;
    
};
