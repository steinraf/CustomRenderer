#pragma once

#define EPSILON 1e-4

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cuda/std/cassert>
#include <thrust/extrema.h>
#include <sstream>
#include <charconv>
#include <iomanip>

namespace CustomRenderer{
    template<typename T>
    [[nodiscard]] __host__ __device__ constexpr __forceinline__ T max(const T&a, const T&b) noexcept{
        return a < b ? b : a;
    }

    template<typename T>
    [[nodiscard]] __host__ __device__ constexpr __forceinline__ T min(const T&a, const T&b) noexcept{
        return a < b ? a : b;
    }
}



class Vector3f{
public:
    __host__ __device__ constexpr Vector3f() noexcept: data{0.0f, 0.0f, 0.0f}{}

    __host__ __device__ constexpr Vector3f(float x, float y, float z) noexcept : data{x, y, z}{}

    __host__ __device__ constexpr explicit Vector3f(float v) noexcept: data{v, v, v}{}

    __host__ constexpr explicit Vector3f(const std::string_view &str) : data{0.f, 0.f, 0.f}{

        auto checkConversion = [](std::from_chars_result res, float result){
            auto [ptr, ec] = res;
            if (ec == std::errc::invalid_argument || ec == std::errc::result_out_of_range){
                throw std::runtime_error("Invalid argument in vector construction from string.");
            }
        };

        std::string_view currentString(str);

        checkConversion(std::from_chars(currentString.begin(), currentString.end(), data[0]), data[0]);
        currentString.remove_prefix(currentString.find(',') + 1);
        currentString.remove_prefix(currentString.find_first_not_of(" \r\n\t\v\f"));

        checkConversion(std::from_chars(currentString.begin(), currentString.end(), data[1]), data[1]);
        currentString.remove_prefix(currentString.find(',') + 1);
        currentString.remove_prefix(currentString.find_first_not_of(" \r\n\t\v\f"));

        checkConversion(std::from_chars(currentString.begin(), currentString.end(), data[2]), data[2]);
        currentString.remove_prefix(currentString.find(',') + 1);
        currentString.remove_prefix(currentString.find_first_not_of(" \r\n\t\v\f"));

    }

    [[nodiscard]] __host__ __device__ constexpr inline float operator[](int i) const noexcept{ return data[i]; }

    [[nodiscard]] __host__ __device__ constexpr inline float &operator[](int i) noexcept{ return data[i]; };

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator-() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator+(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator-(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator*(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator/(const Vector3f &v2) const;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator*(float t) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator/(float t) const;

    __host__ __device__ constexpr inline bool operator==(const Vector3f &v2) const noexcept;

    __host__ __device__ constexpr inline bool operator!=(const Vector3f &v2) const noexcept;

    __host__ __device__ constexpr inline Vector3f &operator+=(const Vector3f &v2) noexcept;

    __host__ __device__ constexpr inline Vector3f &operator-=(const Vector3f &v2) noexcept;

    __host__ __device__ constexpr inline Vector3f &operator*=(const Vector3f &v2) noexcept;

    __host__ __device__ constexpr inline Vector3f &operator/=(const Vector3f &v2);

    __host__ __device__ constexpr inline Vector3f &operator*=(float t) noexcept;

    __host__ __device__ constexpr inline Vector3f &operator/=(float t);

    [[nodiscard]] __host__ __device__ constexpr inline float norm() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline float squaredNorm() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f normalized() const;

    [[nodiscard]] __host__ __device__ constexpr inline float dot(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f cross(const Vector3f &v2) const;

    [[nodiscard]] __host__ __device__ constexpr inline int asColor(size_t i) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f applyTransform(const class Affine3f &transform, bool isTranslationInvariant=false) const noexcept;


    __host__ __device__ constexpr inline Vector3f &clamp(float minimum, float maximum) noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f absValues() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline bool isEmpty() const noexcept;





    friend inline std::ostream &operator<<(std::ostream &os, const Vector3f &t);


private:
    float data[3];
};

using Color3f = Vector3f;

class Matrix3f{

public:
    __host__ __device__ constexpr Matrix3f(const Vector3f &row1 , const Vector3f &row2, const Vector3f &row3) noexcept
        :row1(row1), row2(row2), row3(row3){

    }



    [[nodiscard]] __host__ __device__ constexpr Vector3f operator*(const Vector3f &vec) const noexcept{
        return {
            row1.dot(vec),
            row2.dot(vec),
            row3.dot(vec)
        };
    }

    [[nodiscard]] __host__ __device__ constexpr Matrix3f operator*(const Matrix3f &mat) const noexcept{
        const Vector3f c1 = {mat.row1[0], mat.row2[0], mat.row3[0]},
                       c2 = {mat.row1[1], mat.row2[1], mat.row3[1]},
                       c3 = {mat.row1[2], mat.row2[2], mat.row3[2]};
        return {
                {row1.dot(c1), row1.dot(c2), row1.dot(c3)},
                {row2.dot(c1), row2.dot(c2), row2.dot(c3)},
                {row3.dot(c1), row3.dot(c2), row3.dot(c3)}
        };
    }

    __host__ __device__ constexpr Matrix3f(const Vector3f axis, float theta) noexcept{
        auto u = axis.normalized();
        const float cosT = cos(theta), sinT = sin(theta);
        assert(theta >= -2 * M_PIf && theta <= 2*M_PIf && "rotation should be in radians");
        *this = Matrix3f{
                {cosT + u[0]*u[0]*(1-cosT)     , u[0]*u[1]*(1-cosT) - u[2]*sinT, u[0]*u[2]*(1-cosT) + u[1]*sinT},
                {u[0]*u[1]*(1-cosT) + u[2]*sinT, cosT + u[1]*u[1]*(1-cosT)     , u[1]*u[2]*(1-cosT) - u[0]*sinT},
                {u[0]*u[2]*(1-cosT) - u[1]*sinT, u[1]*u[2]*(1-cosT) + u[0]*sinT, cosT + u[2]*u[2]*(1-cosT) }
        };
    }

    [[nodiscard]] __host__ __device__ static constexpr inline Matrix3f fromDiag(const Vector3f &vec) noexcept{
        return {
            {vec[0], 0, 0},
            {0, vec[1], 0},
            {0, 0, vec[2]}
        };
    }

    [[nodiscard]] __host__ __device__ static constexpr inline Matrix3f makeIdentity() noexcept{
        return fromDiag(Vector3f{1.f});
    }

private:
    Vector3f row1;
    Vector3f row2;
    Vector3f row3;
};

//TODO maybe change to quaternion representation
struct Affine3f{

    __host__ __device__ constexpr Affine3f(const Matrix3f &rotation , const Vector3f &translation) noexcept
            :rotation(rotation), translation(translation){

    }

    __host__ __device__ constexpr Affine3f(const Matrix3f &rotation) noexcept
            :rotation(rotation), translation(0.f){

    }

    __host__ __device__ constexpr Affine3f(const Vector3f &translation) noexcept
            :rotation(Matrix3f::makeIdentity()), translation(translation){

    }

    __host__ __device__ constexpr Affine3f() noexcept
            :rotation(Matrix3f::makeIdentity()), translation(0.f){

    }

    [[nodiscard]] __host__ __device__ constexpr Affine3f operator*(const Affine3f &other) const noexcept{
        return {
            rotation * other.rotation,
            rotation * other.translation + translation
        };
    }


    Matrix3f rotation;
    Vector3f translation;

};



__host__ __device__ constexpr inline Vector3f operator*(float t, const Vector3f &v) noexcept;


inline std::ostream &operator<<(std::ostream &os, const Vector3f &t){
    os << t.asColor(0) << ' '
       << t.asColor(1) << ' '
       << t.asColor(2) << '\n';
    return os;
}


__host__ __device__ constexpr inline int Vector3f::asColor(size_t i) const noexcept{
    return int(255.99 * data[i]);
}

__host__ __device__ constexpr inline float Vector3f::norm() const noexcept{
    return std::sqrt(squaredNorm());
}

__host__ __device__ constexpr inline float Vector3f::squaredNorm() const noexcept{
    return data[0] * data[0]
           + data[1] * data[1]
           + data[2] * data[2];
}


__host__ __device__ constexpr inline Vector3f Vector3f::normalized() const{
    float n = norm();
    assert(n != 0);
    float k = 1.f / n;
    return {
            data[0] * k,
            data[1] * k,
            data[2] * k
    };
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator-() const noexcept{
    return {-data[0], -data[1], -data[2]};
}


__host__ __device__ constexpr inline Vector3f Vector3f::operator+(const Vector3f &v2) const noexcept{
    return {data[0] + v2.data[0],
            data[1] + v2.data[1],
            data[2] + v2.data[2]};
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator-(const Vector3f &v2) const noexcept{
    return {data[0] - v2.data[0],
            data[1] - v2.data[1],
            data[2] - v2.data[2]};
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator*(const Vector3f &v2) const noexcept{
    return {data[0] * v2.data[0],
            data[1] * v2.data[1],
            data[2] * v2.data[2]};
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator/(const Vector3f &v2) const{
    assert(v2.data[0] != 0);
    assert(v2.data[1] != 0);
    assert(v2.data[2] != 0);
    return {data[0] / v2.data[0],
            data[1] / v2.data[1],
            data[2] / v2.data[2]};
}

__host__ __device__ constexpr inline Vector3f operator*(float t, const Vector3f &v) noexcept{
    return v * t;
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator*(float t) const noexcept{
    return {t * data[0],
            t * data[1],
            t * data[2]};
}


__host__ __device__ constexpr inline Vector3f Vector3f::operator/(float t) const{
    assert(t != 0);
    return {data[0] / t,
            data[1] / t,
            data[2] / t};
}


__host__ __device__ constexpr inline float Vector3f::dot(const Vector3f &v2) const noexcept{
    return data[0] * v2.data[0] +
           data[1] * v2.data[1] +
           data[2] * v2.data[2];
}

__host__ __device__ constexpr inline Vector3f Vector3f::cross(const Vector3f &v2) const{
    return {data[1] * v2.data[2] - data[2] * v2.data[1],
            data[2] * v2.data[0] - data[0] * v2.data[2],
            data[0] * v2.data[1] - data[1] * v2.data[0]};
}


__host__ __device__ constexpr inline Vector3f &Vector3f::operator+=(const Vector3f &v) noexcept{
    data[0] += v.data[0];
    data[1] += v.data[1];
    data[2] += v.data[2];
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator*=(const Vector3f &v) noexcept{
    data[0] *= v.data[0];
    data[1] *= v.data[1];
    data[2] *= v.data[2];
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator/=(const Vector3f &v){
    assert(v.data[0] != 0);
    assert(v.data[1] != 0);
    assert(v.data[2] != 0);

    data[0] /= v.data[0];
    data[1] /= v.data[1];
    data[2] /= v.data[2];
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator-=(const Vector3f &v) noexcept{
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    data[2] -= v.data[2];
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator*=(float t) noexcept{
    data[0] *= t;
    data[1] *= t;
    data[2] *= t;
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator/=(float t){
    assert(t != 0);
    float k = 1.f / t;

    data[0] *= k;
    data[1] *= k;
    data[2] *= k;
    return *this;
}

__host__ __device__ constexpr inline Vector3f unit_vector(Vector3f v){
    auto n = v.norm();
    assert(n != 0);
    return v / n;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::clamp(float minimum, float maximum) noexcept{
    data[0] *= CustomRenderer::max(minimum, CustomRenderer::min(maximum, data[0]));
    data[1] *= CustomRenderer::max(minimum, CustomRenderer::min(maximum, data[1]));
    data[2] *= CustomRenderer::max(minimum, CustomRenderer::min(maximum, data[2]));
    return *this;
}

__host__ __device__ constexpr inline Vector3f Vector3f::absValues() const noexcept{
    return {
            abs(data[0]),
            abs(data[1]),
            abs(data[2])
    };
}

__host__ __device__ constexpr inline bool Vector3f::isEmpty() const noexcept{
    return  (data[0] == 0) &&
            (data[1] == 0) &&
            (data[2] == 0);
}

__host__ __device__ constexpr inline bool Vector3f::operator==(const Vector3f &v2) const noexcept{
    return data[0] == v2.data[0] && data[1] == v2.data[1] && data[2] == v2.data[2];
}

__host__ __device__ constexpr inline bool Vector3f::operator!=(const Vector3f &v2) const noexcept{
    return !(*this == v2);
}

__host__ __device__ constexpr Vector3f Vector3f::applyTransform(const Affine3f &transform, bool isTranslationInvariant) const noexcept{
    if(isTranslationInvariant)
        return transform.rotation * (*this);
    else
        return transform.rotation * (*this) + transform.translation;
}


class Vector2f{
public:
    __host__ __device__ constexpr Vector2f() noexcept: data{0.0f, 0.0f}{}

    __host__ __device__ constexpr Vector2f(float x, float y) noexcept: data{x, y}{}

    __host__ __device__ constexpr explicit Vector2f(float v) noexcept: data{v, v}{}

    [[nodiscard]] __host__ __device__ constexpr inline float operator[](int i) const noexcept{ return data[i]; }

    [[nodiscard]] __host__ __device__ constexpr inline float &operator[](int i) noexcept{ return data[i]; };

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator-() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator+(const Vector2f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator-(const Vector2f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator*(const Vector2f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator/(const Vector2f &v2) const;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator*(float t) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator/(float t) const;

    __host__ __device__ constexpr inline Vector2f &operator+=(const Vector2f &v2) noexcept;

    __host__ __device__ constexpr inline Vector2f &operator-=(const Vector2f &v2) noexcept;

    __host__ __device__ constexpr inline Vector2f &operator*=(const Vector2f &v2) noexcept;

    __host__ __device__ constexpr inline Vector2f &operator/=(const Vector2f &v2);

    __host__ __device__ constexpr inline Vector2f &operator*=(float t) noexcept;

    __host__ __device__ constexpr inline Vector2f &operator/=(float t);

    [[nodiscard]] __host__ __device__ constexpr inline float norm() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline float squaredNorm() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f normalized() const;

    [[nodiscard]] __host__ __device__ constexpr inline float dot(const Vector2f &v2) const noexcept;

    __host__ __device__ constexpr inline Vector2f &clamp(float minimum, float maximum) noexcept;


    friend inline std::ostream &operator<<(std::ostream &os, const Vector2f &t);


private:
    float data[2];
};

inline std::ostream &operator<<(std::ostream &os, const Vector2f &t){
    os << t[0] << ' '
       << t[1] << ' ';
    return os;
}


__host__ __device__ constexpr inline float Vector2f::norm() const noexcept{
    return std::sqrt(squaredNorm());
}

__host__ __device__ constexpr inline float Vector2f::squaredNorm() const noexcept{
    return data[0] * data[0]
           + data[1] * data[1];
}


__host__ __device__ constexpr inline Vector2f Vector2f::normalized() const{
    float n = norm();
    assert(n != 0);
    float k = 1.f / n;
    return {
            data[0] * k,
            data[1] * k,
    };
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator-() const noexcept{
    return {-data[0], -data[1]};
}


__host__ __device__ constexpr inline Vector2f Vector2f::operator+(const Vector2f &v2) const noexcept{
    return {data[0] + v2.data[0],
            data[1] + v2.data[1]};
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator-(const Vector2f &v2) const noexcept{
    return {data[0] - v2.data[0],
            data[1] - v2.data[1]};
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator*(const Vector2f &v2) const noexcept{
    return {data[0] * v2.data[0],
            data[1] * v2.data[1]};
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator/(const Vector2f &v2) const{
    assert(v2.data[0] != 0);
    assert(v2.data[1] != 0);
    return {data[0] / v2.data[0],
            data[1] / v2.data[1]};
}

__host__ __device__ constexpr inline Vector2f operator*(float t, const Vector2f &v) noexcept{
    return v * t;
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator*(float t) const noexcept{
    return {t * data[0],
            t * data[1]};
}


__host__ __device__ constexpr inline Vector2f Vector2f::operator/(float t) const{
    assert(t != 0);
    return {data[0] / t,
            data[1] / t};
}


__host__ __device__ constexpr inline float Vector2f::dot(const Vector2f &v2) const noexcept{
    return data[0] * v2.data[0] +
           data[1] * v2.data[1];
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator+=(const Vector2f &v) noexcept{
    data[0] += v.data[0];
    data[1] += v.data[1];
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator*=(const Vector2f &v) noexcept{
    data[0] *= v.data[0];
    data[1] *= v.data[1];
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator/=(const Vector2f &v){
    assert(v.data[0] != 0);
    assert(v.data[1] != 0);

    data[0] /= v.data[0];
    data[1] /= v.data[1];
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator-=(const Vector2f &v) noexcept{
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator*=(float t) noexcept{
    data[0] *= t;
    data[1] *= t;
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator/=(float t){
    assert(t != 0);
    float k = 1.f / t;

    data[0] *= k;
    data[1] *= k;
    return *this;
}

__host__ __device__ constexpr inline Vector2f unit_vector(Vector2f v){
    auto n = v.norm();
    assert(n != 0);
    return v / n;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::clamp(float minimum, float maximum) noexcept{
    data[0] *= CustomRenderer::max(minimum, CustomRenderer::min(maximum, data[0]));
    data[1] *= CustomRenderer::max(minimum, CustomRenderer::min(maximum, data[1]));
    return *this;
}