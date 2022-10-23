#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cuda/std/cassert>

class Vector3f {
public:
    __host__ __device__ constexpr Vector3f() noexcept : data{0.0f, 0.0f, 0.0f} {}

    __host__ __device__ constexpr Vector3f(float x, float y, float z) noexcept : data{x, y, z} {}

    __host__ __device__ constexpr explicit Vector3f(float v) noexcept : data{v, v, v} {}

    [[nodiscard]] __host__ __device__ inline float operator[](int i)  const noexcept { return data[i]; }

    [[nodiscard]] __host__ __device__ inline float &operator[](int i) noexcept { return data[i]; };

    [[nodiscard]] __host__ __device__ inline Vector3f operator-() const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector3f operator+(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector3f operator-(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector3f operator*(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector3f operator/(const Vector3f &v2) const;

    [[nodiscard]] __host__ __device__ inline Vector3f operator*(float t) const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector3f operator/(float t) const;

    __host__ __device__ inline Vector3f &operator+=(const Vector3f &v2) noexcept;

    __host__ __device__ inline Vector3f &operator-=(const Vector3f &v2) noexcept;

    __host__ __device__ inline Vector3f &operator*=(const Vector3f &v2) noexcept;

    __host__ __device__ inline Vector3f &operator/=(const Vector3f &v2);

    __host__ __device__ inline Vector3f &operator*=(float t) noexcept;

    __host__ __device__ inline Vector3f &operator/=(float t);

    [[nodiscard]] __host__ __device__ inline float norm() const noexcept;

    [[nodiscard]] __host__ __device__ inline float squaredNorm() const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector3f normalized() const;

    [[nodiscard]] __host__ __device__ inline float dot(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector3f cross(const Vector3f &v2) const;

    [[nodiscard]] __host__ __device__ inline int asColor(size_t i) const noexcept;

    __host__ __device__ inline Vector3f &clamp(float minimum, float maximum) noexcept;


    friend inline std::ostream &operator<<(std::ostream &os, const Vector3f &t);


private:
    float data[3];
};

using Color = Vector3f;


inline std::ostream &operator<<(std::ostream &os, const Vector3f &t) {
    os << t.asColor(0) << ' '
       << t.asColor(1) << ' '
       << t.asColor(2) << '\n';
    return os;
}


__host__ __device__ inline int Vector3f::asColor(size_t i) const noexcept {
    return int(255.99 * data[i]);
}

__host__ __device__ inline float Vector3f::norm() const noexcept {
    return std::sqrt(squaredNorm());
}

__host__ __device__ inline float Vector3f::squaredNorm() const noexcept {
    return data[0] * data[0]
           + data[1] * data[1]
           + data[2] * data[2];
}


__host__ __device__ inline Vector3f Vector3f::normalized() const {
    float n = norm();
    assert(n != 0);
    float k = 1.f / n;
    return {
            data[0] * k,
            data[1] * k,
            data[2] * k
    };
}

__host__ __device__ inline Vector3f Vector3f::operator-() const noexcept{
    return {-data[0], -data[1], -data[2]};
}


__host__ __device__ inline Vector3f Vector3f::operator+(const Vector3f &v2) const noexcept {
    return {data[0] + v2.data[0],
                    data[1] + v2.data[1],
                    data[2] + v2.data[2]};
}

__host__ __device__ inline Vector3f Vector3f::operator-(const Vector3f &v2) const noexcept {
    return {data[0] - v2.data[0],
                    data[1] - v2.data[1],
                    data[2] - v2.data[2]};
}

__host__ __device__ inline Vector3f Vector3f::operator*(const Vector3f &v2) const noexcept {
    return {data[0] * v2.data[0],
                    data[1] * v2.data[1],
                    data[2] * v2.data[2]};
}

__host__ __device__ inline Vector3f Vector3f::operator/(const Vector3f &v2) const {
    assert(v2.data[0] != 0);
    assert(v2.data[1] != 0);
    assert(v2.data[2] != 0);
    return {data[0] / v2.data[0],
                    data[1] / v2.data[1],
                    data[2] / v2.data[2]};
}

__host__ __device__ inline Vector3f operator*(float t, const Vector3f &v) noexcept {
    return v * t;
}

__host__ __device__ inline Vector3f Vector3f::operator*(float t) const noexcept {
    return {t * data[0],
                    t * data[1],
                    t * data[2]};
}


__host__ __device__ inline Vector3f Vector3f::operator/(float t) const {
    assert(t != 0);
    return {data[0] / t,
                    data[1] / t,
                    data[2] / t};
}


__host__ __device__ inline float Vector3f::dot(const Vector3f &v2) const noexcept {
    return data[0] * v2.data[0] +
           data[1] * v2.data[1] +
           data[2] * v2.data[2];
}

__host__ __device__ inline Vector3f Vector3f::cross(const Vector3f &v2) const {
    return {data[1] * v2.data[2] - data[2] * v2.data[1],
                    data[2] * v2.data[0] - data[0] * v2.data[2],
                    data[0] * v2.data[1] - data[1] * v2.data[0]};
}


__host__ __device__ inline Vector3f &Vector3f::operator+=(const Vector3f &v) noexcept{
    data[0] += v.data[0];
    data[1] += v.data[1];
    data[2] += v.data[2];
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator*=(const Vector3f &v) noexcept {
    data[0] *= v.data[0];
    data[1] *= v.data[1];
    data[2] *= v.data[2];
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator/=(const Vector3f &v) {
    assert(v.data[0] != 0);
    assert(v.data[1] != 0);
    assert(v.data[2] != 0);

    data[0] /= v.data[0];
    data[1] /= v.data[1];
    data[2] /= v.data[2];
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator-=(const Vector3f &v) noexcept {
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    data[2] -= v.data[2];
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator*=(float t) noexcept {
    data[0] *= t;
    data[1] *= t;
    data[2] *= t;
    return *this;
}

__host__ __device__ inline Vector3f &Vector3f::operator/=(float t) {
    assert(t != 0);
    float k = 1.f / t;

    data[0] *= k;
    data[1] *= k;
    data[2] *= k;
    return *this;
}

__host__ __device__ inline Vector3f unit_vector(Vector3f v) {
    auto n = v.norm();
    assert(n != 0);
    return v / n;
}

__host__ __device__ inline Vector3f &Vector3f::clamp(float minimum, float maximum) noexcept{
    data[0] *= max(minimum, min(maximum, data[0]));
    data[1] *= max(minimum, min(maximum, data[1]));
    data[2] *= max(minimum, min(maximum, data[2]));
    return *this;
}


class Vector2f {
public:
    __host__ __device__ constexpr Vector2f() noexcept : data{0.0f, 0.0f} {}

    __host__ __device__ constexpr Vector2f(float x, float y) noexcept : data{x, y} {}

    __host__ __device__ constexpr explicit Vector2f(float v) noexcept : data{v, v} {}

    [[nodiscard]] __host__ __device__ inline float operator[](int i)  const noexcept { return data[i]; }

    [[nodiscard]] __host__ __device__ inline float &operator[](int i) noexcept { return data[i]; };

    [[nodiscard]] __host__ __device__ inline Vector2f operator-() const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector2f operator+(const Vector2f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector2f operator-(const Vector2f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector2f operator*(const Vector2f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector2f operator/(const Vector2f &v2) const;

    [[nodiscard]] __host__ __device__ inline Vector2f operator*(float t) const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector2f operator/(float t) const;

    __host__ __device__ inline Vector2f &operator+=(const Vector2f &v2) noexcept;

    __host__ __device__ inline Vector2f &operator-=(const Vector2f &v2) noexcept;

    __host__ __device__ inline Vector2f &operator*=(const Vector2f &v2) noexcept;

    __host__ __device__ inline Vector2f &operator/=(const Vector2f &v2);

    __host__ __device__ inline Vector2f &operator*=(float t) noexcept;

    __host__ __device__ inline Vector2f &operator/=(float t);

    [[nodiscard]] __host__ __device__ inline float norm() const noexcept;

    [[nodiscard]] __host__ __device__ inline float squaredNorm() const noexcept;

    [[nodiscard]] __host__ __device__ inline Vector2f normalized() const;

    [[nodiscard]] __host__ __device__ inline float dot(const Vector2f &v2) const noexcept;

    __host__ __device__ inline Vector2f &clamp(float minimum, float maximum) noexcept;


    friend inline std::ostream &operator<<(std::ostream &os, const Vector2f &t);


private:
    float data[2];
};

inline std::ostream &operator<<(std::ostream &os, const Vector2f &t) {
    os << t[0] << ' '
       << t[1] << ' ';
    return os;
}


__host__ __device__ inline float Vector2f::norm() const noexcept {
    return std::sqrt(squaredNorm());
}

__host__ __device__ inline float Vector2f::squaredNorm() const noexcept {
    return data[0] * data[0]
           + data[1] * data[1];
}


__host__ __device__ inline Vector2f Vector2f::normalized() const {
    float n = norm();
    assert(n != 0);
    float k = 1.f / n;
    return {
            data[0] * k,
            data[1] * k,
    };
}

__host__ __device__ inline Vector2f Vector2f::operator-() const noexcept{
    return {-data[0], -data[1]};
}


__host__ __device__ inline Vector2f Vector2f::operator+(const Vector2f &v2) const noexcept {
    return {data[0] + v2.data[0],
            data[1] + v2.data[1]};
}

__host__ __device__ inline Vector2f Vector2f::operator-(const Vector2f &v2) const noexcept {
    return {data[0] - v2.data[0],
            data[1] - v2.data[1]};
}

__host__ __device__ inline Vector2f Vector2f::operator*(const Vector2f &v2) const noexcept {
    return {data[0] * v2.data[0],
            data[1] * v2.data[1]};
}

__host__ __device__ inline Vector2f Vector2f::operator/(const Vector2f &v2) const {
    assert(v2.data[0] != 0);
    assert(v2.data[1] != 0);
    return {data[0] / v2.data[0],
            data[1] / v2.data[1]};
}

__host__ __device__ inline Vector2f operator*(float t, const Vector2f &v) noexcept {
    return v * t;
}

__host__ __device__ inline Vector2f Vector2f::operator*(float t) const noexcept {
    return {t * data[0],
            t * data[1]};
}


__host__ __device__ inline Vector2f Vector2f::operator/(float t) const {
    assert(t != 0);
    return {data[0] / t,
            data[1] / t};
}


__host__ __device__ inline float Vector2f::dot(const Vector2f &v2) const noexcept {
    return data[0] * v2.data[0] +
           data[1] * v2.data[1];
}

__host__ __device__ inline Vector2f &Vector2f::operator+=(const Vector2f &v) noexcept{
    data[0] += v.data[0];
    data[1] += v.data[1];
    return *this;
}

__host__ __device__ inline Vector2f &Vector2f::operator*=(const Vector2f &v) noexcept {
    data[0] *= v.data[0];
    data[1] *= v.data[1];
    return *this;
}

__host__ __device__ inline Vector2f &Vector2f::operator/=(const Vector2f &v) {
    assert(v.data[0] != 0);
    assert(v.data[1] != 0);

    data[0] /= v.data[0];
    data[1] /= v.data[1];
    return *this;
}

__host__ __device__ inline Vector2f &Vector2f::operator-=(const Vector2f &v) noexcept {
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    return *this;
}

__host__ __device__ inline Vector2f &Vector2f::operator*=(float t) noexcept {
    data[0] *= t;
    data[1] *= t;
    return *this;
}

__host__ __device__ inline Vector2f &Vector2f::operator/=(float t) {
    assert(t != 0);
    float k = 1.f / t;

    data[0] *= k;
    data[1] *= k;
    return *this;
}

__host__ __device__ inline Vector2f unit_vector(Vector2f v) {
    auto n = v.norm();
    assert(n != 0);
    return v / n;
}

__host__ __device__ inline Vector2f &Vector2f::clamp(float minimum, float maximum) noexcept{
    data[0] *= max(minimum, min(maximum, data[0]));
    data[1] *= max(minimum, min(maximum, data[1]));
    return *this;
}