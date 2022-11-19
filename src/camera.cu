//
// Created by steinraf on 21/08/22.
//

#include "camera.h"
#include "utility/warp.h"

__device__ __host__ Camera::Camera(Vector3f origin, Vector3f lookAt, Vector3f _up, float vFOV,
                                   float aspectRatio, float aperture, float focusDist)
        : origin(origin), lensRadius(aperture / 2.0f){


    const float halfHeight = tan(vFOV * M_PIf / 360.0f);
    const float halfWidth = aspectRatio * halfHeight;


    constexpr int noriConvert = -1; // -1 for nori, 1 for correct handedness


    front = (lookAt - origin).normalized();
    right = noriConvert*(_up.cross(-front)).normalized();
    up = front.cross(noriConvert*-right);

    const Vector3f halfU = halfWidth * focusDist * right;
    const Vector3f halfV = halfHeight * focusDist * up;

    upperLeft = origin
                - halfU
                + halfV
                + focusDist * front;

    horizontal = 2.0f * halfU;
    vertical = -2.0f * halfV;
}

__device__ Ray Camera::getRay(float s, float t, Sampler &sampler) const{
    const Vector2f randomDisk = lensRadius * Warp::squareToUniformDisk(sampler.getSample2D());
    const Vector3f offset = right * randomDisk[0] + up * randomDisk[1];

    const Vector3f pos = origin + offset;


    return {pos, upperLeft + s * horizontal + t * vertical - pos};
}