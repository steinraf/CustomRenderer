//
// Created by steinraf on 21/08/22.
//

#include "camera.h"
#include "utility/warp.h"

__device__ __host__ Camera::Camera(Vector3f origin, Vector3f lookAt, Vector3f up, float vFOV,
                          float aspectRatio, float aperture, float focusDist)
        : origin(origin), lensRadius(aperture / 2.0f) {

    const float theta = vFOV * M_PIf / 180.0f;

    const float halfHeight = tan(theta / 2.0f);
    const float halfWidth = aspectRatio * halfHeight;


    w = (origin - lookAt).normalized();
    u = (up.cross(w)).normalized();
    v = w.cross(u);

    const Vector3f halfU = halfWidth * focusDist * u;
    const Vector3f halfV = halfHeight * focusDist * v;

    lowerLeftCorner = origin - halfU
                      - halfV
                      - focusDist * w;

    horizontal = 2.0f * halfU;
    vertical = 2.0f * halfV;
}

__device__ Ray Camera::getRay(float s, float t, curandState *localRandState) const {
    const Vector3f randomDisk = lensRadius * Warp::RandomInUnitDisk(localRandState);
    const Vector3f offset = u * randomDisk[0] + v * randomDisk[1];

    const Vector3f pos = origin + offset;


    return Ray(pos, lowerLeftCorner + s * horizontal + t * vertical - pos);
}