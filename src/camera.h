//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "utility/ray.h"
#include "utility/vector.h"


class Camera {
public:
    __host__ __device__ Camera(Vector3f origin, Vector3f lookAt, Vector3f _up, float vFOV, float aspectRatio,
                               float aperture,
                               float focusDist);

    __device__ Ray3f getRay(float s, float t, const Vector2f &sample) const;

private:
    Vector3f origin;
    Vector3f right, up, front;

    Matrix4f sampleToCamera;
    Matrix4f cameraToWorld;

    const float lensRadius;
    const float focusDist;
    const float far = 1000.f;
    const float near = 0.0001;
};
