//
// Created by steinraf on 21/08/22.
//

#include "camera.h"
#include "utility/warp.h"

__device__ __host__ Camera::Camera(Vector3f origin, Vector3f lookAt, Vector3f _up, float vFOV,
                                   float aspectRatio, float aperture, float focusDist)
    : origin(origin), lensRadius(aperture / 2.0f), focusDist(focusDist) {

    const float k = tan(vFOV * M_PIf / 360.f);


    sampleToCamera = Matrix4f{
            2*k   , 0.f           , 0.f                      , -k,
            0.f , -2*k/aspectRatio, 0.f                      , k/aspectRatio,
            0.f , 0.f           , 0.f                      , 1.f,
            0.f , 0.f           , (near-far)/(near*far)   , 1.f/near
        };


    constexpr int noriConvert = 1;// -1 for nori, 1 for correct handedness


    front = (lookAt - origin).normalized();
    right = noriConvert * (_up.cross(-front)).normalized();
    up = front.cross(noriConvert * -right).normalized();

    cameraToWorld = Matrix4f{
            right[0], up[0], front[0], origin[0],
            right[1], up[1], front[1], origin[1],
            right[2], up[2], front[2], origin[2],
            0.f, 0.f, 0.f, 1.f,
    };

}

__device__ Ray3f Camera::getRay(float s, float t, const Vector2f &sample) const {


    const Vector2f randomDisk = lensRadius * Warp::squareToUniformDisk(sample);
    //    const Vector2f randomDisk = lensRadius * Warp::squareToUniformSquare(sample);
    //    const Vector3f triaSample =  Warp::squareToUniformTriangle(sample);
    //    const Vector3f triaPoint = lensRadius * (   Vector3f{-0.5, -0.5, 0} * triaSample[0] +
    //                                                Vector3f{-0.5,  0.5, 0} * triaSample[1] +
    //                                                Vector3f{ 0.5,  0.5, 0} * triaSample[2] );
    //    const Vector2f randomDisk{triaPoint[0], triaPoint[1]};




    const Vector3f nearP = (Vector3f{s, t, 0.f}.applyTransform(sampleToCamera)).normalized();
    Vector3f pLens(randomDisk[0], randomDisk[1], 0.f);
    float ft = focusDist / nearP[2];
    Vector3f pFocus = ft * nearP;
    Vector3f d_norm = (pFocus - pLens).normalized();

    return {
        pLens .applyTransform(cameraToWorld),
        (d_norm.applyTransform(cameraToWorld) - origin).normalized(),
        near,
        far
    };


//    Ray3f ray;
//
//    // Compute the sample position on the near plane (local camera space).
//    Vector3f near_p = Vector3f{randomDisk[0], randomDisk[1], 0.f}.applyTransform(sampleToCamera);
//
//    // Aperture position
//    Vector2f tmp = randomDisk;
//    Vector3f aperture_p(tmp[0], tmp[1], 0.f);
//
//    // Sampled position on the focal plane
//    Vector3f focus_p = near_p * (focusDist / near_p[2]);
//
//    // Convert into a normalized ray direction; adjust the ray interval accordingly.
//    Vector3f d = Vector3f(focus_p - aperture_p).normalized();
//
//    ray.o = aperture_p.applyTransform(cameraToWorld);
//    ray.d = d.applyTransform(cameraToWorld);
//
//    return {
//        ray.o + near * ray.d,
//        ray.d,
//        0,
//        far - near
//    };
}