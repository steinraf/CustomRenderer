//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <curand_kernel.h>

#include "acceleration/bvh.h"
#include "camera.h"
#include "utility/ray.h"
#include "utility/warp.h"
#include <cuda/std/limits>


#define checkCudaErrors(val) cudaHelpers::check_cuda((val), #val, __FILE__, __LINE__)

struct FeatureBuffer {
    Color3f *variances;
    Color3f *albedos;
    Vector3f *positions;
    Vector3f *normals;
    int *numSubSamples;
};

struct FeatureBufferAccumulator {
    Color3f color;
    Color3f albedo;
    Vector3f position;
    Vector3f normal;
    int numSubSample = 0;
};

namespace cudaHelpers {


    __device__ bool inline initIndices(int &i, int &j, int &pixelIndex, const int width, const int height) noexcept {
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;

        if((i >= width) || (j >= height)) return false;

        pixelIndex = j * width + i;

        return true;
    }

    __host__ void check_cuda(cudaError_t result, char const *func, const char *file, int line);


    __global__ void initRng(int width, int height, curandState *randState);


    // The findSplit, delta and determineRange are taken from here
    // https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    // https://github.com/nolmoonen/cuda-lbvh/blob/main/src/build.cu

    __device__ int findSplit(const uint32_t *mortonCodes, int first, int last, int numPrimitives);

    __forceinline__ __device__ int delta(int a, int b, unsigned int n, const unsigned int *c, unsigned int ka) {
        // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
        //        assert (b >= 0 && b < n);
        if(b < 0 || b > n - 1) return -1;

        unsigned int kb = c[b];
        if(ka == kb) {
            // if keys are equal, use id as fallback
            // (+32 because they have the same morton code and thus the string-concatenated XOR
            //  version would have 32 leading zeros)
            return 32 + __clz(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
        }
        // clz = count leading zeros
        return __clz(ka ^ kb);
    }

    __forceinline__ __device__ thrust::pair<int, int>
    determineRange(const uint32_t *mortonCodes, int numPrimitives, int i) {
        const unsigned int *c = mortonCodes;
        const unsigned int ki = c[i];// key of i

        // determine direction of the range (+1 or -1)
        const int delta_l = delta(i, i - 1, numPrimitives, c, ki);
        const int delta_r = delta(i, i + 1, numPrimitives, c, ki);

        const auto [d, delta_min] = [&]() -> const thrust::pair<int, int> {
            if(delta_r < delta_l)
                return thrust::pair{-1, delta_r};
            else
                return thrust::pair{1, delta_l};
        }();

        // compute upper bound of the length of the range
        unsigned int l_max = 2;
        while(delta(i, i + l_max * d, numPrimitives, c, ki) > delta_min) {
            l_max <<= 1;
        }

        // find other end using binary search
        unsigned int l = 0;
        for(unsigned int t = l_max >> 1; t > 0; t >>= 1) {
            if(delta(i, i + (l + t) * d, numPrimitives, c, ki) > delta_min) {
                l += t;
            }
        }
        const int j = i + l * d;

        //        printf("Stats of range are i=%i, j=%i, l=%i, d=%i\n", i, j, l, d);

        // ensure i <= j
        return {min(i, j), max(i, j)};
    }


    __global__ void constructBVH(AccelerationNode *bvhNodes, Triangle *primitives, const uint32_t *mortonCodes, int numPrimitives);

    __device__ AABB getBoundingBox(AccelerationNode *root) noexcept;

    __global__ void computeBVHBoundingBoxes(AccelerationNode *bvhNodes) ;

    __global__ void initBVH(BLAS *bvh, AccelerationNode *bvhTotalNodes, float totalArea, const float *cdf,
            size_t numPrimitives, AreaLight *emitter, BSDF bsdf, Texture normalMap);

    __global__ void freeVariables();

     __device__ Color3f constexpr DirectMAS(const Ray3f &ray, TLAS *scene, Sampler &sampler) noexcept {
        Intersection its;
        if(!scene->rayIntersect(ray, its))
            return Color3f{0.f};


        Color3f sample{0.f};

        if(its.mesh->isEmitter()) {
            sample = its.mesh->getEmitter()->eval({ray.o, its.p, its.shFrame.n, its.uv});
        }


        BSDFQueryRecord bsdfQueryRecord{
                its.shFrame.toLocal(-ray.d)};
        bsdfQueryRecord.measure = ESolidAngle;
        bsdfQueryRecord.uv = its.uv;

        auto bsdfSample = its.mesh->getBSDF()->sample(bsdfQueryRecord, sampler.getSample2D());

        Ray3f newRay = {
                its.p,
                its.shFrame.toWorld(bsdfQueryRecord.wo)};

        Intersection emitterIntersect;

        if(scene->rayIntersect(newRay, emitterIntersect) && emitterIntersect.mesh->isEmitter()) {

            const auto &emitter = emitterIntersect.mesh->getEmitter();

            EmitterQueryRecord emitterQueryRecord{
                    ray.o,
                    its.p,
                    its.shFrame.n,
                    its.uv
            };

            sample += emitter->eval(emitterQueryRecord) * bsdfSample;
        }

        return sample;
    }

    __device__ Color3f constexpr DirectMIS(const Ray3f &ray, TLAS *scene, Sampler &sampler) {
        Intersection its;
        if(!scene->rayIntersect(ray, its))
            return Color3f{0.f};

        Color3f sample{0.f};

        if(its.mesh->isEmitter())
            sample = its.mesh->getEmitter()->eval({ray.o, its.p, its.shFrame.n, its.uv});


        const auto &emsLight = scene->getRandomEmitter(sampler.getSample1D());


        EmitterQueryRecord emsEmitterQueryRecord{its.p};


        const auto &currentLight = emsLight->sample(emsEmitterQueryRecord, sampler.getSample3D());


        BSDFQueryRecord emsBSDFRec{
                its.shFrame.toLocal(-ray.d),
                its.shFrame.toLocal(emsEmitterQueryRecord.wi),
                ESolidAngle};

        emsBSDFRec.uv = its.uv;


        Color3f emsSample{0.f};
        float emsWeight = 1;


        if(!scene->rayIntersect(emsEmitterQueryRecord.shadowRay)) {
            emsWeight = emsLight->pdf(emsEmitterQueryRecord) /
                        (emsLight->pdf(emsEmitterQueryRecord) + its.mesh->getBSDF()->pdf(emsBSDFRec));

            emsSample = its.mesh->getBSDF()->eval(emsBSDFRec) * currentLight * std::abs(its.shFrame.n.dot(emsEmitterQueryRecord.shadowRay.d)) * scene->numEmitters;
        }

        BSDFQueryRecord masBSDFQueryRecord{
                its.shFrame.toLocal(-ray.d)};

        masBSDFQueryRecord.uv = its.uv;

        auto masBSDFSample = its.mesh->getBSDF()->sample(masBSDFQueryRecord, sampler.getSample2D());

        Ray3f newMASRay{
                its.p,
                its.shFrame.toWorld(masBSDFQueryRecord.wo)};

        Intersection masEmitterIntersect;
        Color3f masSample{0.f};
        float masWeight = 1;


        if(scene->rayIntersect(newMASRay, masEmitterIntersect) && masEmitterIntersect.mesh->isEmitter()) {

            EmitterQueryRecord masEmitterQueryRecord{
                    its.p,
                    masEmitterIntersect.p,
                    masEmitterIntersect.shFrame.n,
                    masEmitterIntersect.uv
            };

            const auto &masEmitter = masEmitterIntersect.mesh->getEmitter();

            masWeight = its.mesh->getBSDF()->pdf(masBSDFQueryRecord) /
                        (masEmitter->pdf(masEmitterQueryRecord) + its.mesh->getBSDF()->pdf(masBSDFQueryRecord));

            masSample = masEmitter->eval(masEmitterQueryRecord) * masBSDFSample;
        }

        return sample + emsWeight * emsSample + masWeight * masSample;
    }

    __device__ Color3f constexpr PathMAS(const Ray3f &ray, TLAS *scene, int maxRayDepth, Sampler &sampler, FeatureBuffer &featureBuffer) noexcept {
        Intersection its;


        Color3f Li{0.f}, t{1.f};

        Ray3f currentRay = ray;

        int numBounces = 0;

        while(true) {

            if(!scene->rayIntersect(currentRay, its))
                return Li;


            if(its.mesh->isEmitter())
                Li += t * its.mesh->getEmitter()->eval({currentRay.o, its.p, its.shFrame.n, its.uv});

            float successProbability = fmin(t.maxCoeff(), 0.99f);
            //                if((++numBounces > 3) && sampler->next1D() > successProbability)
            if(sampler.getSample1D() >= successProbability || ++numBounces > maxRayDepth)
                return Li;

            t /= successProbability;

            BSDFQueryRecord bsdfQueryRecord{
                    its.shFrame.toLocal(-currentRay.d)};
            bsdfQueryRecord.measure = ESolidAngle;
            bsdfQueryRecord.uv = its.uv;

            const auto bsdfSample = its.mesh->getBSDF()->sample(bsdfQueryRecord, sampler.getSample2D());

            t *= bsdfSample;

            currentRay = {
                    its.p,
                    its.shFrame.toWorld(bsdfQueryRecord.wo)};
        }
    }

    __device__ Color3f constexpr PathMIS(const Ray3f &ray, TLAS *scene, int maxRayDepth, Sampler &sampler,
                                         FeatureBufferAccumulator &featureBuffer, size_t fbIndex) noexcept {
        Intersection its;


        Color3f Li{0.f}, t{1.f};

        Ray3f currentRay = ray;

        int numBounces = 0;

        float wMat = 1.0f;


        while(true) {
            assert(currentRay.getDirection().norm() != 0.f);

            if(!scene->rayIntersect(currentRay, its)) {
                if(t.norm() > EPSILON)
                    return Li + t * scene->environmentEmitter.eval(currentRay);
                else
                    return Li;
            }

            if(numBounces == 0) {
                featureBuffer.position = its.p;
                featureBuffer.normal = its.shFrame.n;
                featureBuffer.albedo = its.mesh->getBSDF()->getAlbedo(its.uv);
            }


            const auto *light = scene->getRandomEmitter(sampler.getSample1D());

            EmitterQueryRecord emitterQueryRecord{
                    its.p};

            const Color3f emsSample = light->sample(emitterQueryRecord, sampler.getSample3D()) * scene->numEmitters;

            if(!scene->rayIntersect(emitterQueryRecord.shadowRay)) {

                BSDFQueryRecord bsdfQueryRecord{
                        its.shFrame.toLocal(-currentRay.d),
                        its.shFrame.toLocal(emitterQueryRecord.wi),
                        ESolidAngle};
                bsdfQueryRecord.measure = ESolidAngle;
                bsdfQueryRecord.uv = its.uv;

                Li += emsSample * its.mesh->getBSDF()->eval(bsdfQueryRecord) * Frame::cosTheta(its.shFrame.toLocal(emitterQueryRecord.wi)) * light->pdf(emitterQueryRecord) /
                      (its.mesh->getBSDF()->pdf(bsdfQueryRecord) + light->pdf(emitterQueryRecord)) * t;
            }

//            EmitterQueryRecord envEQR{its.p};
//            const Color3f envSample = scene->environmentEmitter.sample(envEQR, sampler.getSample3D());
//            if(!scene->rayIntersect(envEQR.shadowRay)){
//                Li += t * wMat * envSample;
//            }

            if(its.mesh->isEmitter())
                Li += t * wMat * its.mesh->getEmitter()->eval({currentRay.o, its.p, its.shFrame.n, its.uv});

            float successProbability = fmin(t.maxCoeff(), 0.99f);
            //                if((++numBounces > 3) && sampler->next1D() > successProbability)
            if(sampler.getSample1D() >= successProbability || numBounces > maxRayDepth){
//                if(t.norm() > EPSILON)
//                    return Li + t * scene->environmentEmitter.eval(currentRay);
//                else
                    return Li;
            }

            t /= successProbability;

            BSDFQueryRecord bsdfQueryRecord{
                    its.shFrame.toLocal(-currentRay.d)};
            bsdfQueryRecord.measure = ESolidAngle;
            bsdfQueryRecord.uv = its.uv;

            t *= its.mesh->getBSDF()->sample(bsdfQueryRecord, sampler.getSample2D());

            currentRay = {
                    its.p,
                    its.shFrame.toWorld(bsdfQueryRecord.wo)
            };


            const float masPDF = its.mesh->getBSDF()->pdf(bsdfQueryRecord);

            Intersection masEmitterIntersect;
            if(!scene->rayIntersect(currentRay, masEmitterIntersect)){
                if(t.norm() > EPSILON)
                    return Li + t * scene->environmentEmitter.eval(currentRay);
                else
                    return Li;
                //TODO handle case where bsfd sample returns zero better
            }

            if(masEmitterIntersect.mesh->isEmitter()) {
                const float emsPDF = masEmitterIntersect.mesh->getEmitter()->pdf({currentRay.o,
                                                                                  masEmitterIntersect.p,
                                                                                  masEmitterIntersect.shFrame.n,
                                                                                  masEmitterIntersect.uv});
                wMat = masPDF + emsPDF > 0.f ? masPDF / (masPDF + emsPDF) : masPDF;
            }

            if(bsdfQueryRecord.measure == EDiscrete)
                wMat = 1.0f;

            ++numBounces;
        }
    }

    __device__ Color3f constexpr PathMISEnv(const Ray3f &ray, TLAS *scene, int maxRayDepth, Sampler &sampler,
                                         FeatureBufferAccumulator &featureBuffer, size_t fbIndex) noexcept {
        Intersection its;


        Color3f Li{0.f}, t{1.f};

        Ray3f currentRay = ray;

        int numBounces = 0;

        float wMat = 1.0f;


        while(true) {
            assert(currentRay.getDirection().norm() != 0.f);

            if(!scene->rayIntersect(currentRay, its)) {
                if(t.norm() > EPSILON)
                    return Li + t * scene->environmentEmitter.eval(currentRay);
                else
                    return Li;
            }

            //TODO fix normals for ajax

            if(numBounces == 0) {
                featureBuffer.position = its.p;
                featureBuffer.normal = its.shFrame.n;
                featureBuffer.albedo = its.mesh->getBSDF()->getAlbedo(its.uv);
            }


            //environmentMap Sampling
            EmitterQueryRecord envMapEQR{its.p};

            const Color3f envMapEMSSample = scene->environmentEmitter.sample(envMapEQR, sampler.getSample3D());

            if(!scene->rayIntersect(envMapEQR.shadowRay)) {

                BSDFQueryRecord bsdfQueryRecord{
                        its.shFrame.toLocal(-currentRay.d),
                        its.shFrame.toLocal(envMapEQR.wi),
                        ESolidAngle};
                bsdfQueryRecord.measure = ESolidAngle;
                bsdfQueryRecord.uv = its.uv;

                const float tempPDF = scene->environmentEmitter.pdf(envMapEQR);

//                printf("Normal is %f, %f, %f\n", its.shFrame.n[0], its.shFrame.n[1], its.shFrame.n[2]);
                Li += envMapEMSSample
                      * t
                      * its.mesh->getBSDF()->eval(bsdfQueryRecord)
                      * Frame::cosTheta(its.shFrame.toLocal(envMapEQR.wi))
                      * tempPDF
                      / (its.mesh->getBSDF()->pdf(bsdfQueryRecord) + tempPDF)
                ;
            }


            //Emitter sampling
            const auto *light = scene->getRandomEmitter(sampler.getSample1D());

            EmitterQueryRecord emitterQueryRecord{
                    its.p};
//
            const Color3f emsSample = light->sample(emitterQueryRecord, sampler.getSample3D()) * scene->numEmitters;

            if(!scene->rayIntersect(emitterQueryRecord.shadowRay)) {

                BSDFQueryRecord bsdfQueryRecord{
                        its.shFrame.toLocal(-currentRay.d),
                        its.shFrame.toLocal(emitterQueryRecord.wi),
                        ESolidAngle};
                bsdfQueryRecord.measure = ESolidAngle;
                bsdfQueryRecord.uv = its.uv;

                Li += emsSample * its.mesh->getBSDF()->eval(bsdfQueryRecord) * Frame::cosTheta(its.shFrame.toLocal(emitterQueryRecord.wi)) * light->pdf(emitterQueryRecord) /
                      (its.mesh->getBSDF()->pdf(bsdfQueryRecord) + light->pdf(emitterQueryRecord)) * t;
            }

            if(its.mesh->isEmitter())
                Li += t * wMat * its.mesh->getEmitter()->eval({currentRay.o, its.p, its.shFrame.n, its.uv});

            float successProbability = fmin(t.maxCoeff(), 0.99f);
            //                if((++numBounces > 3) && sampler->next1D() > successProbability)
            if(sampler.getSample1D() >= successProbability || numBounces > maxRayDepth){
//                if(t.norm() > EPSILON)
//                    return Li + t * scene->environmentEmitter.eval(currentRay)
//                else
                    return Li;
            }

            t /= successProbability;

            BSDFQueryRecord bsdfQueryRecord{
                    its.shFrame.toLocal(-currentRay.d)};
            bsdfQueryRecord.measure = ESolidAngle;
            bsdfQueryRecord.uv = its.uv;

            t *= its.mesh->getBSDF()->sample(bsdfQueryRecord, sampler.getSample2D());

            currentRay = {
                    its.p,
                    its.shFrame.toWorld(bsdfQueryRecord.wo)
            };


            const float masPDF = its.mesh->getBSDF()->pdf(bsdfQueryRecord);

            Intersection masEmitterIntersect;
            if(!scene->rayIntersect(currentRay, masEmitterIntersect)){
                if(t.norm() > EPSILON)
                    return Li + t * scene->environmentEmitter.eval(currentRay);
                else
                    return Li;
                //TODO handle case where bsfd sample returns zero better
            }

            if(masEmitterIntersect.mesh->isEmitter()) {
                const float emsPDF = masEmitterIntersect.mesh->getEmitter()->pdf({currentRay.o,
                                                                                  masEmitterIntersect.p,
                                                                                  masEmitterIntersect.shFrame.n,
                                                                                  masEmitterIntersect.uv});
                wMat = masPDF + emsPDF > 0.f ? masPDF / (masPDF + emsPDF) : masPDF;
            }

            if(bsdfQueryRecord.measure == EDiscrete)
                wMat = 1.0f;

            ++numBounces;
        }
    }

    __device__ Color3f constexpr normalMapper(const Ray3f &ray, TLAS *scene, Sampler &sampler) noexcept {
        Intersection its;
        Color3f Li{0.f};
        if(!scene->rayIntersect(ray, its))
            return Li;


        return its.shFrame.n.absValues();
    }

    __device__ Color3f constexpr checkerboard(const Ray3f &ray, TLAS *scene, int maxRayDepth, Sampler &sampler,
                                              FeatureBuffer &featureBuffer) noexcept {
        Intersection its;

        if(!scene->rayIntersect(ray, its))
            return Color3f{0.f};

        Vector2f m_scale{0.5f, 0.5f}, m_delta{0.f, 0.f};
        Color3f m_value1{1.f}, m_value2{0.f};

        Vector2f p = its.uv / m_scale - m_delta;

        auto a = static_cast<int>(floorf(p[0]));
        auto b = static_cast<int>(floorf(p[1]));

        auto mod = [] __device__ (int a, int b) -> int {
            const int r = a % b;

            return (r < 0) ? r + b : r;
        };

        if(mod(a + b, 2) == 0.0)
            return m_value1;

        return m_value2;
    }

    __device__ Color3f constexpr depthMapper(const Ray3f &ray, TLAS *scene, Sampler &sampler) noexcept {
        Intersection its;
        Color3f Li{0.f};
        if(!scene->rayIntersect(ray, its))
            return Li;

        return (its.p + Vector3f(EPSILON)).normalized().absValues();
    }


    __device__ Color3f constexpr getColor(const Ray3f &ray, TLAS *scene, int maxRayDepth, Sampler &sampler,
                                          FeatureBufferAccumulator &featureBuffer, size_t fbIndex) noexcept {


        //        return DirectMAS(ray, scene, sampler);
        //        return DirectMIS(ray, scene, sampler);
//                return PathMAS(ray, scene, maxRayDepth, sampler, featureBuffer);
//        return PathMIS(ray, scene, maxRayDepth, sampler, featureBuffer, fbIndex);
        return PathMISEnv(ray, scene, maxRayDepth, sampler, featureBuffer, fbIndex);

        //        return normalMapper(ray, scene, sampler);
        //        return depthMapper(ray, scene, sampler);
//                return checkerboard(ray, scene, maxRayDepth, sampler, featureBuffer);
    }

    __global__ void constructTLAS(TLAS *tlas,
                                  BLAS **meshBlasArr, size_t numMeshes,
                                  BLAS **emitterBlasArr, size_t numEmitters,
                                  EnvironmentEmitter environmentEmitter) ;

    template<typename T>
    [[nodiscard]] __host__ T *hostVecToDeviceRawPtr(std::vector<T> hostVec) noexcept(false) {
        T *deviceVec;
        auto numBytes = sizeof(T) * hostVec.size();

        checkCudaErrors(cudaMalloc(&deviceVec, numBytes));
        checkCudaErrors(cudaMemcpy(deviceVec, hostVec.data(), numBytes, cudaMemcpyHostToDevice));

        return deviceVec;
    }

    __device__ void bilateralFilterWiki(Vector3f *input, Vector3f *output, int i, int j, int width, int height);
    __device__ void bilateralFilterSlides(Vector3f *input, Vector3f *output, int i, int j, int width, int height);

    __global__ void applyGaussian(Vector3f *input, Vector3f *output, int width, int height, float sigma=0.1, int windowRadius=3);

    __global__ void denoise(Vector3f *input, Vector3f *output, FeatureBuffer featureBuffer, float *weights, int width, int height,
                            Vector3f cameraOrigin = Vector3f{0.f});

    __global__ void denoiseApplyWeights(Vector3f *output, float *weights, int width, int height);

    __global__ void render(Vector3f *output, Camera cam, TLAS *tlas, int width, int height, int numSubsamples,
           int maxRayDepth, curandState *globalRandState, FeatureBuffer featureBuffer, unsigned *progressCounter);




}// namespace cudaHelpers
