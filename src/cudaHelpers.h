//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <curand_kernel.h>

#include "utility/warp.h"
//#include "utility/vector.h"
#include "utility/ray.h"
#include "camera.h"
//#include "utility/meshLoader.h"
#include "acceleration/bvh.h"
#include "constants.h"
#include <cuda/std/limits>


#define checkCudaErrors(val) cudaHelpers::check_cuda( (val), #val, __FILE__, __LINE__ )


namespace cudaHelpers{

    __device__ bool inline initIndices(int &i, int &j, int &pixelIndex, const int width, const int height){
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

    __forceinline__ __device__ int delta(int a, int b, unsigned int n, const unsigned int *c, unsigned int ka){
        // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
//        assert (b >= 0 && b < n);
        if(b < 0 || b > n - 1) return -1;

        unsigned int kb = c[b];
        if(ka == kb){
            // if keys are equal, use id as fallback
            // (+32 because they have the same morton code and thus the string-concatenated XOR
            //  version would have 32 leading zeros)
            return 32 + __clz(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
        }
        // clz = count leading zeros
        return __clz(ka ^ kb);
    }

    __forceinline__ __device__ thrust::pair<int, int>
    determineRange(const uint32_t *mortonCodes, int numPrimitives, int i){
        const unsigned int *c = mortonCodes;
        const unsigned int ki = c[i]; // key of i

        // determine direction of the range (+1 or -1)
        const int delta_l = delta(i, i - 1, numPrimitives, c, ki);
        const int delta_r = delta(i, i + 1, numPrimitives, c, ki);

        const auto [d, delta_min] = [&]() -> const thrust::pair<int, int>{
            if(delta_r < delta_l)
                return thrust::pair{-1, delta_r};
            else
                return thrust::pair{1, delta_l};
        }();

        // compute upper bound of the length of the range
        unsigned int l_max = 2;
        while(delta(i, i + l_max * d, numPrimitives, c, ki) > delta_min){
            l_max <<= 1;
        }

        // find other end using binary search
        unsigned int l = 0;
        for(unsigned int t = l_max >> 1; t > 0; t >>= 1){
            if(delta(i, i + (l + t) * d, numPrimitives, c, ki) > delta_min){
                l += t;
            }
        }
        const int j = i + l * d;

//        printf("Stats of range are i=%i, j=%i, l=%i, d=%i\n", i, j, l, d);

        // ensure i <= j
        return {min(i, j), max(i, j)};
    }


    template<typename Primitive>
    __global__ void
    constructBVH(AccelerationNode<Primitive> *bvhNodes, Primitive *primitives, const uint32_t *mortonCodes,
                 int numPrimitives){

        const int i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i > numPrimitives - 1) return;


        // [0, numPrimitives-2]                    -> internal nodes
        // [numPrimitives-1, (2*numPrimitives)-1]     -> leaf nodes
        bvhNodes[numPrimitives - 1 + i] = {
                nullptr,
                nullptr,
                &primitives[i],
                primitives[i].boundingBox,
                true,
        };


        if(i == numPrimitives - 1) return;


        auto [first, last] = determineRange(mortonCodes, numPrimitives, i);

        int split = findSplit(mortonCodes, first, last, numPrimitives);

        AccelerationNode<Primitive> *childA = (split == first) ? &bvhNodes[numPrimitives - 1 + split]
                                                               : &bvhNodes[split];
        AccelerationNode<Primitive> *childB = (split + 1 == last) ? &bvhNodes[numPrimitives - 1 + split + 1]
                                                                  : &bvhNodes[split +
                                                                              1];

        bvhNodes[i] = {
                childA,
                childB,
                nullptr,
                /*(childA && childB && childA->isLeaf && childB->isLeaf) ? childA->boundingBox + childB->boundingBox : */AABB{}, //somehow this optimization does not always work :oof:
                false,
        };

//        if(i == 0)
//            printf("children of %p are %p and %p", bvhNodes[i], childA, childB);

    }

    template<typename Primitive>
    __device__ AABB getBoundingBox(AccelerationNode<Primitive> *root){

        typedef AccelerationNode<Primitive> *NodePtr;

        constexpr int stackSize = 1024;
        NodePtr stack[stackSize];
        int idx = 0;
        stack[0] = root;

        assert(root);

        NodePtr currentNode;

        do{

//            printf("STACK: "); for(int tmp = 0; tmp < idx+1; ++tmp) printf("| %p ", stack[tmp]);
//            printf("\n");

            assert(idx < stackSize);

            currentNode = stack[idx];

            NodePtr left = currentNode->left;
            NodePtr right = currentNode->right;

            assert(left && right);


//            printf("Left bounding box (%f, %f, %f) -> (%f, %f, %f)\n",
//                   left->boundingBox.min[0], left->boundingBox.min[1], left->boundingBox.min[2],
//                   left->boundingBox.max[0], left->boundingBox.max[1], left->boundingBox.max[2]);
//
//            printf("Right bounding box (%f, %f, %f) -> (%f, %f, %f)\n",
//                   right->boundingBox.min[0], right->boundingBox.min[1], right->boundingBox.min[2],
//                   right->boundingBox.max[0], right->boundingBox.max[1], right->boundingBox.max[2]);

            if(left->hasBoundingBox() && right->hasBoundingBox()){
//                for(int tmp = 0; tmp < idx; ++tmp) printf("\t");
//                printf("oB %p\n", currentNode);
                assert(!left->boundingBox.isEmpty() && !right->boundingBox.isEmpty());
                currentNode->boundingBox = left->boundingBox + right->boundingBox;
//                printf("New bounding box (%f, %f, %f) -> (%f, %f, %f)\n",
//                       currentNode->boundingBox.min[0], currentNode->boundingBox.min[1], currentNode->boundingBox.min[2],
//                       currentNode->boundingBox.max[0], currentNode->boundingBox.max[1], currentNode->boundingBox.max[2]);
                --idx;
            }else if(right->hasBoundingBox()){
//                for(int tmp = 0; tmp < idx; ++tmp) printf("\t");
//                printf("Le %p\n", currentNode);
                stack[++idx] = left;
            }else if(left->hasBoundingBox()){
//                for(int tmp = 0; tmp < idx; ++tmp) printf("\t");
//                printf("Ri %p\n", currentNode);
                stack[++idx] = right;
            }else{
//                for(int tmp = 0; tmp < idx; ++tmp) printf("\t");
//                printf("Bo %p\n", currentNode);

                stack[++idx] = right;
                stack[++idx] = left;
            }
        }while(idx >= 0);

        return root->boundingBox;

    }

    template<typename Primitive>
    __global__ void computeBVHBoundingBoxes(AccelerationNode<Primitive> *bvhNodes){
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

//        printf("Starting BLAS BB Computation...\n");


        const AABB &totalBoundingBox = getBoundingBox(&bvhNodes[0]);

        printf("\tTotal bounding box is (%f, %f, %f) -> (%f, %f, %f)\n",
               totalBoundingBox.min[0], totalBoundingBox.min[1], totalBoundingBox.min[2],
               totalBoundingBox.max[0], totalBoundingBox.max[1], totalBoundingBox.max[2]);

    }

    template<typename Primitive>
    __global__ void initBVH(BLAS<Primitive> *bvh, AccelerationNode<Primitive> *bvhTotalNodes, float *cdf, size_t numPrimitives){
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        *bvh = BLAS<Primitive>(bvhTotalNodes, cdf, numPrimitives);
    }

    __global__ void freeVariables();


    template<typename Primitive>
    __device__ Color3f getColor(const Ray &ray, TLAS<Primitive> *scene, int maxRayDepth, Sampler &sampler){

        Intersection its;

        if(!scene->rayIntersect(ray, its))
            return Color3f{0.f};

//
        return its.n.absValues();

//        printf("UV's are (%f, %f)\n", its.uv[0], its.uv[1]);


        Vector2f m_scale{0.5f, 0.5f}, m_delta{0.f, 0.f};
        Color3f m_value1{1.f}, m_value2{0.f};

        Vector2f p = its.uv/m_scale - m_delta;

        auto a = static_cast<int>(floorf(p[0]));
        auto b = static_cast<int>(floorf(p[1]));

        auto mod = [](int a, int b){
            const int r = a % b;
            return (r < 0) ? r + b : r;
        };

        if (mod(a + b, 2) == 0.f)
            return m_value1;

        return m_value2;



//        const Vector3f scatter = Warp::sampleUniformHemisphere(sampler, its.n);
//
//
//        if(!scene->rayIntersect({its.p, scatter, EPSILON, 1000}, its))
//            return Color3f{1.f};
//
//
//        return Color3f{0.f};

//        Ray scattered;
//        Color3f attenuation;
//
//        Color3f currentAttenuation{1.f};
//
//        for(int depth = 0; depth < maxRayDepth; ++depth){
//            if(bvh->rayIntersect(currentRay, FLT_EPSILON, cuda::std::numeric_limits<float>::infinity(), record)){
//                if(record.triangle->bsdf.scatter(currentRay, record, attenuation, scattered, sampler)){
//                    currentRay = scattered;
////                    currentAttenuation *= attenuation;
////                    return record.n;// * 255.99;
//                }else{
//                    return Color3f{0.f};
//                }
//
//            }else{
////                return Color3f{1.f};
////                float t = 0.5f * (r.getDirection().normalized()[1] + 1.f);
////                Color3f c = (1 - t) * Vector3f{1.f} + t * Color3f{0.5f, 0.7f, 1.0f};
//                const Color3f c{1.f};
//                return currentAttenuation * c;
//            }
//        }



    }

    template<typename Primitive>
    __global__ void
//    constructTLAS(AccelerationNode<Blas<Primitive>> *tlas, Blas<Primitive> *meshBlasArr, int numMeshes){
    constructTLAS(TLAS<Primitive> *tlas,
                      BLAS<Primitive> **meshBlasArr, size_t numMeshes,
                      BLAS<Primitive> **emitterBlasArr, Vector3f **emitterRadiance, size_t numEmitters){

        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        *tlas = TLAS(meshBlasArr, numMeshes, emitterBlasArr, emitterRadiance, numEmitters);

    }

    template<typename T>
    [[nodiscard]] __host__ T *hostVecToDeviceRawPtr(std::vector<T> hostVec) noexcept(false){
        T *deviceVec;
        auto numBytes = sizeof(T) * hostVec.size();

        checkCudaErrors(cudaMalloc(&deviceVec, numBytes));
        checkCudaErrors(cudaMemcpy(deviceVec, hostVec.data(), numBytes, cudaMemcpyHostToDevice));

        return deviceVec;
    }

    __global__ void denoise(Vector3f *input, Vector3f *output, int width, int height);

    template<typename Primitive>
    __global__ void render(Vector3f *output, Camera cam, TLAS<Primitive> *tlas, int width, int height, int numSubsamples,
                           curandState *globalRandState){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

//        if(i == 0 && j == 0){
//            printf("Testing Hit: \n");
//
//            auto r = Ray(customRenderer::getCameraOrigin(), customRenderer::getCameraLookAt() - customRenderer::getCameraOrigin());
//            Intersection h;
//            bool didHit = bvh->rayIntersect(r, FLT_EPSILON, INFINITY, h);
//            printf("Testing Hit: %d \n", didHit);
//            output[pixelIndex] = Vector3f{1.f};
//            return;
//        }else{
//            output[pixelIndex] = Vector3f{0.f};
//            return;
//        }


        auto sampler = Sampler(&globalRandState[pixelIndex]);

        const auto iFloat = static_cast<float>(i);
        const auto jFloat = static_cast<float>(j);

        const auto widthFloat = static_cast<float>(width);
        const auto heightFloat = static_cast<float>(height);

        const int maxRayDepth = customRenderer::getMaxRayDepth();

        Color3f col{0.0f};

        for(int subSamples = 0; subSamples < numSubsamples; ++subSamples){
            const float s = (iFloat + sampler.getSample1D()) / (widthFloat - 1);
            const float t = (jFloat + sampler.getSample1D()) / (heightFloat - 1);

            const auto ray = cam.getRay(s, t, sampler);

            col += getColor(ray, tlas, maxRayDepth, sampler);
        }


        col /= static_cast<float>(numSubsamples);

        output[pixelIndex] = col;

    }


}


