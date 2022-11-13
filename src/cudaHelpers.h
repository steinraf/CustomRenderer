//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <curand_kernel.h>

#include "utility/vector.h"
#include "utility/ray.h"
#include "hittableList.h"
#include "camera.h"
#include "utility/meshLoader.h"
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
        if (b < 0 || b > n - 1) return -1;

        unsigned int kb = c[b];
        if (ka == kb) {
            // if keys are equal, use id as fallback
            // (+32 because they have the same morton code and thus the string-concatenated XOR
            //  version would have 32 leading zeros)
            return 32 + __clz(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
        }
        // clz = count leading zeros
        return __clz(ka ^ kb);
    }

    __forceinline__ __device__ thrust::pair<int, int> determineRange(const uint32_t *mortonCodes, int numPrimitives, int i){
        const unsigned int *c = mortonCodes;
        unsigned int ki = c[i]; // key of i

        // determine direction of the range (+1 or -1)
        const int delta_l = delta(i, i - 1, numPrimitives, c, ki);
        const int delta_r = delta(i, i + 1, numPrimitives, c, ki);

        int d; // direction
        int delta_min; // min of delta_r and delta_l
        if (delta_r < delta_l) {
            d = -1;
            delta_min = delta_r;
        } else {
            d = 1;
            delta_min = delta_l;
        }

        // compute upper bound of the length of the range
        unsigned int l_max = 2;
        while (delta(i, i + l_max * d, numPrimitives, c, ki) > delta_min) {
            l_max <<= 1;
        }

        // find other end using binary search
        unsigned int l = 0;
        for (unsigned int t = l_max >> 1; t > 0; t >>= 1) {
            if (delta(i, i + (l + t) * d, numPrimitives, c, ki) > delta_min) {
                l += t;
            }
        }
        const int j = i + l * d;

//        printf("Stats of range are i=%i, j=%i, l=%i, d=%i\n", i, j, l, d);

        // ensure i <= j
        return {min(i, j), max(i, j)};
    }


    template<typename Primitive>
    __global__ void constructBVH(BVHNode<Primitive> *bvhNodes, Primitive *primitives, const uint32_t *mortonCodes, int numPrimitives){

        const int i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i > numPrimitives-1) return;

//        if(i == 0){
//            for(int tmp = 0; tmp < numPrimitives; ++tmp){
//                printf("morton %i: %u\n", tmp, mortonCodes[tmp]);
//            }
//        }


        // [0, numPrimitives-1]                     -> internal nodes
        // [numPrimitives, (2*numPrimitives)-1]     -> leaf nodes
        bvhNodes[numPrimitives + i - 1] = {
              nullptr,
              nullptr,
              &primitives[i],
              AABB{},
              true,
        };


//        printf("%p is leaf\n", &bvhNodes[numPrimitives + i - 1]);


        if(i == numPrimitives-1) return;


        auto[first, last] = determineRange(mortonCodes, numPrimitives, i);

        int split = findSplit(mortonCodes, first, last, numPrimitives);

//        printf("Constructing BVH %i (%i | %i | %i)...\n",i, first, split, last);


        auto childA = (split == first)  ? &bvhNodes[numPrimitives-1 + split]    : &bvhNodes[split];
        auto childB = (split+1 == last) ? &bvhNodes[numPrimitives-1 + split + 1]: &bvhNodes[split+1];

//        if(split == first){
//            printf("%i: Leaf node with index %i was taken.\n", i, numPrimitives-1 + split);
//        } else {
//            printf("%i: Internal node with index %i was taken.\n", i, split);
//        }

//        if(split+1 == last){
//            printf("%i: Leaf node with index %i was taken.\n", i, numPrimitives-1 + split+1);
//        } else {
//            printf("%i: Internal node with index %i was taken.\n", i, split+1  );
//        }

        bvhNodes[i] = {
                childA,
                childB,
                nullptr,
                AABB{},//nullptr, //                childA->boundingBox + childB ->boundingBox,
                false,
        };

//        printf("%p is interior\n", &bvhNodes[i]);


//        throw std::runtime_error("Must initialize all children before accessing their BoundingBoxes");


//        bvhInternalNodes, deviceTrianglePtr, deviceMortonCodes.data().get(), numTriangles-1
    }

    template<typename Primitive>
    __device__ const AABB &getBoundingBox(BVHNode<Primitive> *node){

//        printf("Started recursion... %p \n", node);



        if(node->isLeaf){
//            printf("Found Leaf...\n");
            assert(node->primitive);
            node->boundingBox = node->primitive->boundingBox;
        }else{
//            printf("Diverging Path...\n");
            assert(node->left && node->right);

            const AABB &leftAABB = getBoundingBox(node->left);
//            printf("Got left AABB\n");
            node->boundingBox = leftAABB + getBoundingBox(node->right);
//            printf("Completed Path...\n");
        }

        return node->boundingBox;

    }

    template<typename Primitive>
    __global__ void computeBVHBoundingBoxes(BVHNode<Primitive> *bvhNodes){
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        printf("Starting BVH BB Computation...\n");

        const AABB &totalBoundingBox = getBoundingBox(&bvhNodes[0]);

        printf("Total bounding box is (%f, %f, %f) -> (%f, %f, %f)\n",
               totalBoundingBox.min[0],  totalBoundingBox.min[1],  totalBoundingBox.min[2],
               totalBoundingBox.max[0],  totalBoundingBox.max[1],  totalBoundingBox.max[2]);


    }


    template<typename Primitive>
    __global__ void initBVH(BVH<Primitive> *bvh, Primitive *primitives, int numPrimitives){
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

//        for(int i = 0; i < numPrimitives; ++i)
//            printf("Receiving GPU Tria with coordinates\n"
//                   "(%f, %f, %f) \n"
//                   "(%f, %f, %f) \n"
//                   "(%f, %f, %f) \n",
//                   primitives[i].p0[0], primitives[i].p0[1], primitives[i].p0[2],
//                   primitives[i].p1[0], primitives[i].p1[1], primitives[i].p1[2],
//                   primitives[i].p2[0], primitives[i].p2[1], primitives[i].p2[2]);

        *bvh = BVH<Primitive>(primitives, numPrimitives);
    }

    __global__ void freeVariables(int width, int height);


    template<typename Primitive>
    __device__ Color getColor(const Ray &r, BVH<Primitive> *bvh, int maxRayDepth, Sampler &sampler){

        HitRecord record;

        Ray currentRay = r;

        Ray scattered;
        Color attenuation;

        Color currentAttenuation{1.f};

        for(int depth = 0; depth < maxRayDepth; ++depth){
            if(bvh->hit(currentRay, 1e-8, cuda::std::numeric_limits<float>::infinity(), record)){
                if(record.triangle->bsdf.scatter(currentRay, record, attenuation, scattered, sampler)){
                    currentRay = scattered;
                    currentAttenuation *= attenuation;
                    return record.normal * 255.99;
                }else{
                    return Color{0.f};
                }

            }else{
//                return Color{1.f};
                float t = 0.5f * (r.getDirection().normalized()[1] + 1.f);
                Color c = (1 - t) * Vector3f{1.f} + t * Color{0.5f, 0.7f, 1.0f};
                return currentAttenuation * c;
            }
        }

        return Vector3f{0.f};

    }

    __global__ void denoise(Vector3f *input, Vector3f *output, int width, int height);

    template<typename Primitive>
    __global__ void render(Vector3f *output, Camera cam, BVH<Primitive> *bvh, int width, int height,
                           curandState *globalRandState){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;


        auto sampler = Sampler(&globalRandState[pixelIndex]);

        const auto iFloat = static_cast<float>(i);
        const auto jFloat = static_cast<float>(j);

        const auto widthFloat = static_cast<float>(width);
        const auto heightFloat = static_cast<float>(height);

        const int maxRayDepth = customRenderer::getMaxRayDepth();
        const int numSubsamples = customRenderer::getNumSubsamples();

        Color col{0.0f};

        for(int subSamples = 0; subSamples < numSubsamples; ++subSamples){
            const float s = (iFloat + sampler.getSample1D()) / (widthFloat - 1);
            const float t = (jFloat + sampler.getSample1D()) / (heightFloat - 1);

            const auto ray = cam.getRay(s, t, sampler);

            col += getColor(ray, bvh, maxRayDepth, sampler);
        }

        constexpr float scale = 1.f / numSubsamples;

        col = {
                sqrt(col[0] * scale),
                sqrt(col[1] * scale),
                sqrt(col[2] * scale)
        };

        output[pixelIndex] = col;

    }





};


