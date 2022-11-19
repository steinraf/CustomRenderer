//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <curand_kernel.h>

#include "utility/warp.h"
//#include "utility/vector.h"
#include "utility/ray.h"
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
                (childA->isLeaf && childB->isLeaf) ? childA->boundingBox + childB->boundingBox : AABB{},
                false,
        };

    }

    template<typename Primitive>
    __device__ AABB getBoundingBox(AccelerationNode<Primitive> *root, int indentLevel = 1){

        typedef AccelerationNode<Primitive> *NodePtr;

        constexpr int stackSize = 1024;
        NodePtr stack[stackSize];
        int idx = 0;
        stack[0] = root;

        assert(root);

        NodePtr currentNode = root;

        do{

//            printf("STACK: "); for(int tmp = 0; tmp < idx+1; ++tmp) printf("| %p ", stack[tmp]);
//            printf("\n");

            assert(idx < stackSize);

            currentNode = stack[idx];

            NodePtr left = currentNode->left;
            NodePtr right = currentNode->right;

            assert(left && right);

            if(left->hasBoundingBox() && right->hasBoundingBox()){
//                for(int tmp = 0; tmp < idx; ++tmp) printf("\t");
//                printf("oB %p\n", currentNode);
                assert(!left->boundingBox.isEmpty() && !right->boundingBox.isEmpty());
                currentNode->boundingBox = left->boundingBox + right->boundingBox;
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

//        do {
//            assert(!currentNode->isLeaf);

        NodePtr left = currentNode->left;
        NodePtr right = currentNode->right;

        if(left && right){
            if(left->hasBoundingBox()){
                if(right->hasBoundingBox()){
                    for(int tmp = 0; tmp < idx - 1; ++tmp) printf("\t");
                    printf("Pop : Both filled\n");
                    currentNode->boundingBox = left->boundingBox + right->boundingBox;
                    currentNode = stack[--idx];
                }else{
                    for(int tmp = 0; tmp < idx + 1; ++tmp) printf("\t");
                    printf("Push: Right No BB\n");
                    stack[++idx] = right;
                    currentNode = right;
                }
            }else if(right->hasBoundingBox()){
                for(int tmp = 0; tmp < idx + 1; ++tmp) printf("\t");
                printf("Push: Left No BB\n");
                stack[++idx] = left;
                currentNode = left;
            }else{
                for(int tmp = 0; tmp < idx + 1; ++tmp) printf("\t");
                printf("Push: Neither BB\n");
                stack[++idx] = currentNode;
//                    stack[++idx] = right;
                currentNode = left;
            }
        }else if(left){
            if(left->hasBoundingBox()){
                for(int tmp = 0; tmp < idx - 1; ++tmp) printf("\t");
                printf("Pop : Right Nullptr\n");
                currentNode->boundingBox = left->boundingBox;
                currentNode = stack[--idx];
            }else{
                for(int tmp = 0; tmp < idx + 1; ++tmp) printf("\t");
                printf("Push: Right Nullptr\n");
                stack[++idx] = left;
                currentNode = left;
            }
        }else if(right){
            if(right->hasBoundingBox()){
                for(int tmp = 0; tmp < idx - 1; ++tmp) printf("\t");
                printf("Pop : Left Nullptr\n");
                currentNode->boundingBox = right->boundingBox;
                currentNode = stack[--idx];
            }else{
                for(int tmp = 0; tmp < idx + 1; ++tmp) printf("\t");
                printf("Push: Left Nullptr\n");
                stack[++idx] = right;
                currentNode = right;
            }
        }else{
            assert(!currentNode->isLeaf);
            assert(false && "Right and right are nullptr and this is not a leaf");
            currentNode = stack[--idx];
        }




//
//            if(!currentNode->right->isLeaf && !currentNode->right->isLeaf){
//                if(currentNode->right->boundingBox.isEmpty()){
//                    for(int tmp = 0; tmp < idx+1; ++tmp) printf("\t");
//                    printf("Push: Left  empty\n");
//                    stack[++idx] = currentNode;
//                    currentNode = currentNode->right;
//                } else if(currentNode->right->boundingBox.isEmpty()){
//                    for(int tmp = 0; tmp < idx+1; ++tmp) printf("\t");
//                    printf("Push: Right empty\n");
//                    stack[++idx] = currentNode;
//                    currentNode = currentNode->right;
//                } else{
//                    for(int tmp = 0; tmp < idx-1; ++tmp) printf("\t");
//                    printf("Pop : Both  full\n");
//                    currentNode->boundingBox = currentNode->right->boundingBox + currentNode->right->boundingBox;
//                    currentNode = stack[--idx];
//                }
//            } else if (currentNode->right->isLeaf && !currentNode->right->isLeaf){
//                for(int tmp = 0; tmp < idx+1; ++tmp) printf("\t");
//                printf("Push: Left  Leaf\n");
//                stack[++idx] = currentNode;
//                currentNode = currentNode->right;
//            } else if (!currentNode->right->isLeaf &&currentNode->right->isLeaf){
//                for(int tmp = 0; tmp < idx+1; ++tmp) printf("\t");
//                printf("Push: Right Leaf\n");
//                stack[++idx] = currentNode;
//                currentNode = currentNode->right;
//            } else {
//                for(int tmp = 0; tmp < idx-1; ++tmp) printf("\t");
//                printf("Pop : Both  Leaf\n");
//                currentNode->boundingBox = currentNode->right->boundingBox + currentNode->right->boundingBox;
//                currentNode = stack[--idx];
//            }
//            assert(idx < stackSize);
//        } while(idx >= 0);
//
//        assert(root);
//
//        printf("Started recursion... %p \n", root);
//
//
//        if(root->isLeaf){
//            printf("Found Leaf...\n");
//            assert(root->primitive);
//            root->boundingBox = root->primitive->boundingBox;
//            printf("Got Leaf AABB %f\n", root->boundingBox.min[0]);
//        }else{
////            printf("Diverging Path...\n");
//            assert(root->right && root->right);
//
//            printf("Diverging Path %p...\n", root);
//
//            AABB leftAABB = getBoundingBox(root->right);
//
//            printf("Got right AABB %f\n", leftAABB.min[0]);
//            AABB rightAABB = getBoundingBox(root->right);
//
//
//            root->boundingBox = leftAABB + rightAABB;
//            printf("Completed Path...\n");
//        }


//        printf("Computed BB (%f, %f, %f) -> (%f, %f, %f)",
//               root->boundingBox.min[0], root->boundingBox.min[1], root->boundingBox.min[2],
//               root->boundingBox.max[0], root->boundingBox.max[1], root->boundingBox.max[2]);

        return root->boundingBox;

    }

    template<typename Primitive>
    __global__ void computeBVHBoundingBoxes(AccelerationNode<Primitive> *bvhNodes, int numPrimitives){
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

//        for(int tmp = 0; tmp < 2*numPrimitives-1; ++tmp){
//            printf("%i, (%p, %p, %p, %p, %d)\n", tmp, bvhNodes + tmp, bvhNodes[tmp].right, bvhNodes[tmp].right, bvhNodes[tmp].primitive, bvhNodes[tmp].isLeaf);
//        }

//        printf("Starting BLAS BB Computation...\n");


//        printf("Root %p -> %p -> %p \n", bvhNodes, bvhNodes + 544566 - 1, bvhNodes + (2 * 544566 - 1));
        const AABB &totalBoundingBox = getBoundingBox(&bvhNodes[0]);

//        printf("Total bounding box is (%f, %f, %f) -> (%f, %f, %f)\n",
//               totalBoundingBox.min[0], totalBoundingBox.min[1], totalBoundingBox.min[2],
//               totalBoundingBox.max[0], totalBoundingBox.max[1], totalBoundingBox.max[2]);


#ifdef NDEBUG
        for(int idx = numPrimitives - 1; idx < 2 * numPrimitives - 1; ++idx){
            assert(bvhNodes[idx].isPointedTo);
        }
#endif
    }

    template<typename Primitive>
    __global__ void initBVH(BLAS<Primitive> *bvh, AccelerationNode<Primitive> *bvhTotalNodes){
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        *bvh = BLAS<Primitive>(bvhTotalNodes);
    }

    __global__ void freeVariables(int width, int height);


    template<typename Primitive>
    __device__ Color3f getColor(const Ray &r, BLAS<Primitive> *bvh, int maxRayDepth, Sampler &sampler){

        HitRecord record;

        if(!bvh->hit(r, record))
            return Color3f{0.f};

        return record.normal.absValues();

//        Ray scattered;
//        Color3f attenuation;
//
//        Color3f currentAttenuation{1.f};
//
//        for(int depth = 0; depth < maxRayDepth; ++depth){
//            if(bvh->hit(currentRay, FLT_EPSILON, cuda::std::numeric_limits<float>::infinity(), record)){
//                if(record.triangle->bsdf.scatter(currentRay, record, attenuation, scattered, sampler)){
//                    currentRay = scattered;
////                    currentAttenuation *= attenuation;
////                    return record.normal;// * 255.99;
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

        if(!bvh->hit(r, record))
            return Color3f{1.f};

        const Vector3f scatter = Warp::sampleUniformHemisphere(sampler, record.normal);



        if(!bvh->hit({record.position, scatter, FLT_EPSILON, 10}, record))
            return Color3f{1.f};



        return Color3f{0.f};

    }

    __global__ void denoise(Vector3f *input, Vector3f *output, int width, int height);

    template<typename Primitive>
    __global__ void render(Vector3f *output, Camera cam, BLAS<Primitive> *bvh, int width, int height,
                           curandState *globalRandState){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

//        if(i == 0 && j == 0){
//            printf("Testing Hit: \n");
//
//            auto r = Ray(customRenderer::getCameraOrigin(), customRenderer::getCameraLookAt() - customRenderer::getCameraOrigin());
//            HitRecord h;
//            bool didHit = bvh->hit(r, FLT_EPSILON, INFINITY, h);
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
        const int numSubsamples = customRenderer::getNumSubsamples();

        Color3f col{0.0f};

        for(int subSamples = 0; subSamples < numSubsamples; ++subSamples){
            const float s = (iFloat + sampler.getSample1D()) / (widthFloat - 1);
            const float t = (jFloat + sampler.getSample1D()) / (heightFloat - 1);

            const auto ray = cam.getRay(s, t, sampler);

            col += getColor(ray, bvh, maxRayDepth, sampler);
        }

//        constexpr float scale = 1.f ;

//        col = {
//                sqrt(col[0] * scale),
//                sqrt(col[1] * scale),
//                sqrt(col[2] * scale)
//        };

        col /= numSubsamples;

        output[pixelIndex] = col;

    }


}


