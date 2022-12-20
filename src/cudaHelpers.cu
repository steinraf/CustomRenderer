//
// Created by steinraf on 19/08/22.
//

#include "cudaHelpers.h"


#include "bsdf.h"
#include "utility/ray.h"


#include <iostream>


namespace cudaHelpers {


    __host__ void
    check_cuda(cudaError_t result, char const *const func, const char *const file, int line) noexcept(false) {
        if(result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
            cudaDeviceReset();
            exit(99);
        }
    }


    __global__ void initRng(int width, int height, curandState *randState) {
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

        curand_init(42, pixelIndex, 0, &randState[pixelIndex]);
    }

    __global__ void initBVH(BLAS *bvh, AccelerationNode *bvhTotalNodes, float totalArea, const float *cdf,
                            size_t numPrimitives, AreaLight *emitter, BSDF bsdf, Texture normalMap, GalaxyMedium *medium) {
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        *bvh = BLAS{bvhTotalNodes, totalArea, cdf, numPrimitives, emitter, bsdf, normalMap, medium};
    }


    __global__ void freeVariables() {
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, 1, 1)) return;
    }

    enum class BOUNDARY {
        PERIODIC,
        REFLECTING,
        ZERO
    };

    __global__ void constructBVH(AccelerationNode *bvhNodes, Triangle *primitives, const uint32_t *mortonCodes, int numPrimitives) {

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

        AccelerationNode *childA = (split == first) ? &bvhNodes[numPrimitives - 1 + split]
                                                    : &bvhNodes[split];
        AccelerationNode *childB = (split + 1 == last) ? &bvhNodes[numPrimitives - 1 + split + 1]
                                                       : &bvhNodes[split +
                                                                   1];

        bvhNodes[i] = {
                childA,
                childB,
                nullptr,
                AABB{},
                false,
        };

    }

    __device__ AABB getBoundingBox(AccelerationNode *root) noexcept {

        typedef AccelerationNode *NodePtr;

        constexpr int stackSize = 1024;
        NodePtr stack[stackSize];
        int idx = 0;
        stack[0] = root;

        assert(root);

        NodePtr currentNode;

        do {

            assert(idx < stackSize);

            currentNode = stack[idx];

            NodePtr left = currentNode->left;
            NodePtr right = currentNode->right;

            assert(left && right);

            if(left->hasBoundingBox() && right->hasBoundingBox()) {
                assert(!left->boundingBox.isEmpty() && !right->boundingBox.isEmpty());
                currentNode->boundingBox = left->boundingBox + right->boundingBox;
                --idx;
            } else if(right->hasBoundingBox()) {
                stack[++idx] = left;
            } else if(left->hasBoundingBox()) {
                stack[++idx] = right;
            } else {
                stack[++idx] = right;
                stack[++idx] = left;
            }
        } while(idx >= 0);

        return root->boundingBox;
    }

    __global__ void computeBVHBoundingBoxes(AccelerationNode *bvhNodes) {
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        const AABB &totalBoundingBox = getBoundingBox(&bvhNodes[0]);

        printf("\tTotal bounding box is (%f, %f, %f) -> (%f, %f, %f)\n",
               totalBoundingBox.min[0], totalBoundingBox.min[1], totalBoundingBox.min[2],
               totalBoundingBox.max[0], totalBoundingBox.max[1], totalBoundingBox.max[2]);
    }

    __global__ void constructTLAS(TLAS *tlas,
                                  BLAS **meshBlasArr, size_t numMeshes,
                                  BLAS **emitterBlasArr, size_t numEmitters,
                                  EnvironmentEmitter environmentEmitter) {

        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        *tlas = TLAS(meshBlasArr, numMeshes, emitterBlasArr, numEmitters, environmentEmitter);
    }



    __global__ void applyGaussian(Vector3f *input, Vector3f *output, int width, int height, float sigma, int windowRadius){

        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

        const float alpha = -1.0f / (2.0f * sigma * sigma);
        const float constant = 0.f;//std::exp(alpha * windowRadius * windowRadius);


        auto gaussian = [alpha, constant] __device__ (float x){
            return CustomRenderer::max(0.0f, std::exp(alpha * x * x) - constant);
        };

        float integral = 0.f;
        Vector3f tmp{0.f};

        for(int xNew = -windowRadius; xNew <= windowRadius; ++xNew) {
            if(i + xNew < 0 || i + xNew >= width) continue;

            for(int yNew = -windowRadius; yNew <= windowRadius; ++yNew){
                if(j + yNew < 0 || j + yNew >= height) continue;

                const float gaussianContrib = gaussian(sqrtf(xNew*xNew + yNew*yNew));
                tmp += gaussianContrib * input[(j+yNew)*width + i+xNew];
                integral += gaussianContrib;
            }
        }

        output[pixelIndex] = tmp/integral;

    }




    __device__ int findSplit(const uint32_t *mortonCodes, int first, int last, int numPrimitives) {

        const unsigned int first_code = mortonCodes[first];

        // calculate the number of highest bits that are the same
        // for all objects, using the count-leading-zeros intrinsic

        const int common_prefix =
                delta(first, last, numPrimitives, mortonCodes, first_code);

        // use binary search to find where the next bit differs
        // specifically, we are looking for the highest object that
        // shares more than commonPrefix bits with the first one

        int split = first;// initial guess
        int step = last - first;

        do {
            step = (step + 1) >> 1;            // exponential decrease
            const int new_split = split + step;// proposed new p

            if(new_split < last) {
                const int split_prefix = delta(
                        first, new_split, numPrimitives, mortonCodes, first_code);
                if(split_prefix > common_prefix) {
                    split = new_split;// accept proposal
                }
            }
        } while(step > 1);

        return split;
    }

    __global__ void render(Vector3f *output, Camera cam, TLAS *tlas, int width, int height, int numSubsamples,
                           int maxRayDepth, curandState *globalRandState, FeatureBuffer featureBuffer, unsigned *progressCounter) {
        int i, j, pixelIndex;
        bool inBounds = initIndices(i, j, pixelIndex, width, height);

        __shared__ int counter[1];

        if(threadIdx.x == 0 && threadIdx.y == 0)
            counter[0] = 0;

        __syncthreads();

        if(!inBounds) {
            atomicAdd(counter, 1);
            return;
        }
        bool updatedVariance = false;

        auto sampler = Sampler(&globalRandState[pixelIndex]);

        const auto iFloat = static_cast<float>(i);
        const auto jFloat = static_cast<float>(j);

        const auto widthFloat = static_cast<float>(width);
        const auto heightFloat = static_cast<float>(height);

        Color3f totalColor{0.0f};

        //welfords algorithm to compute variance
        Color3f mean{0.f}, m2{0.f};

        FeatureBufferAccumulator tmpBuffer;
        //TODO add variance computations

        int actualSamples = numSubsamples;

        for(int subSamples = 0; subSamples < numSubsamples; ++subSamples) {

            const float s = (iFloat + sampler.getSample1D()) / (widthFloat + 1);
            const float t = (jFloat + sampler.getSample1D()) / (heightFloat + 1);

////            //SampleVisualization remember to set xml scene size to dimension
//            EmitterQueryRecord eQR{Vector3f{0.f}};
//            if(i % 3 != 0 || j % 3 != 0) return;
//            auto sample = tlas->environmentEmitter.sample(eQR, sampler.getSample3D());
//            const int pixelCoord = static_cast<int>((eQR.uv[1]*height)*width + eQR.uv[0]*width);
//            Vector3f::atomicCudaAdd(output + pixelCoord,  tlas->environmentEmitter.pdf(eQR) * width * height  * Vector3f{1.f, 1.f, 1.f});
//            return;

//            //Texture CDF
//            const Texture &texture = tlas->environmentEmitter.texture;
////            printf("Index is %f\n", static_cast<int>(t*height * width + s*width)*1.f/width/height);
//            const float cdfRowStart = texture.deviceCDF[static_cast<int>(jFloat * texture.width)];
//            const float cdfColPrev = [&](){
//                if(j == 0) return 0.f;
//                return texture.deviceCDF[static_cast<int>((jFloat-1) * texture.width + iFloat)];
//            }();
//            const float cdf = texture.deviceCDF[static_cast<int>(jFloat * texture.width + iFloat)];
//            const float cdfRowEnd = texture.deviceCDF[static_cast<int>(jFloat * texture.width + width - 1)];
//            const float cdfColNext = [&](){
//                if(j == height-1) return 1.f;
//                return texture.deviceCDF[static_cast<int>((jFloat+1) * texture.width + iFloat)];
//            }();
//            if(sampler.getSample1D() < 0.01f)
//                printf("CDF (%f, %f, %i)->%f\n", s, t, static_cast<int>(jFloat*texture.width + iFloat), cdf);
//            output[pixelIndex] = Color3f{0.f*(cdf-cdfColPrev)/(cdfColNext-cdfColPrev), (cdf-cdfRowStart)/(cdfRowEnd - cdfRowStart), 0};
//            return;

            const auto ray = cam.getRay(s, t, sampler.getSample2D());


            const Vector3f currentColor = getColor(ray, tlas, maxRayDepth, sampler, tmpBuffer, pixelIndex);


            const Vector3f delta = currentColor - mean;
            mean += delta / static_cast<float>(subSamples + 1);
            const Vector3f delta2 = currentColor - mean;
            m2 += delta * delta2;

            totalColor += currentColor;

            //TODO redirect samples to pixels that need it
//            assert(false);
            //Adaptive sampling
            if(!updatedVariance && subSamples > 64 && (m2 / static_cast<float>((subSamples - 1))).maxCoeff() < EPSILON) {
                atomicAdd(counter, 1);
                updatedVariance = true;
            }
            if(counter[0] >= blockDim.x*blockDim.y){
                assert(updatedVariance);
                actualSamples = subSamples;
                break;
            }
        }


        atomicAdd(progressCounter, 1);
        if(*progressCounter % 1000 == 0) {
            const float progress = 100.f * (*progressCounter) / (width * height);
            printf("Current progress is %f% \r", progress);
        }

        const Vector3f unbiasedVarianceMean = m2 / (static_cast<float>(actualSamples - 1));



        totalColor /= static_cast<float>(actualSamples);

        featureBuffer.albedos[pixelIndex] = tmpBuffer.albedo;
        featureBuffer.normals[pixelIndex] = tmpBuffer.normal;
        featureBuffer.positions[pixelIndex] = tmpBuffer.position;
        featureBuffer.numSubSamples[pixelIndex] = tmpBuffer.numSubSample;


        output[pixelIndex] = totalColor;

        featureBuffer.variances[pixelIndex] = unbiasedVarianceMean;
        featureBuffer.numSubSamples[pixelIndex] = actualSamples;
    }

}// namespace cudaHelpers
