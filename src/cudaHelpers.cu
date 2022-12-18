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
                            size_t numPrimitives, AreaLight *emitter, BSDF bsdf, Texture normalMap) {
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        *bvh = BLAS{bvhTotalNodes, totalArea, cdf, numPrimitives, emitter, bsdf, normalMap};
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

            if(left->hasBoundingBox() && right->hasBoundingBox()) {
                //                for(int tmp = 0; tmp < idx; ++tmp) printf("\t");
                //                printf("oB %p\n", currentNode);
                assert(!left->boundingBox.isEmpty() && !right->boundingBox.isEmpty());
                currentNode->boundingBox = left->boundingBox + right->boundingBox;
                //                printf("New bounding box (%f, %f, %f) -> (%f, %f, %f)\n",
                //                       currentNode->boundingBox.min[0], currentNode->boundingBox.min[1], currentNode->boundingBox.min[2],
                //                       currentNode->boundingBox.max[0], currentNode->boundingBox.max[1], currentNode->boundingBox.max[2]);
                --idx;
            } else if(right->hasBoundingBox()) {
                //                for(int tmp = 0; tmp < idx; ++tmp) printf("\t");
                //                printf("Le %p\n", currentNode);
                stack[++idx] = left;
            } else if(left->hasBoundingBox()) {
                //                for(int tmp = 0; tmp < idx; ++tmp) printf("\t");
                //                printf("Ri %p\n", currentNode);
                stack[++idx] = right;
            } else {
                //                for(int tmp = 0; tmp < idx; ++tmp) printf("\t");
                //                printf("Bo %p\n", currentNode);

                stack[++idx] = right;
                stack[++idx] = left;
            }
        } while(idx >= 0);

        return root->boundingBox;
    }

    __global__ void computeBVHBoundingBoxes(AccelerationNode *bvhNodes) {
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        //        printf("Starting BLAS BB Computation...\n");


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

    __device__ void bilateralFilterWiki(Vector3f *input, Vector3f *output, int i, int j, int width, int height){

        auto getNeighbour = [i, j, width, height]__device__ (Vector3f *array, int dx, int dy,
                                                             BOUNDARY boundary = BOUNDARY::PERIODIC) {
            switch(boundary) {
                case BOUNDARY::PERIODIC:
                    return array[(j + height + dy) % height * width + (i + width + dx) % width];
                case BOUNDARY::REFLECTING:
                    //TODO implement?
                    assert(false && "Not implemented.");
                    //                    break;
                case BOUNDARY::ZERO:
                    if(i >= 0 && i < width && j >= 0 && j < height)
                        return array[j*width + i];

                    return Vector3f{0.f};
            }
        };

        const int pixelIndex = j*width + i;

        const int kernelWidth = width/200, kernelHeight = height/200;

        float sum = 0.f;

        Color3f currentColor{0.f};

        float dVarInv = 1.f/(80*80);
        float rVarInv = 1.f/(80*80);

        for(int k = -kernelWidth/2; k < kernelWidth/2; ++k){
            for(int l = -kernelHeight/2; l < kernelHeight/2; ++l){
                const Color3f neighbourColor = getNeighbour(input, k, l);

                const float w = expf(
                        -(k*k + l*l)*0.5f*dVarInv
                        -(input[pixelIndex] - neighbourColor).squaredNorm()*0.5f*rVarInv
                    );
//                const float w = -(input[pixelIndex] - neighbourColor).squaredNorm();
                sum += w;
                currentColor += w*neighbourColor;
            }
        }

//        printf("Sum of weights is %f\n", sum);

        output[pixelIndex] = currentColor/sum;


    }

    __device__ void bilateralFilterSlides(Vector3f *input, Vector3f *output, FeatureBuffer &featureBuffer, float *weights, int i, int j, int width, int height){


//        constexpr int neighbourDiameter = 21;
//        constexpr int patchDiameter = 7;

        constexpr int neighbourDiameter = 11;
        constexpr int patchDiameter = 1;

        constexpr float k = 0.45f;

        const int pixelI = i, pixelJ = j;


        for(int pixelQI = CustomRenderer::max(0, pixelI - neighbourDiameter /2); pixelQI < CustomRenderer::min(width, pixelI + neighbourDiameter /2 + 1); ++pixelQI){
            for(int pixelQJ = CustomRenderer::max(0, pixelJ - neighbourDiameter /2); pixelQJ < CustomRenderer::min(height, pixelJ + neighbourDiameter /2 + 1); ++pixelQJ) {
//                if(pixelQI == pixelI && pixelQJ == pixelJ) continue;
                float meanDist = 0.f;
                for(int pI = CustomRenderer::max(0, pixelI - patchDiameter / 2); pI < CustomRenderer::min(width, pixelI + patchDiameter / 2 + 1); ++pI) {
                    for(int pJ = CustomRenderer::max(0, pixelJ - patchDiameter / 2); pJ < CustomRenderer::min(height, pixelJ + patchDiameter / 2 + 1); ++pJ) {
                        const int pIndex = pJ * width + pI;
                        const Vector3f pVarianceMean = featureBuffer.variances[pIndex]/static_cast<float>(featureBuffer.numSubSamples[pIndex]);

                        for(int qI = CustomRenderer::max(0, pixelQI - patchDiameter / 2); qI < CustomRenderer::min(width, pixelQI + patchDiameter / 2 + 1); ++qI) {
                            for(int qJ = CustomRenderer::max(0, pixelQJ - patchDiameter / 2); qJ < CustomRenderer::min(height, pixelQJ + patchDiameter / 2 + 1); ++qJ) {
//                                if(qI == pI && qJ == pJ) continue;
                                const int qIndex = qJ * width + qI;

                                const Vector3f qVarianceMean = featureBuffer.variances[qIndex]/static_cast<float>(featureBuffer.numSubSamples[qIndex]);


//                                if(k * k * (pVarianceMean +  qVarianceMean).squaredNorm() < FLT_EPSILON)
//                                    printf("Variance sum very small %f\n", (pVarianceMean +  qVarianceMean).squaredNorm());

                                for(int col = 0; col < 3; ++col)
                                    meanDist += (powf(input[pIndex][col] - input[qIndex][col], 2) - (  pVarianceMean[col] + CustomRenderer::min(pVarianceMean[col], qVarianceMean[col]))) / (EPSILON + k * k * (pVarianceMean[col] +  qVarianceMean[col]));
                                //meanDist .= ((input[pI] - input[qI])^2 - (var[pI] + min(var[pI], var[qI])
                                //            --------------------------------------------------------
                                //              (EPSILON + k^2 * (var[pI] + var[qI]))
                            }
                        }
                    }
                }

//                printf("Mean distance is %f\n", meanDist/(3 * patchDiameter * patchDiameter));

                float w = expf(-CustomRenderer::max(0.f, meanDist/(3 * patchDiameter * patchDiameter)));
//                w = CustomRenderer::max(EPSILON*EPSILON, w);
//
//                if(w < EPSILON){
//#ifndef NDEBUG
//                    printf("W too small: %f -> %f, %f\n", w, meanDist, meanDist/(3 * patchDiameter * patchDiameter));
//#endif
//                    w = EPSILON;
////                    assert(false);
//                }

                for(int pI = CustomRenderer::max(0, pixelI - patchDiameter / 2); pI < CustomRenderer::min(width, pixelI + patchDiameter / 2 + 1); ++pI) {
                    for(int pJ = CustomRenderer::max(0, pixelJ - patchDiameter / 2); pJ < CustomRenderer::min(height, pixelJ + patchDiameter / 2 + 1); ++pJ) {

                        const int pIndex = pJ * width + pI;

                        for(int qI = CustomRenderer::max(0, pixelQI - patchDiameter / 2); qI < CustomRenderer::min(width, pixelQI + patchDiameter / 2 + 1); ++qI) {
                            for(int qJ = CustomRenderer::max(0, pixelQJ - patchDiameter / 2); qJ < CustomRenderer::min(height, pixelQJ + patchDiameter / 2 + 1); ++qJ) {

                                const int qIndex = qJ * width + qI;

                                atomicAdd(weights + pIndex, w);
                                Vector3f::atomicCudaAdd(output + pIndex, w * input[qIndex]);
                            }
                        }
                    }
                }
            }
        }
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

    __global__ void denoiseApplyWeights(Vector3f *output, float *weights, int width, int height){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;
//#ifndef NDEBUG
//        if(abs(weights[pixelIndex]) < EPSILON)
//            printf("Denoising weights are smaller than epsilon...\n");
//#endif
//        if(weights[pixelIndex] == 0.f) {
//            printf("Output would have been (%f, %f, %f)\n", output[pixelIndex][0], output[pixelIndex][1], output[pixelIndex][2]);
//            output[pixelIndex] = Vector3f{0.f, 0.f, 1.f};
//        }
//        output[pixelIndex] /= weights[pixelIndex];
    }


    __global__ void denoise(Vector3f *input, Vector3f *output, FeatureBuffer featureBuffer, float *weights, int width, int height,
                            Vector3f cameraOrigin) {
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

//        bilateralFilterWiki(input, output, i, j, width, height);
//        bilateralFilterSlides(input, output, featureBuffer, weights, i, j, width, height);
//        return;

//        auto getNeighbour = [i, j, width, height]__device__ (Vector3f *array, int dx, int dy,
//                                                             BOUNDARY boundary = BOUNDARY::PERIODIC) {
//            switch(boundary) {
//                case BOUNDARY::PERIODIC:
//                    return array[(j + height + dy) % height * width + (i + width + dx) % width];
//                case BOUNDARY::REFLECTING:
//                    //TODO implement?
//                    assert(false && "Not implemented.");
////                    break;
//                case BOUNDARY::ZERO:
//                    if(i >= 0 && i < width && j >= 0 && j < height)
//                        return array[j*width + i];
//
//                    return Vector3f{0.f};
//            }
//        };

//        const int m_radius = 2;
//        const float m_stddev = 0.5f;
//
//        const float alpha = -1.0f / (2.0f * m_stddev * m_stddev);
//        const float constant = std::exp(alpha * m_radius * m_radius);


//        auto gaussian = [alpha, constant] __device__ (float x){
//            return CustomRenderer::max(0.0f, std::exp(alpha * x * x) - constant);
//        };

//        float integral = 0.f;
//        Vector3f tmp{0.f};

//        for(int xNew = -m_radius; xNew <= m_radius; ++xNew) {
//            for(int yNew = -m_radius + 1; yNew < m_radius; ++yNew){
//                const float gaussianContrib = gaussian(sqrtf(xNew*xNew + yNew*yNew));
//                tmp += gaussianContrib * getNeighbour(input, xNew, yNew);
//                integral += gaussianContrib;
//            }
//        }
//
//        output[pixelIndex] = tmp/integral;

        //TODO when modifying something here, remember to maybe uncomment weights application

        //        output[pixelIndex] = Vector3f{(featureBuffer[pixelIndex].position-cameraOrigin).norm()/200};
                output[pixelIndex] = featureBuffer.normals[pixelIndex].absValues();
//                output[pixelIndex] = featureBuffer.albedos[pixelIndex];
//        output[pixelIndex] = featureBuffer.variances[pixelIndex];
//        output[pixelIndex] = Color3f(featureBuffer.variances[pixelIndex].norm());
//        constexpr float numSamples = 512.f;
//        output[pixelIndex] = Vector3f{powf(static_cast<float>(featureBuffer.numSubSamples[pixelIndex])/(numSamples), 2.f)};


//        if(featureBuffer[pixelIndex].variance.maxCoeff() > 0.1) {
//            output[pixelIndex] = 0.25 * getNeighbour(input, 0, -1) + 0.25 * getNeighbour(input, 1, 0) + 0.25 * getNeighbour(input, 0, 1) + 0.25 * getNeighbour(input, -1, 0);
//        }
//        else {
//            output[pixelIndex] = input[pixelIndex];
//        }

        //        output[pixelIndex] = Color3f(featureBuffer[pixelIndex].variance.norm());
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

//            //SampleVisualization remember to set xml scene size to dimension
//            EmitterQueryRecord eQR{Vector3f{0.f}};
////            if(i % 3 != 0 || j % 3 != 0) return;
//            auto sample = tlas->environmentEmitter.sample(eQR, sampler.getSample3D());
//            Vector3f::atomicCudaAdd(output + static_cast<int>((eQR.uv[1]*height)*width + eQR.uv[0]*width), /* tlas->environmentEmitter.pdf(eQR)/(width * height) * */ Vector3f{1.f, 1.f, 1.f});
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


            //Adaptive sampling
//            if(!updatedVariance && subSamples > 64 && (m2 / static_cast<float>((subSamples - 1))).maxCoeff() < EPSILON) {
//                atomicAdd(counter, 1);
//                updatedVariance = true;
//            }
//            if(counter[0] >= blockDim.x*blockDim.y){
//                actualSamples = subSamples;
//                break;
//            }
        }


        atomicAdd(progressCounter, 1);
        if(*progressCounter % 1000 == 0) {
            const float progress = 100.f * (*progressCounter) / (width * height);
            printf("Current progress is %f%\r", progress);

            //            const float spentTime = ((double) (clock() - renderStart)) / CLOCKS_PER_SEC;
            //
            //            printf("Estimated Time left: %f (%f, %f) s\n", 100*progress/spentTime, progress, spentTime);
        }

        //        const Vector3f biasedVariance = m2 / static_cast<float>(actualSamples);
        const Vector3f unbiasedVarianceMean = m2 / (static_cast<float>(actualSamples - 1));



        totalColor /= static_cast<float>(actualSamples);

//        featureBuffer.variances[pixelIndex] = tmpBuffer.color;
        featureBuffer.albedos[pixelIndex] = tmpBuffer.albedo;
        featureBuffer.normals[pixelIndex] = tmpBuffer.normal;
        featureBuffer.positions[pixelIndex] = tmpBuffer.position;
        featureBuffer.numSubSamples[pixelIndex] = tmpBuffer.numSubSample;


        output[pixelIndex] = totalColor;

        featureBuffer.variances[pixelIndex] = unbiasedVarianceMean;
        featureBuffer.numSubSamples[pixelIndex] = actualSamples;
    }

}// namespace cudaHelpers
