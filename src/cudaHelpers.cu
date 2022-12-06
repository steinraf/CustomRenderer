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


    __global__ void freeVariables() {
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, 1, 1)) return;
    }

    enum class BOUNDARY {
        PERIODIC,
        REFLECTING,
        ZERO
    };

    __global__ void denoise(Vector3f *input, Vector3f *output, FeatureBuffer *featureBuffer, int width, int height,
                            Vector3f cameraOrigin) {
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

        auto getNeighbour = [i, j, width, height]__device__ (Vector3f *array, int dx, int dy,
                                                             BOUNDARY boundary = BOUNDARY::PERIODIC) {
            switch(boundary) {
                case BOUNDARY::PERIODIC:
                    return array[(j + height + dy) % height * width + (i + width + dx) % width];
                case BOUNDARY::REFLECTING:
                    //TODO implement?
                    //                    const int toXBoundary = CustomRenderer::min(j + dx)
                    assert(false && "Not implemented.");
                    break;
                case BOUNDARY::ZERO:
                    if(i >= 0 && i < width && j >= 0 && j < height)
                        return array[j*width + i];

                    return Vector3f{0.f};
            }
        };

        const int m_radius = 2;
        const float m_stddev = 0.5f;

        const float alpha = -1.0f / (2.0f * m_stddev * m_stddev);
        const float constant = std::exp(alpha * m_radius * m_radius);


        auto gaussian = [alpha, constant] __device__ (float x){
            return CustomRenderer::max(0.0f, std::exp(alpha * x * x) - constant);
        };

        float integral = 0.f;
        Vector3f tmp{0.f};

        for(int xNew = -m_radius; xNew <= m_radius; ++xNew) {
            for(int yNew = -m_radius + 1; yNew < m_radius; ++yNew){
                const float gaussianContrib = gaussian(sqrtf(xNew*xNew + yNew*yNew));
                tmp += gaussianContrib * getNeighbour(input, xNew, yNew);
                integral += gaussianContrib;
            }
        }

        output[pixelIndex] = tmp/integral;

        //        output[pixelIndex] = Vector3f{(featureBuffer[pixelIndex].position-cameraOrigin).norm()/200};
//                output[pixelIndex] = featureBuffer[pixelIndex].normal.absValues();
                output[pixelIndex] = featureBuffer[pixelIndex].albedo;
        //        output[pixelIndex] = featureBuffer[pixelIndex].variance;
        //        constexpr float numSamples = 512.f;
        //        output[pixelIndex] = Vector3f{powf(static_cast<float>(featureBuffer[pixelIndex].numSubSamples)/(2*numSamples), 2.f)};


//        if(featureBuffer[pixelIndex].variance.maxCoeff() > 0.1) {
//            output[pixelIndex] = 0.25 * getNeighbour(input, 0, -1) + 0.25 * getNeighbour(input, 1, 0) + 0.25 * getNeighbour(input, 0, 1) + 0.25 * getNeighbour(input, -1, 0);
//        } else {
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
}// namespace cudaHelpers
