//
// Created by steinraf on 19/08/22.
//

#include "cudaHelpers.h"


#include "utility/ray.h"
#include "bsdf.h"


#include <iostream>


namespace cudaHelpers{


    __host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int line) noexcept(false){
        if(result){
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                      file << ":" << line << " '" << func << "' \n";
            cudaDeviceReset();
            exit(99);
        }
    }


    __global__ void initRng(int width, int height, curandState *randState){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

        curand_init(42, pixelIndex, 0, &randState[pixelIndex]);
    }


    __global__ void freeVariables(){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, 1, 1)) return;


    }

    enum class BOUNDARY{
        PERIODIC,
        REFLECTING,
        ZERO
    };

    __global__ void denoise(Vector3f *input, Vector3f *output, FeatureBuffer*featureBuffer, int width, int height, Vector3f cameraOrigin){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

        auto getNeighbour = [i, j, width, height] __device__ (auto *array, int dx, int dy, BOUNDARY boundary=BOUNDARY::PERIODIC){
            switch(boundary){
                case BOUNDARY::PERIODIC:
                    return array[(j + height + dy)%height * width + (i + width + dx)%width];
                case BOUNDARY::REFLECTING:
                    //TODO implement?
//                    const int toXBoundary = CustomRenderer::min(j + dx)
                    assert(false && "Not implemented.");
                    break;
                case BOUNDARY::ZERO:
                    assert(false && "Not implemented.");
                    break;
            }
        };

//        output[pixelIndex] = Vector3f{(featureBuffer[pixelIndex].position-cameraOrigin).norm()/200};
//        output[pixelIndex] = featureBuffer[pixelIndex].normal.absValues();
//        output[pixelIndex] = featureBuffer[pixelIndex].albedo;
        output[pixelIndex] = featureBuffer[pixelIndex].variance;
//        constexpr float numSamples = 64.f;
//        output[pixelIndex] = Vector3f{powf(static_cast<float>(featureBuffer[pixelIndex].numSubSamples)/numSamples, 2.f)};


//        if(featureBuffer[pixelIndex].variance.maxCoeff() > 0.1){
//            output[pixelIndex] =  0.25 * getNeighbour(input, 0,-1)
//                                + 0.25 * getNeighbour(input, 1, 0)
//                                + 0.25 * getNeighbour(input, 0, 1)
//                                + 0.25 * getNeighbour(input,-1, 0);
//        }else{
//            output[pixelIndex] = input[pixelIndex];
//        }

//        output[pixelIndex] = Color3f(featureBuffer[pixelIndex].variance.norm());
    }


    __device__ int findSplit(const uint32_t *mortonCodes, int first, int last, int numPrimitives){

        const unsigned int first_code = mortonCodes[first];

        // calculate the number of highest bits that are the same
        // for all objects, using the count-leading-zeros intrinsic

        const int common_prefix =
                delta(first, last, numPrimitives, mortonCodes, first_code);

        // use binary search to find where the next bit differs
        // specifically, we are looking for the highest object that
        // shares more than commonPrefix bits with the first one

        int split = first; // initial guess
        int step = last - first;

        do{
            step = (step + 1) >> 1; // exponential decrease
            const int new_split = split + step; // proposed new p

            if(new_split < last){
                const int split_prefix = delta(
                        first, new_split, numPrimitives, mortonCodes, first_code);
                if(split_prefix > common_prefix){
                    split = new_split; // accept proposal
                }
            }
        }while(step > 1);

        return split;
    }
}
