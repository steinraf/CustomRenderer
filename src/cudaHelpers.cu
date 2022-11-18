//
// Created by steinraf on 19/08/22.
//

#include "cudaHelpers.h"


#include "utility/ray.h"
#include "bsdf.h"


#include <iostream>


namespace cudaHelpers{


    __host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int line){
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


    __global__ void freeVariables(int width, int height){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, 1, 1)) return;


    }

    __global__ void denoise(Vector3f *input, Vector3f *output, int width, int height){
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;


        Vector3f tmp{0.f};
        int count = 0;

        auto between = [](int val, int low, int high){ return val >= low && val < high; };

        const int range = 2;


        const float filter[5][5] = {
                { 0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091},
                {0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119},
                {0.01094545, 0.11405416, 0.2491172,  0.11405416, 0.01094545},
                {0.00501119, 0.0522178,  0.11405416, 0.0522178,  0.00501119},
                {0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091},
        };

        for(int a = -range; a <= range; ++a){
            for(int b = -range; b <= range; ++b){
                if(between(i + a, 0, width) && between(j + b, 0, height)){
                    ++count;
                    tmp += filter[range + a][range + b] * input[(j + b) * width + i + a];
                }
            }
        }

        output[pixelIndex] = tmp.clamp(0.f, 1.f); //(input[pixelIndex] + tmp/(count))/2.0;
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
            const int new_split = split + step; // proposed new position

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
