#include <filesystem>


#include "src/utility/vector.h"
#include "src/scene.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

//struct AABB{
//    Vector3f start;
//    Vector3f end;
//};
//
//struct Test{
//    int a;
//};


int main(int argc, char **argv) {

//    for(int i = 0; i < argc; ++i){
//        std::cout << i << ": " << argv[i] << '\n';
//    }

//    exit(0);

//    std::cout << "Sorting test...\n";
//
//    int vecSize = 100;
//
//    thrust::host_vector<Test> hostVector(vecSize);
//    for(auto &test : hostVector)
//        test = Test{rand()};
//
//    for(const auto& test : hostVector){
//        std::cout << test.a << ' ';
//    }
//
//    std::cout << '\n';
//
//    thrust::device_vector<Test> deviceVector(hostVector);
//
//    thrust::sort(deviceVector.begin(), deviceVector.end(),  []__device__(const Test &a, const Test &b){return a.a < b.a;});
//
//    thrust::copy(deviceVector.begin(), deviceVector.end(), hostVector.begin());
//
//    for(int i = 0; i < vecSize-1; ++i){
//        std::cout << hostVector[i+1].a - hostVector[i].a << ' ';
//    }
//
//
//    exit(0);



    std::cout << "Starting rendering...\n";




    Scene s(384, 216);
    s.renderCPU();
//    s.renderGPU();




    std::cout << "Drew image to file\n";


    cudaDeviceReset();

    return EXIT_SUCCESS;
}
