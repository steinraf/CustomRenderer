#include <filesystem>


#include "src/utility/vector.h"
#include "src/scene.h"


int main(int argc, char **argv) {

//    for(int i = 0; i < argc; ++i){
//        std::cout << i << ": " << argv[i] << '\n';
//    }

//    exit(0);

    std::cout << "Starting rendering...\n";


    clock_t start, stop;
    start = clock();

    Scene s(384, 216);
    s.renderCPU();
//    s.renderGPU();


    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cout << "Computation took " << timer_seconds << " seconds.\n";

    std::cout << "Drew image to file\n";


    cudaDeviceReset();

    return EXIT_SUCCESS;
}
