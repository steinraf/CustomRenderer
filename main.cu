#include <filesystem>


#include "src/utility/vector.h"
#include "src/scene.h"



int main(int argc, char **argv) {

//    for(int i = 0; i < argc; ++i){
//        std::cout << i << ": " << argv[i] << '\n';
//    }

//    exit(0);



    std::cout << "Starting rendering...\n";




    Scene s(3840, 2160);
    s.renderCPU();
//    s.renderGPU();




    std::cout << "Drew image to file\n";


    cudaDeviceReset();

    return EXIT_SUCCESS;
}
