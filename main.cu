#include <filesystem>


#include "src/utility/vector.h"
#include "src/scene/scene.h"
#include "src/scene/sceneLoader.h"


int main(int argc, char **argv){
    std::cout << "Parsing obj...\n";

    if(argc != 2){
        throw std::runtime_error("Please add a file as argument.");
    }

    const std::filesystem::path filePath = argv[1];

//    const std::filesystem::path filePath = "./scenes/simple.xml";
//    const std::filesystem::path filePath = "./scenes/clocks.xml";


    assert(filePath.extension() == ".xml");

    std::cout << "Starting rendering...\n";

    Scene s(SceneRepresentation(filePath), Device::CPU);

    s.renderCPU();
//    s.renderGPU();




    std::cout << "Drew image to file\n";


//    cudaDeviceReset();

    return EXIT_SUCCESS;
}
