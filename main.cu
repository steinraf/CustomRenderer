#include <filesystem>


#include "src/utility/vector.h"
#include "src/scene/scene.h"
#include "src/scene/sceneLoader.h"


static void glfw_error_callback(int error, const char* description){
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

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


    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()){
        std::cerr << "Error while initializing glfw. Exiting.\n";
        exit(1);
    }



    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    const auto monitor = glfwGetPrimaryMonitor();

    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Dear ImGui GLFW+OpenGL3 example", monitor, nullptr);
    if (window == nullptr){
        std::cerr << "Window creation failed. Exiting.\n";
        exit(1);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);


    if(glewInit() != GLEW_OK){
        std::cerr << "Glew not correctly initialized. Exiting.\n";
        exit(1);
    }

    Scene scene(SceneRepresentation(filePath), Device::CPU);

    bool needsRender = true;


    while (!glfwWindowShouldClose(window))
    {

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Debug Info");

            ImGui::Text("Progress: %f percent", scene.getPercentage());


            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        {
            ImGui::Begin("CUDA GPU Path Tracing");

            if(needsRender){
                if(!scene.render()){
                    needsRender = false;
                    scene.denoise();
                    scene.saveOutput();
                }
                const auto availableSize = ImVec2{
                    ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
                    ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y,
                };
                ImGui::Image((void *) (intptr_t)scene.hostImageTexture, availableSize);
            }else{
                const auto availableSize = ImVec2{
                        ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
                        ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y,
                };
                ImGui::Image((void *) (intptr_t)scene.hostImageTexture, availableSize);
            }



//            scene.denoise();

            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }


    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    scene.saveOutput();
    std::cout << "Drew image to file\n";







    return EXIT_SUCCESS;
}
