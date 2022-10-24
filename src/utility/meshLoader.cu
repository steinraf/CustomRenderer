//
// Created by steinraf on 24/10/22.
//

#include "meshLoader.h"
#include "vector.h"
#include "../cuda_helpers.h"

HostMeshInfo loadMesh(const std::filesystem::path &filePath){

    std::cout << "Reading mesh " + filePath.filename().string() + " ...\n";
    std::ifstream file(filePath, std::ios::in);
    if(!file.is_open())
        throw std::runtime_error("Error, file " + filePath.string() + " could not be opened in the MeshLoader.");

    std::string lineString;

    thrust::host_vector<Vector3f> vertices;
    thrust::host_vector<Vector2f> textures;
    thrust::host_vector<Vector3f> normals;


    thrust::host_vector<FaceElement> faces;

    while(std::getline(file, lineString)){
        std::istringstream line{lineString};
        std::string start;

        line >> start;

//        std::cout << "Read line: " << lineString << '\n';

        if(start == "v"){
            float x, y, z;
            line >> x >> y >> z;
//            printf("VERT: %f,%f,%f \n", x, y, z);
            vertices.push_back({x, y, z});

        }else if(start == "vt"){
            float u, v, w;
            line >> u >> v >> w;
            textures.push_back({u, v});
        }else if(start == "vn"){
            float x, y, z;
            line >> x >> y >> z;
            normals.push_back({x, y, z});

        }else if(start == "f"){
            std::string e1, e2, e3, e4;
            line >> e1 >> e2 >> e3 >> e4;

            std::istringstream s1(e1), s2(e2), s3(e3);

            int v1, v2, v3,
                    t1, t2, t3,
                    n1, n2, n3;

            char delim;

            s1 >> v1 >> delim >> t1 >> delim >> n1;
            s2 >> v2 >> delim >> t2 >> delim >> n2;
            s3 >> v3 >> delim >> t3 >> delim >> n3;

            faces.push_back(FaceElement(v1 - 1, v2 - 1, v3 - 1, t1 - 1, t2 - 1, t3 - 1, n1 - 1, n2 - 1, n3 - 1));

            if(!e4.empty()){
                int v4, t4, n4;
                std::istringstream s4(e4);
                s4 >> v4 >> delim >> t4 >> delim >> n4;
                faces.push_back(FaceElement(v1 - 1, v3 - 1, v4 - 1, t1 - 1, t3 - 1, t4 - 1, n1 - 1, n3 - 1, n4 - 1));
            }
        }
    }

    return {
            vertices,
            textures,
            normals,
            faces
    };
}

Triangle *meshToGPU(const HostMeshInfo &mesh){
    auto *hostTriangles = (Triangle *) malloc(mesh.faces.size() * sizeof(Triangle));
    Triangle *deviceTriangles;
    checkCudaErrors(cudaMalloc((void **) &deviceTriangles, mesh.faces.size() * sizeof(Triangle)));

    const auto &[vertices, textures, normals, faces] = mesh;

//    printf("Received vertices:\n");
//    for(const auto &vertex: mesh.vertices)
//        printf("(%f,%f,%f)\n", vertex[0], vertex[1], vertex[2]);
//
//    printf("Received Vertex Indices:\n");
//    for(const auto &face: mesh.faces)
//        printf("(%i,%i,%i)\n", face.vertices[0], face.vertices[1], face.vertices[2]);

    int counter = 0;
    for(const auto &face: faces){
//        printf("Assembling triangle with p0 = (%f,%f,%f)\n",
//               vertices[face.vertices[0]][0],
//               vertices[face.vertices[0]][1],
//               vertices[face.vertices[0]][2]);

        hostTriangles[counter++] = {
                vertices[face.vertices[0]],
                vertices[face.vertices[1]],
                vertices[face.vertices[2]],
                BSDF{Material::DIFFUSE}
        };
    }

    checkCudaErrors(
            cudaMemcpy(deviceTriangles, hostTriangles, mesh.faces.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    delete[] hostTriangles;
    return deviceTriangles;
}



