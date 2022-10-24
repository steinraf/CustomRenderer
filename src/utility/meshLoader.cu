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

    std::string lineString, start;

    thrust::host_vector<Vector3f> vertices;
    thrust::host_vector<Vector2f> textures;
    thrust::host_vector<Vector3f> normals;


    thrust::host_vector<FaceElement> faces;

    while(std::getline(file, lineString)){
        std::istringstream line{lineString};

        line >> start;

        if(start == "v"){
            float x, y, z;
            line >> x >> y >> z;

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

            faces.push_back(FaceElement(v1, v2, v3, t1, t2, t3, n1, n2, n3));

            if(!e4.empty()){
                int v4, t4, n4;
                std::istringstream s4(e4);
                s4 >> v4 >> delim >> t4 >> delim >> n4;
                faces.push_back(FaceElement(v1, v3, v4, t1, t3, t4, n1, n3, n4));
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
    auto *hostTriangles = (Triangle *) malloc(mesh.faces.size());
    Triangle *deviceTriangles;
    checkCudaErrors(cudaMalloc((void **) &deviceTriangles, mesh.faces.size() * sizeof(Triangle)));

    const auto &[vertices, textures, normals, faces] = mesh;

    int counter = 0;
    for(const auto &face: faces){

        hostTriangles[counter++] = {
                vertices[face.vertices[0]],
                vertices[face.vertices[1]],
                vertices[face.vertices[2]],
                BSDF{Material::DIFFUSE}
        };
    }

    checkCudaErrors(cudaMemcpy(deviceTriangles, &hostTriangles, mesh.faces.size(), cudaMemcpyHostToDevice));
    return deviceTriangles;
}



