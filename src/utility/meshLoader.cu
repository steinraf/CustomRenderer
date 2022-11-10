//
// Created by steinraf on 24/10/22.
//

#include "meshLoader.h"
#include "vector.h"
#include "../cudaHelpers.h"

HostMeshInfo loadMesh(const std::filesystem::path &filePath){

    std::cout << "Reading mesh " + filePath.filename().string() + " ...\n";
    std::ifstream file(filePath, std::ios::in);
    if(!file.is_open())
        throw std::runtime_error("Error, file " + filePath.string() + " could not be opened in the MeshLoader.");

    std::string lineString;

    thrust::host_vector<Vector3f> vertices;
    thrust::host_vector<Vector2f> textures;
    thrust::host_vector<Vector3f> normals;


    thrust::host_vector<int> vertexIndices1;
    thrust::host_vector<int> vertexIndices2;
    thrust::host_vector<int> vertexIndices3;

    thrust::host_vector<int> textureIndices1;
    thrust::host_vector<int> textureIndices2;
    thrust::host_vector<int> textureIndices3;

    thrust::host_vector<int> normalIndices1;
    thrust::host_vector<int> normalIndices2;
    thrust::host_vector<int> normalIndices3;


//    thrust::host_vector<FaceElement> faces;

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

            vertexIndices1.push_back(v1 - 1);
            vertexIndices2.push_back(v2 - 1);
            vertexIndices3.push_back(v3 - 1);

            textureIndices1.push_back(t1 - 1);
            textureIndices2.push_back(t2 - 1);
            textureIndices3.push_back(t3 - 1);

            normalIndices1.push_back(n1 - 1);
            normalIndices2.push_back(n2 - 1);
            normalIndices3.push_back(n3 - 1);

            if(!e4.empty()){
                int v4, t4, n4;
                std::istringstream s4(e4);
                s4 >> v4 >> delim >> t4 >> delim >> n4;

                vertexIndices1.push_back(v1 - 1);
                vertexIndices2.push_back(v3 - 1);
                vertexIndices3.push_back(v4 - 1);

                textureIndices1.push_back(t1 - 1);
                textureIndices2.push_back(t3 - 1);
                textureIndices3.push_back(t4 - 1);

                normalIndices1.push_back(n1 - 1);
                normalIndices2.push_back(n3 - 1);
                normalIndices3.push_back(n4 - 1);
            }
        }
    }

    return {
            vertices,
            textures,
            normals,
            {vertexIndices1, vertexIndices2, vertexIndices3},
            {textureIndices1, textureIndices2, textureIndices3},
            {normalIndices1, normalIndices2, normalIndices3}
    };
}

Triangle *mesh2GPU(const HostMeshInfo &mesh){
    thrust::host_vector<Triangle> trias;

    auto triangleFunctorBuilder = [](){

    };


    return trias.data();
}

Triangle *meshToGPU(const HostMeshInfo &mesh){
    const int numTriangles = mesh.normalsIndices.first.size();
    auto *hostTriangles = (Triangle *) malloc(numTriangles * sizeof(Triangle));
    Triangle *deviceTriangles;
    checkCudaErrors(cudaMalloc((void **) &deviceTriangles, numTriangles * sizeof(Triangle)));

    const auto &[
            vertices,
            textures,
            normals,
            vertexIndexList,
            textureIndexList,
            normalIndexList] = mesh;
    
//#pragma omp parallel for
    for(int i = 0; i < numTriangles; ++i){
        hostTriangles[i] = {
                vertices[vertexIndexList.first[i]],
                vertices[vertexIndexList.second[i]],
                vertices[vertexIndexList.third[i]],
                textures[textureIndexList.first[i]],
                textures[textureIndexList.second[i]],
                textures[textureIndexList.third[i]],
                normals[normalIndexList.first[i]],
                normals[normalIndexList.second[i]],
                normals[normalIndexList.third[i]],
                BSDF{Material::DIFFUSE}
        };
    }

    checkCudaErrors(
            cudaMemcpy(deviceTriangles, hostTriangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    delete[] hostTriangles;
    return deviceTriangles;
}


//BVH<Triangle> getBVHFromMesh(const HostMeshInfo &mesh){
//
//}




//DeviceMeshInfo::DeviceMeshInfo(HostMeshInfo meshInfo){
//
//
//
//    totalBoundingBox = thrust::transform_reduce()



//    thrust::device_vector<Vector3f> vertices;
//    thrust::device_vector<Vector2f> textures;
//    thrust::device_vector<Vector3f> normals;
//    thrust::device_vector<FaceElement> faces;
//    thrust::device_vector<uint32_t> mortonCodes;
//    thrust::device_vector<AABB> boundingBoxes;
//    AABB totalBoundingBox;
//}
