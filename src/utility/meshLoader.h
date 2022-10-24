//
// Created by steinraf on 24/10/22.
//

#ifndef MESHLOADER_H
#define MESHLOADER_H

#include <fstream>
#include <filesystem>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "vector.h"
#include "../shapes/triangle.h"


struct FaceElement{
//        __device__ __host__ FaceElement() = default;

    __device__ __host__ FaceElement(
            int v1, int v2, int v3,
            int t1, int t2, int t3,
            int n1, int n2, int n3
    )
            : vertices{v1, v2, v3},
              textures{t1, t2, t3},
              normals{n1, n2, n3}{

    }


    int vertices[3];
    int textures[3];
    int normals[3];

};

struct HostMeshInfo{
    thrust::host_vector<Vector3f> vertices;
    thrust::host_vector<Vector2f> textures;
    thrust::host_vector<Vector3f> normals;
    thrust::host_vector<FaceElement> faces;
};


HostMeshInfo loadMesh(const std::filesystem::path &filePath);

Triangle *meshToGPU(const HostMeshInfo &mesh);

#endif
