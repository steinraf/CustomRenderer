//
// Created by steinraf on 24/10/22.
//

#pragma once

#include <fstream>
#include <filesystem>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../cudaHelpers.h"
#include "vector.h"
#include "../shapes/triangle.h"
#include "../acceleration/bvh.h"


//struct FaceElement{
////        __device__ __host__ FaceElement() = default;
//
//    __device__ __host__ FaceElement(
//            int v1, int v2, int v3,
//            int t1, int t2, int t3,
//            int n1, int n2, int n3
//    )
//            : vertices{v1, v2, v3},
//              textures{t1, t2, t3},
//              normals{n1, n2, n3}{
//
//    }
//
//
//    int vertices[3];
//    int textures[3];
//    int normals[3];
//
//};

struct TriangleIndexList{
    thrust::host_vector<int> first;
    thrust::host_vector<int> second;
    thrust::host_vector<int> third;
};

struct HostMeshInfo{
    thrust::host_vector<Vector3f> vertices;
    thrust::host_vector<Vector2f> textures;
    thrust::host_vector<Vector3f> normals;
//    thrust::host_vector<FaceElement> faces;
    TriangleIndexList vertexIndices;
    TriangleIndexList textureIndices;
    TriangleIndexList normalsIndices;
};

struct DeviceMeshInfo{
    thrust::device_vector<Triangle> triangles;
    thrust::device_vector<uint32_t> mortonCodes;
//    AABB totalBoundingBox;

//    explicit DeviceMeshInfo(HostMeshInfo meshInfo);

    [[nodiscard]] auto toTuple() const{
        return std::tuple{triangles, mortonCodes/*, totalBoundingBox*/};
    }

};


HostMeshInfo loadMesh(const std::filesystem::path &filePath);

Triangle *mesh2GPU(const HostMeshInfo &mesh);

DeviceMeshInfo meshToGPU(const HostMeshInfo &mesh);

struct TriaToAABB{
    __host__ __device__ AABB operator()(const Triangle& tria) const{

        return tria.boundingBox;
    }
};

struct AABBAdder{
    __device__ AABB operator()(const AABB& a1, const AABB& a2) const{
        return a1+a2;
    }
};

struct TriangleToMortonCode{
    __device__ __host__ explicit TriangleToMortonCode(const AABB& ref) : lower(ref.min), dims(ref.max-ref.min){}

    // https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#LeftShift3

    __device__ static inline uint32_t LeftShift3(uint32_t x) {
        if (x == (1 << 10)) --x;
        x = (x | (x << 16)) & 0b00000011000000000000000011111111;
        x = (x | (x <<  8)) & 0b00000011000000001111000000001111;
        x = (x | (x <<  4)) & 0b00000011000011000011000011000011;
        x = (x | (x <<  2)) & 0b00001001001001001001001001001001;
        return x;
    }

    __device__ uint32_t operator()(const Triangle &tria){
        int numBits = 10;
        const Vector3f normalized = static_cast<float>(1u << numBits) * (tria.boundingBox.getCenter() - lower)/dims;

//        printf("BoundingBoxCenter is (%f, %f, %f)\n", tria.boundingBox.getCenter()[0], tria.boundingBox.getCenter()[1], tria.boundingBox.getCenter()[2]);

        assert(normalized[0] >= 0 && normalized[1] >= 0 && normalized[2] >= 0);
        assert(normalized[0] <= (1u << numBits) && normalized[1] <= (1u << numBits) && normalized[2] <= (1u << numBits));

//        printf("Position (%f, %f, %f) becomes morton code %i \n", normalized[0], normalized[1], normalized[2],
//       (LeftShift3(static_cast<uint32_t>(normalized[2])) << 2) |
//        (LeftShift3(static_cast<uint32_t>(normalized[1])) << 1) |
//        LeftShift3(static_cast<uint32_t>(normalized[0])) );

        return  (LeftShift3(static_cast<uint32_t>(normalized[2])) << 2) |
                (LeftShift3(static_cast<uint32_t>(normalized[1])) << 1) |
                (LeftShift3(static_cast<uint32_t>(normalized[0]))       );
    }

private:
    Vector3f lower, dims;
};


