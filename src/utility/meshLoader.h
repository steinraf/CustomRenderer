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


struct TriangleIndexList{
    thrust::host_vector<int> first;
    thrust::host_vector<int> second;
    thrust::host_vector<int> third;
};

struct HostMeshInfo{
    thrust::host_vector<Vector3f> vertices;
    thrust::host_vector<Vector2f> textures;
    thrust::host_vector<Vector3f> normals;
    TriangleIndexList vertexIndices;
    TriangleIndexList textureIndices;
    TriangleIndexList normalsIndices;
};

struct DeviceMeshInfo{
    thrust::device_vector<Triangle> triangles;
    thrust::device_vector<uint32_t> mortonCodes;
    thrust::device_vector<float> areaCDF;
    float totalArea;

    [[nodiscard]] auto toTuple() const noexcept{
        return std::tuple{triangles, mortonCodes, areaCDF, totalArea};
    }

};


HostMeshInfo loadMesh(const std::filesystem::path &filePath, const Affine3f &transform) noexcept(false);

DeviceMeshInfo meshToGPU(const HostMeshInfo &mesh) noexcept;

struct TriaToAABB{
    __host__ __device__ constexpr AABB operator()(const Triangle &tria) const noexcept {
        return tria.boundingBox;
    }
};



struct AABBAdder{
    __device__ constexpr AABB operator()(const AABB &a1, const AABB &a2) const noexcept{
        return a1 + a2;
    }
};

struct TriaToArea{
    __host__ __device__ constexpr float operator()(const Triangle &tria) const noexcept {
        return tria.getArea();
    }
};

struct TriangleToCDF{
private:
    float totalArea;
public:
    __device__ __host__ explicit TriangleToCDF(float totalArea) noexcept
        :totalArea(totalArea){

    }

    __host__ __device__ constexpr float operator()(const Triangle &tria) const noexcept {
        return tria.getArea()/totalArea;
    }
};

struct TriangleToMortonCode{
    __device__ __host__ explicit TriangleToMortonCode(const AABB &ref) noexcept
        : lower(ref.min), dims(ref.max - ref.min){

    }

    // https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#LeftShift3
    __device__ static constexpr inline uint32_t LeftShift3(uint32_t x) noexcept {
        if(x == (1 << 10)) --x;
        x = (x | (x << 16)) & 0b00000011000000000000000011111111;
        x = (x | (x << 8)) & 0b00000011000000001111000000001111;
        x = (x | (x << 4)) & 0b00000011000011000011000011000011;
        x = (x | (x << 2)) & 0b00001001001001001001001001001001;
        return x;
    }

    __device__ constexpr uint32_t operator()(const Triangle &tria) const noexcept{
        int numBits = 10;
        const Vector3f normalized = static_cast<float>(1u << numBits) * (tria.boundingBox.getCenter() - lower) / dims;

        assert(normalized[0] >= 0 && normalized[1] >= 0 && normalized[2] >= 0);
        assert(normalized[0] <= (1u << numBits) && normalized[1] <= (1u << numBits) &&
               normalized[2] <= (1u << numBits));

        return (LeftShift3(static_cast<uint32_t>(normalized[2])) << 2) |
               (LeftShift3(static_cast<uint32_t>(normalized[1])) << 1) |
               (LeftShift3(static_cast<uint32_t>(normalized[0])));
    }

private:
    Vector3f lower, dims;
};

template<typename Primitive>
__host__ BLAS<Primitive> * getMeshFromFile(const std::string &filename, thrust::device_vector<Primitive> &deviceTrias,
                                            thrust::device_vector<float> &areaCDF, float &totalArea, const Affine3f &transform, AreaLight *deviceEmitter = nullptr){
    clock_t startGeometryBVH = clock();

//    if(!radiance.isZero())
//        std::cout << "\tLoading Emitter with radiance " << radiance << "...\n";

    auto mesh = loadMesh(filename, transform);
    auto[deviceTriangles, deviceMortonCodes, deviceCDF, area] = meshToGPU(mesh).toTuple();
    totalArea = area;
    Primitive *deviceTriaPtr = deviceTriangles.data().get();

    const size_t numTriangles = mesh.vertexIndices.first.size();

    std::cout << "\tLoading Geometry took "
              << ((double) (clock() - startGeometryBVH)) / CLOCKS_PER_SEC
              << " seconds.\n";

    std::cout << "\tManaging " << numTriangles << " triangles.\n";

    clock_t bvhConstructStart = clock();

    BLAS<Primitive> *bvh;

    AccelerationNode<Primitive> *bvhTotalNodes;
    checkCudaErrors(cudaMalloc((void **) &bvhTotalNodes,
                               sizeof(AccelerationNode<Primitive>) * (2 * numTriangles - 1))); //n-1 internal, n leaf
    checkCudaErrors(cudaMalloc((void **) &bvh, sizeof(BLAS<Primitive> *)));

    //    printf("BLAS has allocation range of (%p, %p)\n", bvhTotalNodes, bvhTotalNodes + (2 * numTriangles - 1));



    cudaHelpers::constructBVH<<<(numTriangles + 1024 - 1) /
                                1024, 1024>>>(bvhTotalNodes, deviceTriaPtr, deviceMortonCodes.data().get(), numTriangles);


    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "\tBLAS construction took "
              << ((double) (clock() - bvhConstructStart)) / CLOCKS_PER_SEC
              << " seconds.\n";

    clock_t bvhBoundingBox = clock();

    cudaHelpers::computeBVHBoundingBoxes<<<1, 1>>>(bvhTotalNodes);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "\tBLAS boundingBox Compute took "
              << ((double) (clock() - bvhBoundingBox)) / CLOCKS_PER_SEC
              << " seconds.\n";

    clock_t initBVHTime = clock();

    cudaHelpers::initBVH<<<1, 1>>>(bvh, bvhTotalNodes, deviceCDF.data().get(), numTriangles, deviceEmitter);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "\tBLAS init took "
              << ((double) (clock() - initBVHTime)) / CLOCKS_PER_SEC
              << " seconds.\n";

    std::cout << "\tLoading Geometry and BLAS construction took "
              << ((double) (clock() - startGeometryBVH)) / CLOCKS_PER_SEC
              << " seconds.\n";

    deviceTrias = std::move(deviceTriangles);
    areaCDF = std::move(deviceCDF);

    return bvh;
};



