//
// Created by steinraf on 19/11/22.
//

#pragma once

#include <string>
#include <map>
#include <filesystem>
#include <iostream>
#include <cassert>

#include "pugixml.hpp"
#include "../utility/vector.h"
#include "../bsdf.h"



struct SceneRepresentation{
    explicit SceneRepresentation(const std::filesystem::path &file) noexcept(false){
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(file.c_str());




        if (!result){
            std::cout << "XML [" << file.string() << "] parsed with errors\n";
            std::cout << "Error description: " << result.description() << "\n";
            std::cout << "Error offset: " << result.offset << "\n\n";
        }

        auto root = doc.document_element();

        std::cout << "Started parsing " << root.name() << '\n';

        if(std::string(root.name()) != "scene"){
            throw std::runtime_error("Unrecognized XML file root\n");
        }

        for(const auto& node : root.children()){

            if (node.type() == pugi::node_comment || node.type() == pugi::node_declaration)
                continue;
            if (node.type() != pugi::node_element)
                throw std::runtime_error("Unknown XML Node encounted.");

            const std::string &name = node.name();
            std::cout << '\t' << name;
            if(node.attribute("id"))
                std::cout << ": " << getString(node.attribute("id"));
            std::cout << '\n';


            if(name == "default"){
                addDefault(node);
            }else if(name == "integrator"){
                setIntegrator(node);
            }else if(name == "sensor"){
                setSensor(node);
            }else if(name == "shape"){
                if(node.child("emitter")){
                    addEmitter(node);
                }else{
                    addMesh(node);
                }
            }else{
                    throw std::runtime_error("Unrecognized node name \"" + name + "\" while parsing XML.");
            }
        }
    }

    void inline addDefault(const pugi::xml_node &node) noexcept{
        map[node.attribute("name").value()] = node.attribute("value").value();
        std::cout << "\t\t" << "Added mapping " << node.attribute("name").value() << " -> " << node.attribute("value").value() << '\n';
    }

    void inline setIntegrator(const pugi::xml_node &node) const noexcept(false){
        if(getString(node.attribute("type")) != "path")
            throw std::runtime_error("Integrator must be path, not \"" + getString(node.attribute("type")) + "\".");
        std::cout << "\t\t path\n";
    }

    void inline setSensor(const pugi::xml_node &node) noexcept(false) {
        if(getString(node.attribute("type")) != "perspective")
            throw std::runtime_error("Sensor must be perspective, not \"" + getString(node.attribute("type")) + "\" .");

        for(const auto& child : node.children()){
            const std::string &childName = child.name();
            if(childName == "float"){
                if(getString(child.attribute("name")) != "fov")
                    throw std::runtime_error("Unrecognized sensor float option " + getString(child.attribute("name")));

                cameraInfo.fov = std::stof(getString(child.attribute("value")));
                std::cout << "\t\tFOV: \n\t\t\t" << cameraInfo.fov << '\n';

            }else if(childName == "transform"){
                std::cout << "\t\t" << "Transform: \n";
                auto lookAt = child.child("lookat");
                if(lookAt){
                    cameraInfo.target = getVector3f(lookAt, "target", "\t\t\t");
                    cameraInfo.origin = getVector3f(lookAt, "origin", "\t\t\t");
                    cameraInfo.up = getVector3f(lookAt, "up", "\t\t\t");
                }else{
                    throw std::runtime_error("Sensor Transform must contain lookAt.");
                }

            }else if(childName == "sampler"){

                std::cout << "\t\t" << "Sampler: " << '\n';

                if(getString(child.attribute("type")) != "independent")
                    throw std::runtime_error("Sampler type must be independent, not \"" + getString(child.attribute("type")) + "\".");

                auto sampleCount = child.child("integer");
                if(getString(sampleCount.attribute("name")) != "sample_count")
                    throw std::runtime_error("Encountered unknown attribute \"" + getString(sampleCount.attribute("name")) + "\" in sampler.");
                sceneInfo.samplePerPixel = std::stoi(getString(sampleCount.attribute("value")));
                std::cout << "\t\t\t" << "SampleCount: " << sceneInfo.samplePerPixel << '\n';
            }else if(childName == "film"){

                if(getString(child.attribute("type")) != "hdrfilm")
                    throw std::runtime_error("Invalid Film type \"" + getString(child.attribute("type")) + "\" encountered.");

                for(const auto& filmChild : child.children()){
                    const std::string &filmChildName = filmChild.name();
                    if(filmChildName == "rfilter"){
                        std::cerr << "WARNING, IGNORING FILTERS\n";
                    }else if(filmChildName == "integer"){
                        if(getString(filmChild.attribute("name")) == "width"){
                            sceneInfo.width = std::stoi(getString(filmChild.attribute("value")));
                            std::cout << "\t\t\t" << "Width: " << sceneInfo.width << '\n';
                        }else if(getString(filmChild.attribute("name")) == "height"){
                            sceneInfo.height = std::stoi(getString(filmChild.attribute("value")));
                            std::cout << "\t\t\t" << "Height: " << sceneInfo.height << '\n';
                        }else{
                            throw std::runtime_error("Unknown integer option \"" + getString(filmChild.attribute("name")) + "\"for hdrfilm encountered.");
                        }
                    }else{
                        throw std::runtime_error("Unknown option \"" + filmChildName + "\" for hdrfilm encountered.");
                    }
                }
            }else{
                throw std::runtime_error("Invalid child tag \"" + childName + "\" found for sensor.");
            }
        }

    }

    void addFilename(const std::string &name, bool isEmitter=false){

        if(isEmitter){
            emitterInfos.back().filename = name;
        }else{
            meshInfos.back().filename = name;
        }


        std::cout << "\t\tFilename: " << meshInfos.back().filename << '\n';
    }

    void addBSDF(const pugi::xml_node &node){
        if(getString(node.attribute("type")) == "diffuse"){
            const auto color = node.child("rgb");
            if(getString(color.attribute("name")) != "reflectance")
                throw std::runtime_error("Invalid Tag \"" + getString(color.attribute("name")) + "\"found for material.");
            std::cout << "\t\tMaterial:\n";
            std::cout << "\t\t\tBSDF: DIFFUSE\n";
            meshInfos.back().bsdf = {Material::DIFFUSE, getVector3f(color, "value", "\t\t\t", "reflectance")};

        }else{
            throw std::runtime_error("Invalid Material \"" + getString(node.attribute("type")) + "\". Only diffuse is supported.");
        }
    }

    void inline addMesh(const pugi::xml_node &node) noexcept(false){
        if(getString(node.attribute("type")) != "obj")
            throw std::runtime_error("Error while parsing shape. Only .obj files are supported.");

        meshInfos.emplace_back();

        for(const auto& child : node.children()){
            const std::string &childName = child.name();
            if(childName == "string"){
                addFilename(getString(child.attribute("value")));
            }else if(childName == "bsdf"){
                addBSDF(child);
            }else if(childName == "transform"){
                createTransform(child);
            }else{
                throw std::runtime_error("Invalid Tag \"" + childName + "\" found for shape.");
            }
        }
    }

    void inline addEmitter(const pugi::xml_node &node) noexcept(false){
        if(getString(node.attribute("type")) != "obj")
            throw std::runtime_error("Error while parsing shape. Only .obj files are supported.");

        emitterInfos.emplace_back();

        for(const auto& child : node.children()){
            const std::string &childName = child.name();
            if(childName == "string"){
                addFilename(getString(child.attribute("value")), true);
            }else if(childName == "bsdf"){
                addBSDF(child);
            }else if(childName == "emitter"){
                if(getString(child.attribute("type")) != "area")
                    throw std::runtime_error("Invalid Emitter \"" + getString(child.attribute("type")) + "\". Only area is supported.");

                const auto color = child.child("rgb");
                if(getString(color.attribute("name")) != "radiance")
                    throw std::runtime_error("Invalid Tag \"" + getString(color.attribute("name")) + "\" found for emitter.");

                std::cout << "\t\tEmitter:\n";
                std::cout << "\t\t\tType: Area\n";
//                        emitters.emplace_back(filenames[filenames.size()-1], getVector3f(color, "value", "\t\t\t"));
//                        filenames.pop_back(); //
            }else if(childName == "transform"){
                createTransform(child, true);
            }else{
                throw std::runtime_error("Invalid Tag \"" + childName + "\" found for shape.");
            }
        }
    }


    [[nodiscard]] Vector3f inline getVector3f(pugi::xml_node node, const std::string &name, const std::string &indents="", const std::string &outName = "") const noexcept{
        auto attrib = node.attribute(name.c_str());
        const std::string targetS = getString(attrib);
        std::cout << indents << (outName.empty() ? name : outName) << ": {" << targetS << "}\n";

        return Vector3f{targetS};
    }

    [[nodiscard]] std::string getString(const pugi::xml_attribute& attrib) const noexcept{
        if(attrib.value()[0] == '$')
            return map.at(std::string(attrib.value()).substr(1));

        return attrib.value();
    }

    void inline createTransform(const pugi::xml_node &transform, bool isEmitter=false) noexcept(false) {


        Affine3f &currentTransform = [&]() -> Affine3f &{
            if(isEmitter)
                return emitterInfos.back().transform;
            else
                return meshInfos.back().transform;
        }();


        std::cout << "\t\tTransform: \n";
        for(auto &child : transform.children()){
            const std::string & tfChildName = child.name();
            if(tfChildName == "translate"){
                currentTransform = Affine3f(getVector3f(child, "value", "\t\t\t", "translation")) * currentTransform;
            }else if(tfChildName == "scale"){
                currentTransform = Affine3f(Matrix3f::fromDiag(getVector3f(child, "value", "\t\t\t", "scale"))) * currentTransform;
            }else if(tfChildName == "rotateAxis"){
                const float angle = std::stof(child.attribute("angle").value());
                currentTransform = Affine3f({
                                                    getVector3f(child, "axis", "\t\t\t", "rotation axis"),
                                                    angle
                                            }) * currentTransform;
                std::cout << "\t\t\trotation angle: " << angle << "°\n";
            }else if(tfChildName == "matrix"){
                throw std::runtime_error("Matrix transforms are not implemented yet.");
                //TODO implement matrix transform
            }else{
                throw std::runtime_error("Invalid Tag \"" + tfChildName + "\" found for transform");
            }
        }
    }

//private:

    struct CameraInfo{
        CameraInfo()
            :target(0.f, 0.f, -1.f), origin(0.f), up(0.f, 1.f, 0.f), fov(30){

        }

        Vector3f target, origin, up;
        float fov;
    };

    CameraInfo cameraInfo;

    struct MeshInfo{
        MeshInfo() = default;
        std::string filename;
        Affine3f transform;
        BSDF bsdf;
    };

    std::vector<MeshInfo> meshInfos{};

    struct EmitterInfo{
        EmitterInfo() = default;
        std::string filename;
        Affine3f transform;
        BSDF bsdf;
        Vector3f color;
    };

    std::vector<EmitterInfo> emitterInfos{};

    struct SceneInfo{
        SceneInfo()
            :samplePerPixel(4), width(100), height(100){

        }

        int samplePerPixel;
        int width, height;
    };

    SceneInfo sceneInfo;


private:
    std::unordered_map<std::string ,std::string> map;

};
