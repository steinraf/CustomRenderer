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



//        std::vector<Vector3f> translate{},
//                scale{},
//                rotateAxis{};
//        std::vector<float> rotateAngle{};
//
//        for(int i = 0; i < translate.size(); ++i){
//            Affine3f{{rotateAxis[i], rotateAngle[i]}, translate[i]}
//                *Affine3f{Matrix3f::fromDiag(scale[i]), Vector3f{0.f}};
//        }

        for(const auto& node : root.children()){

            const std::string &name = node.name();


            if (node.type() == pugi::node_comment || node.type() == pugi::node_declaration)
                continue;
            if (node.type() != pugi::node_element)
                throw std::runtime_error("Unknown XML Node encounted.");

            std::cout << '\t' << name;
            if(node.attribute("id"))
                std::cout << ": " << getString(node.attribute("id"));
            std::cout << '\n';


            if(name == "default"){
                map[node.attribute("name").value()] = node.attribute("value").value();
                std::cout << "\t\t" << "Added mapping " << node.attribute("name").value() << " -> " << node.attribute("value").value() << '\n';
            }else if(name == "integrator"){
                if(getString(node.attribute("type")) != "path")
                    throw std::runtime_error("Integrator must be path.");
                std::cout << "\t\t path\n";
            }else if(name == "sensor"){
                if(getString(node.attribute("type")) != "perspective")
                    throw std::runtime_error("Sensor must be perspective.");

                for(const auto& child : node.children()){
                    const std::string &childName = child.name();
                    if(childName == "float"){
                        if(getString(child.attribute("name")) == "fov"){
                            fov = std::stof(getString(child.attribute("value")));
                        }else{
                            throw std::runtime_error("Unrecognized sensor float option " + getString(child.attribute("name")));
                        }
                    }else if(childName == "transform"){
                        std::cout << "\t\t" << "Transform: \n";
                        auto lookAt = child.child("lookat");
                        target = getVector3f(lookAt, "target", "\t\t\t");
                        origin = getVector3f(lookAt, "origin", "\t\t\t");
                        up = getVector3f(lookAt, "up", "\t\t\t");
                    }else if(childName == "sampler"){

                        std::cout << "\t\t" << "Sampler: " << '\n';

                        if(getString(child.attribute("type")) != "independent")
                            throw std::runtime_error("Type must be independent.");

                        auto sampleCount = child.child("integer");
                        if(getString(sampleCount.attribute("name")) != "sample_count")
                            throw std::runtime_error("Encountered unknown attribute in sampler.");
                        samplePerPixel = std::stoi(getString(sampleCount.attribute("value")));
                        std::cout << "\t\t\t" << "SampleCount: " << samplePerPixel << '\n';
                    }else if(childName == "film"){

                        if(getString(child.attribute("type")) != "hdrfilm")
                            throw std::runtime_error("Invalid Film type encountered.");

                        for(const auto& filmChild : child.children()){
                            const std::string &filmChildName = filmChild.name();
                            if(filmChildName == "rfilter"){
                                std::cerr << "WARNING, IGNORING FILTERS\n";
                            }else if(filmChildName == "integer"){
                                if(getString(filmChild.attribute("name")) == "width"){
                                    width = std::stoi(getString(filmChild.attribute("value")));
                                    std::cout << "\t\t\t" << "Width: " << width << '\n';
                                }else if(getString(filmChild.attribute("name")) == "height"){
                                    height = std::stoi(getString(filmChild.attribute("value")));
                                    std::cout << "\t\t\t" << "Height: " << height << '\n';
                                }else{
                                    throw std::runtime_error("Unknown integer option for hdrfilm encountered.");
                                }
                            }else{
                                throw std::runtime_error("Unknown option for hdrfilm encountered.");
                            }
                        }
                    }else{
                        throw std::runtime_error("Invalid child tag found for sensor.");
                    }
                }

            }else if(name == "shape"){

                if(getString(node.attribute("type")) != "obj")
                    throw std::runtime_error("Error while parsing shape. Only .obj files are supported.");

                meshTransforms.emplace_back();
//                if(transform){
//
//                }

                for(const auto& child : node.children()){
                    const std::string &childName = child.name();
                    if(childName == "string"){
                        filenames.push_back(getString(child.attribute("value")));
                        std::cout << "\t\tFilename: " << filenames[filenames.size()-1] << '\n';
                    }else if(childName == "bsdf"){
                        if(getString(child.attribute("type")) == "diffuse"){
                            const auto color = child.child("rgb");
                            if(getString(color.attribute("name")) != "reflectance")
                                throw std::runtime_error("Invalid Tag \"" + getString(color.attribute("name")) + "\"found for material.");
                            std::cout << "\t\tMaterial:\n";
                            std::cout << "\t\t\tBSDF: DIFFUSE\n";
                            bsdfs.emplace_back(Material::DIFFUSE, getVector3f(color, "value", "\t\t\t", "reflectance"));

                        }else{
                            throw std::runtime_error("Invalid Material \"" + getString(child.attribute("type")) + "\". Only diffuse is supported.");
                        }
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
                        createTransform(child);
//                        meshTransforms.emplace_back();
//                        createTransform(child);
                    }else{
                        throw std::runtime_error("Invalid Tag \"" + childName + "\" found for shape.");
                    }
                }
            }else{
                    throw std::runtime_error("Unrecognized node name \"" + name + "\" while parsing XML.");
            }
        }



        assert(samplePerPixel > 0);
        assert(width > 0);
        assert(height > 0);
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

//    auto createTransform = [&](){
//
//    };

    void inline createTransform(const pugi::xml_node &transform, bool isEmitter=false) noexcept(false) {


        Affine3f &currentTransform = meshTransforms[meshTransforms.size() - 1];

//        auto currentTransform = [&]() -> Affine3f &{
//            if(isEmitter)
//                return meshTransforms[meshTransforms.size() - 1];
//            else
//                return meshTransforms[meshTransforms.size() - 1];
//        }();


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
            }else{
                throw std::runtime_error("Invalid Tag \"" + tfChildName + "\" found for transform");
            }
        }
    }

//private:

    struct CameraInfo{
        Vector3f target, origin, up;
        float fov=30;
    };

    struct MeshInfo{
        std::string filename;
        Affine3f transform;
        BSDF bsdf;
    };

    struct EmitterInfo{
        std::string filename;
        Affine3f transform;
        BSDF bsdf;
        Vector3f color;
    };

    struct SceneInfo{
        int samplePerPixel;
        int width, height;
    };

    Vector3f target, origin, up;
    std::vector<Affine3f> meshTransforms{};
    float fov=30;
    int samplePerPixel=-1;
    std::vector<std::string> filenames;
    std::vector<BSDF> bsdfs;
//    std::vector<std::pair<std::string, Vector3f>> emitters;
    int width=-1, height=-1;

private:
    std::unordered_map<std::string ,std::string> map;

};
