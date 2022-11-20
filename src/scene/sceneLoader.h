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
    explicit SceneRepresentation(const std::filesystem::path &file) {
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
                    if(childName == "transform"){
                        std::cout << "\t\t" << "Transform: \n";
                        const auto lookAt = child.child("lookat");
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

                for(const auto& child : node.children()){
                    const std::string &childName = child.name();
                    if(childName == "string"){
                        filenames.push_back(getString(child.attribute("value")));
                        std::cout << "\t\tFilename: " << filenames[filenames.size()-1] << '\n';
                    }else if(childName == "bsdf"){
                        if(getString(child.attribute("type")) != "diffuse")
                            throw std::runtime_error("Invalid Material. Only diffuse is supported.");

                        const auto color = child.child("rgb");
                        if(getString(color.attribute("name")) != "reflectance")
                            throw std::runtime_error("Invalid Tag found for material.");
                        std::cout << "\t\tMaterial:\n";
                        std::cout << "\t\t\tBSDF: DIFFUSE\n";
                        bsdfs.emplace_back(Material::DIFFUSE, getVector3f(color, "value", "\t\t\t"));

                    }else{
                        throw std::runtime_error("Invalid Tag found for shape.");
                    }
                }
            }else{
                    throw std::runtime_error("Unrecognized node name while parsing XML.");
            }
        }
    }

    [[nodiscard]] Vector3f inline getVector3f(const pugi::xml_node &node, const std::string &name, const std::string &indents=""){
        const std::string targetS = getString(node.attribute(name.c_str()));
        std::cout << indents << name << ": {" << targetS << "}\n";
        return Vector3f{targetS};
    }

    [[nodiscard]] std::string getString(const pugi::xml_attribute& attrib){
        if(attrib.value()[0] == '$')
            return map.at(std::string(attrib.value()).substr(1));

        return attrib.value();
    }

//private:
    std::unordered_map<std::string ,std::string> map;
    Vector3f target, origin, up;
    int samplePerPixel;
    std::vector<std::string> filenames;
    std::vector<BSDF> bsdfs;
    int width, height;
};


//[[nodiscard]] SceneRepresentation loadScene(const std::string &filename){
//
//    return SceneRepresentation{filename};
//}