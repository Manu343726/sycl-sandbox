#include "scene_registry.h"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace fs = std::filesystem;

SceneRegistry::SceneRegistry(std::string scenes_dir)
    : scenes_dir_(std::move(scenes_dir)) {
    rescan();
}

void SceneRegistry::rescan() {
    scenes_.clear();
    if (!fs::is_directory(scenes_dir_)) {
        fprintf(stderr, "[scenes] directory not found: %s\n", scenes_dir_.c_str());
        return;
    }
    for (auto& entry : fs::directory_iterator(scenes_dir_)) {
        if (entry.path().extension() == ".yaml" ||
            entry.path().extension() == ".yml") {
            SceneDef def;
            def.yaml_path = entry.path().string();
            load_yaml(def.yaml_path, def);
            if (!def.name.empty() && !def.kernel.empty())
                scenes_.push_back(std::move(def));
            else
                fprintf(stderr, "[scenes] skipping %s (missing name/kernel)\n",
                        def.yaml_path.c_str());
        }
    }
    fprintf(stderr, "[scenes] loaded %zu scene(s)\n", scenes_.size());
}

const SceneDef* SceneRegistry::find(const std::string& name) const {
    for (auto& s : scenes_)
        if (s.name == name) return &s;
    return nullptr;
}

void SceneRegistry::load_yaml(const std::string& path, SceneDef& def) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        def.name   = root["name"].as<std::string>("");
        def.kernel = root["kernel"].as<std::string>("");

        auto params_node = root["params"];
        if (params_node) {
            // We store raw floats, but we need the kernel's ParamMeta to
            // know how to interpret them. That happens in apply_params().
            // Here we just remember that overrides exist and will be
            // resolved when apply_params is called with the kernel desc.
            def.has_overrides = true;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[scenes] YAML error %s: %s\n", path.c_str(), e.what());
    }
}

void SceneRegistry::apply_params(const SceneDef& scene,
                                 const KernelDesc& desc,
                                 float* out_buffer,
                                 size_t buffer_size) const {
    // 1. Fill buffer with kernel defaults
    for (int i = 0; i < desc.param_count; i++) {
        const auto& p = desc.params[i];
        auto* dst = reinterpret_cast<char*>(out_buffer) + p.buffer_offset;
        switch (p.type) {
        case ParamType::FLOAT:
            std::memcpy(dst, &p.default_f, 4); break;
        case ParamType::INT:
        case ParamType::ENUM:
            std::memcpy(dst, &p.default_i, 4); break;
        case ParamType::BOOL:
            std::memcpy(dst, &p.default_b, 4); break;
        case ParamType::COLOR_RGB:
            std::memcpy(dst, &p.default_c3, 12); break;
        case ParamType::COLOR_RGBA:
            std::memcpy(dst, &p.default_c4, 16); break;
        case ParamType::VEC3:
            std::memcpy(dst, &p.default_c3, 12); break;
        }
    }

    // 2. Overlay YAML values
    if (!scene.has_overrides) return;
    try {
        YAML::Node root = YAML::LoadFile(scene.yaml_path);
        auto params_node = root["params"];
        if (!params_node) return;

        for (int i = 0; i < desc.param_count; i++) {
            const auto& p = desc.params[i];
            auto val = params_node[p.name];
            if (!val) continue;

            auto* dst = reinterpret_cast<char*>(out_buffer) + p.buffer_offset;
            switch (p.type) {
            case ParamType::FLOAT:
                *reinterpret_cast<float*>(dst) = val.as<float>();
                break;
            case ParamType::INT:
            case ParamType::ENUM:
                *reinterpret_cast<int32_t*>(dst) = val.as<int32_t>();
                break;
            case ParamType::BOOL:
                *reinterpret_cast<int32_t*>(dst) = val.as<bool>() ? 1 : 0;
                break;
            case ParamType::COLOR_RGB: {
                auto v = val.as<std::vector<float>>();
                if (v.size() >= 3)
                    std::memcpy(dst, v.data(), 12);
                break;
            }
            case ParamType::COLOR_RGBA: {
                auto v = val.as<std::vector<float>>();
                if (v.size() >= 4)
                    std::memcpy(dst, v.data(), 16);
                break;
            }
            case ParamType::VEC3: {
                auto v = val.as<std::vector<float>>();
                if (v.size() >= 3)
                    std::memcpy(dst, v.data(), 12);
                break;
            }
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[scenes] error applying overrides: %s\n", e.what());
    }
}
