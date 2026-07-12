#include "registry.h"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

SceneRegistry::SceneRegistry(std::string scenes_dir) : scenes_dir_(std::move(scenes_dir)) {
    rescan();
}

void SceneRegistry::rescan() {
    scenes_.clear();
    if ( !fs::is_directory(scenes_dir_) ) {
        spdlog::error("[scenes] directory not found: {}", scenes_dir_);
        return;
    }
    for ( auto &entry : fs::directory_iterator(scenes_dir_) ) {
        if ( entry.path().extension() == ".yaml" || entry.path().extension() == ".yml" ) {
            SceneDef def;
            def.yaml_path = entry.path().string();
            load_yaml(def.yaml_path, def);
            if ( !def.name.empty() && !def.kernel.empty() ) {
                scenes_.push_back(std::move(def));
            } else {
                spdlog::warn("[scenes] skipping {} (missing name/kernel)", def.yaml_path);
            }
        }
    }
    spdlog::info("[scenes] loaded {} scene(s)", scenes_.size());
}

const SceneDef *SceneRegistry::find(const std::string &name) const {
    for ( auto &s : scenes_ ) {
        if ( s.name == name ) {
            return &s;
        }
    }
    return nullptr;
}

void SceneRegistry::load_yaml(const std::string &path, SceneDef &def) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        def.name = root["name"].as<std::string>("");
        def.kernel = root["kernel"].as<std::string>("");

        auto params_node = root["params"];
        if ( params_node ) {
            def.has_overrides = true;
        }
    } catch ( const std::exception &e ) {
        spdlog::error("[scenes] YAML error {}: {}", path, e.what());
    }
}
