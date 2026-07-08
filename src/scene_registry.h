#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "sandbox_api.h"

struct SceneDef {
    std::string name;
    std::string kernel;                     // which .so to load
    std::string yaml_path;
    std::vector<float> param_overrides;     // indexed by param buffer_offset
    bool has_overrides = false;
};

class SceneRegistry {
public:
    explicit SceneRegistry(std::string scenes_dir);

    void rescan();
    const std::vector<SceneDef>& all() const { return scenes_; }
    const SceneDef* find(const std::string& name) const;

    // Merge YAML overrides into a param buffer initialized from kernel defaults
    void apply_params(const SceneDef& scene, const KernelDesc& desc,
                      float* out_buffer, size_t buffer_size) const;

private:
    std::string scenes_dir_;
    std::vector<SceneDef> scenes_;

    void load_yaml(const std::string& path, SceneDef& def);
};
