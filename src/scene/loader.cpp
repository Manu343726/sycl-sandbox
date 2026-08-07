#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/profiler.h>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <random>
#include <cmath>
#include <cstring>

using namespace rt;
using rt::hittables::Sphere;
using rt::hittables::Quad;
using rt::hittables::Box;
using rt::hittables::Triangle;
using rt::materials::lambertian;
using rt::materials::metal;
using rt::materials::dielectric;
using rt::materials::diffuse_light;
using rt::materials::textured_lambertian;

namespace scene_loader {

// ── Value::array_length() ───────────────────────────────────────────

size_t Value::array_length() const {
    switch (type) {
        case ValueType::IntArray:    return int_arr.size();
        case ValueType::FloatArray:  return float_arr.size();
        case ValueType::Vec3Array:   return vec3_arr.size();
        case ValueType::StringArray: return string_arr.size();
        default:                     return 0;
    }
}

// ── resolve_reference ───────────────────────────────────────────────

const Value* resolve_reference(const std::string& name, const ResolvedScene& config) {
    // Check params first
    auto pit = config.params.find(name);
    if (pit != config.params.end()) return &pit->second;
    // Then data sources
    auto dit = config.data_sources.find(name);
    if (dit != config.data_sources.end()) return &dit->second;
    return nullptr;
}

// ── Auto camera param descriptors ─────────────────────────────────────

std::vector<ParamDescriptor> auto_camera_params_2d() {
    std::vector<ParamDescriptor> out;
    out.push_back({
        .name = "center_x",
        .description = "Pan X — horizontal offset of the view centre",
        .type = ParamType::FLOAT,
        .category = ParamCategory::Camera2D,
        .default_f = 0.0f,
        .is_camera_2d_center = true,
    });
    out.push_back({
        .name = "center_y",
        .description = "Pan Y — vertical offset of the view centre",
        .type = ParamType::FLOAT,
        .category = ParamCategory::Camera2D,
        .default_f = 0.0f,
        .is_camera_2d_center = true,
    });
    out.push_back({
        .name = "zoom",
        .description = "Zoom level (larger = more zoomed in)",
        .type = ParamType::FLOAT,
        .category = ParamCategory::Camera2D,
        .default_f = 1.0f,
        .has_range = true,
        .range_min_f = 0.001f,
        .range_max_f = 1000.0f,
        .is_camera_2d_zoom = true,
    });
    return out;
}

std::vector<ParamDescriptor> auto_camera_params_3d() {
    std::vector<ParamDescriptor> out;
    out.push_back({
        .name = "cam_eye",
        .description = "Camera position in world space",
        .type = ParamType::VEC3,
        .category = ParamCategory::Camera3D,
        .default_c3 = {0.0f, 1.5f, 4.5f},
        .buffer_size = 12,
        .is_camera_eye = true,
    });
    out.push_back({
        .name = "cam_at",
        .description = "Look-at target position",
        .type = ParamType::VEC3,
        .category = ParamCategory::Camera3D,
        .default_c3 = {0.0f, 1.2f, 0.0f},
        .buffer_size = 12,
        .is_camera_target = true,
    });
    out.push_back({
        .name = "cam_up",
        .description = "Camera up vector",
        .type = ParamType::VEC3,
        .category = ParamCategory::Camera3D,
        .default_c3 = {0.0f, 1.0f, 0.0f},
        .buffer_size = 12,
        .is_camera_up = true,
    });
    out.push_back({
        .name = "cam_fov",
        .description = "Vertical field of view (degrees)",
        .type = ParamType::FLOAT,
        .category = ParamCategory::Camera3D,
        .default_f = 35.0f,
        .has_range = true,
        .range_min_f = 1.0f,
        .range_max_f = 120.0f,
        .is_camera_fov = true,
    });
    out.push_back({
        .name = "cam_aperture",
        .description = "Depth-of-field aperture size (0 = no DOF)",
        .type = ParamType::FLOAT,
        .category = ParamCategory::Camera3D,
        .default_f = 0.0f,
        .has_range = true,
        .range_min_f = 0.0f,
        .range_max_f = 1.0f,
        .is_camera_aperture = true,
    });
    return out;
}

// ── ParamDescriptor helpers ───────────────────────────────────────────

static uint32_t param_buffer_size_for_type(ParamType t) {
    switch (t) {
        case ParamType::FLOAT:      return 4;
        case ParamType::INT:        return 4;
        case ParamType::BOOL:       return 4;
        case ParamType::COLOR_RGB:  return 12;
        case ParamType::COLOR_RGBA: return 16;
        case ParamType::VEC3:       return 12;
        case ParamType::ENUM:       return 4;
    }
    return 4;
}

// ── parse_param_type ─────────────────────────────────────────────────

ParamType parse_param_type(const std::string &type_str) {
    if (type_str == "float") return ParamType::FLOAT;
    if (type_str == "int") return ParamType::INT;
    if (type_str == "bool") return ParamType::BOOL;
    if (type_str == "color" || type_str == "color_rgb") return ParamType::COLOR_RGB;
    if (type_str == "color_rgba") return ParamType::COLOR_RGBA;
    if (type_str == "vec3") return ParamType::VEC3;
    if (type_str == "enum") return ParamType::ENUM;
    return ParamType::FLOAT;
}

// ── auto_standard_stats ──────────────────────────────────────────────

std::vector<StatDescriptor> auto_standard_stats() {
    std::vector<StatDescriptor> out;
    out.push_back({
        .name = "fps",
        .description = "Frames per second (instantaneous)",
        .type = ParamType::FLOAT,
        .viz = VisualizationHint::Graph,
        .category = ParamCategory::Render,
        .has_range = true,
        .range_min_f = 0.0f,
        .range_max_f = 500.0f,
    });
    out.push_back({
        .name = "frame_time_ms",
        .description = "Frame time in milliseconds (render + compositing)",
        .type = ParamType::FLOAT,
        .viz = VisualizationHint::Graph,
        .category = ParamCategory::Render,
        .has_range = true,
        .range_min_f = 0.0f,
        .range_max_f = 100.0f,
    });
    out.push_back({
        .name = "spp",
        .description = "Current samples per pixel",
        .type = ParamType::INT,
        .viz = VisualizationHint::Counter,
        .category = ParamCategory::Render,
    });
    out.push_back({
        .name = "pixel_count",
        .description = "Render resolution in megapixels",
        .type = ParamType::FLOAT,
        .viz = VisualizationHint::Text,
        .category = ParamCategory::Render,
    });
    out.push_back({
        .name = "device_memory_mb",
        .description = "Device memory usage (MB)",
        .type = ParamType::FLOAT,
        .viz = VisualizationHint::Gauge,
        .category = ParamCategory::Render,
        .has_range = true,
        .range_min_f = 0.0f,
        .range_max_f = 1024.0f,
    });
    out.push_back({
        .name = "host_memory_mb",
        .description = "Host-mapped memory usage (MB)",
        .type = ParamType::FLOAT,
        .viz = VisualizationHint::Gauge,
        .category = ParamCategory::Render,
        .has_range = true,
        .range_min_f = 0.0f,
        .range_max_f = 512.0f,
    });
    return out;
}

// ── load_scene_descriptor ───────────────────────────────────────────

SceneDescriptor load_scene_descriptor(const std::string &yaml_path) {
    PROFILER_FUNCTION();
    SceneDescriptor desc;
    desc.yaml_path = yaml_path;

    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const std::exception &e) {
        spdlog::error("[scene] failed to load '{}': {}", yaml_path, e.what());
        desc.name = "ERROR: " + std::string(e.what());
        return desc;
    }

    desc.name = root["name"].as<std::string>("Unnamed");
    desc.kernel_name = root["kernel"].as<std::string>("");
    desc.scene_type = root["scene_type"].as<std::string>("3d");

    // ── 1. Auto-generated render params ─────────────────────────────
    {
        ParamDescriptor spp;
        spp.name = "spp_frame";
        spp.description = "Samples per frame (1 = one pass, higher = progressive)";
        spp.type = ParamType::INT;
        spp.category = ParamCategory::Render;
        spp.default_i = 1;
        spp.has_range = true;
        spp.range_min_i = 1;
        spp.range_max_i = 64;
        desc.params.push_back(std::move(spp));
    }
    {
        ParamDescriptor mb;
        mb.name = "max_bounces";
        mb.description = "Maximum ray bounce depth";
        mb.type = ParamType::INT;
        mb.category = ParamCategory::Render;
        mb.default_i = 10;
        mb.has_range = true;
        mb.range_min_i = 1;
        mb.range_max_i = 50;
        desc.params.push_back(std::move(mb));
    }

    // ── 1b. Auto-generated animation params (tick / time) ──────────
    {
        ParamDescriptor tick;
        tick.name = "tick";
        tick.description = "Frame counter — increments each sample (like Shadertoy iFrame)";
        tick.type = ParamType::INT;
        tick.category = ParamCategory::Render;
        tick.default_i = 0;
        desc.params.push_back(std::move(tick));
    }
    {
        ParamDescriptor tm;
        tm.name = "time";
        tm.description = "Elapsed time in seconds since scene start (like Shadertoy iTime)";
        tm.type = ParamType::FLOAT;
        tm.category = ParamCategory::Render;
        tm.default_f = 0.0f;
        desc.params.push_back(std::move(tm));
    }

    // ── 1c. Auto-generated debug params ───────────────────────────
    // Standard debugging switches for the kernel processing and rendering
    // pipeline.  Consumed by the render thread; see render_loop.h.
    {
        ParamDescriptor dbg;
        dbg.name = "debug_colorchecker";
        dbg.description =
            "Bypass kernel execution and render a standard 24-patch "
            "ColorChecker chart instead. The tone-mapping step and the "
            "display swapchain continue as usual.";
        dbg.type = ParamType::BOOL;
        dbg.category = ParamCategory::Render;
        dbg.default_b = false;
        desc.params.push_back(std::move(dbg));
    }
    {
        ParamDescriptor dbg;
        dbg.name = "debug_colorchecker_raw";
        dbg.description =
            "Bypass kernel execution AND the tone-mapping step; display "
            "the ColorChecker chart directly (raw sRGB, no gamma).";
        dbg.type = ParamType::BOOL;
        dbg.category = ParamCategory::Render;
        dbg.default_b = false;
        desc.params.push_back(std::move(dbg));
    }

    // ── 1d. Auto-generated tone-map params ─────────────────────────
    // Standard display-pipeline settings, consumed by the render
    // thread (see render_loop.h).  Scenes may override the defaults
    // through the YAML `params:` section (merged in step 3).
    {
        ParamDescriptor tm;
        tm.name = "tonemap_enabled";
        tm.description =
            "Enable the tone-mapping stage of the display pipeline. "
            "When disabled, accumulated linear values are normalized "
            "and hard-clamped to [0,1] with no operator or gamma.";
        tm.type = ParamType::BOOL;
        tm.category = ParamCategory::Render;
        tm.default_b = false;
        desc.params.push_back(std::move(tm));
    }
    {
        ParamDescriptor tm;
        tm.name = "tonemap_operator";
        tm.description =
            "Tone-map operator: 0 = Reinhard (x/(1+x)), "
            "1 = ACES fitted (Narkowicz 2015), "
            "2 = Filmic (Hable / Uncharted 2 curve).";
        tm.type = ParamType::ENUM;
        tm.category = ParamCategory::Render;
        tm.default_i = 0;
        tm.enum_options = {"Reinhard", "ACES (fitted)", "Filmic (Hable)"};
        desc.params.push_back(std::move(tm));
    }
    {
        ParamDescriptor tm;
        tm.name = "tonemap_exposure";
        tm.description =
            "Exposure multiplier applied to the linear HDR value "
            "before the tone-map operator.";
        tm.type = ParamType::FLOAT;
        tm.category = ParamCategory::Render;
        tm.default_f = 1.0f;
        tm.has_range = true;
        tm.range_min_f = 0.0f;
        tm.range_max_f = 8.0f;
        desc.params.push_back(std::move(tm));
    }
    {
        ParamDescriptor tm;
        tm.name = "tonemap_gamma";
        tm.description =
            "Display gamma for the final sRGB-like correction "
            "(value = pow(clamp(c), 1/gamma)).";
        tm.type = ParamType::FLOAT;
        tm.category = ParamCategory::Render;
        tm.default_f = 2.2f;
        tm.has_range = true;
        tm.range_min_f = 1.0f;
        tm.range_max_f = 4.0f;
        desc.params.push_back(std::move(tm));
    }

    // ── 2. Auto-generated camera params  ────────────────────────────
    if (desc.scene_type == "2d") {
        for (auto &p : auto_camera_params_2d())
            desc.params.push_back(std::move(p));
    } else {
        for (auto &p : auto_camera_params_3d())
            desc.params.push_back(std::move(p));
    }

    // ── 3. Kernel-specific params from YAML ─────────────────────────
    // YAML-declared params may also OVERRIDE the default value, range,
    // description, or enum choices of an auto-generated standard param
    // (e.g. `tonemap_enabled: true` in a scene that wants tone-mapping
    // on by default).  The merge keeps the auto-generated layout/type.
    auto params_node = root["params"];
    if (params_node && params_node.IsMap()) {
        for (auto it = params_node.begin(); it != params_node.end(); ++it) {
            std::string pname = it->first.as<std::string>();

            YAML::Node pval = it->second;
            ParamDescriptor pd;
            pd.name = pname;
            pd.category = ParamCategory::Kernel;
            bool has_default = false;

            if (pval.IsMap() && pval["type"]) {
                std::string ptype = pval["type"].as<std::string>();
                pd.description = pval["description"].as<std::string>("");

                if (ptype == "int") {
                    pd.type = ParamType::INT;
                    has_default = pval["default"].IsDefined();
                    pd.default_i = pval["default"] ? pval["default"].as<int>() : 0;
                    if (pval["range"] && pval["range"].size() == 2) {
                        pd.has_range = true;
                        pd.range_min_i = pval["range"][0].as<int>();
                        pd.range_max_i = pval["range"][1].as<int>();
                    }
                } else if (ptype == "float") {
                    pd.type = ParamType::FLOAT;
                    has_default = pval["default"].IsDefined();
                    pd.default_f = pval["default"] ? pval["default"].as<float>() : 0.0f;
                    if (pval["range"] && pval["range"].size() == 2) {
                        pd.has_range = true;
                        pd.range_min_f = pval["range"][0].as<float>();
                        pd.range_max_f = pval["range"][1].as<float>();
                    }
                } else if (ptype == "color" || ptype == "color_rgb") {
                    pd.type = ParamType::COLOR_RGB;
                    pd.buffer_size = 12;
                    has_default = pval["default"].IsDefined();
                    if (pval["default"] && pval["default"].IsSequence() && pval["default"].size() >= 3) {
                        pd.default_c3 = {pval["default"][0].as<float>(),
                                         pval["default"][1].as<float>(),
                                         pval["default"][2].as<float>()};
                    }
                } else if (ptype == "vec3") {
                    pd.type = ParamType::VEC3;
                    pd.buffer_size = 12;
                    has_default = pval["default"].IsDefined();
                    if (pval["default"] && pval["default"].IsSequence() && pval["default"].size() >= 3) {
                        pd.default_c3 = {pval["default"][0].as<float>(),
                                         pval["default"][1].as<float>(),
                                         pval["default"][2].as<float>()};
                    }
                } else if (ptype == "bool") {
                    pd.type = ParamType::BOOL;
                    has_default = pval["default"].IsDefined();
                    pd.default_b = pval["default"] ? pval["default"].as<bool>() : false;
                } else if (ptype == "enum") {
                    pd.type = ParamType::ENUM;
                    has_default = pval["default"].IsDefined();
                    pd.default_i = pval["default"] ? pval["default"].as<int>() : 0;
                    if (pval["options"] && pval["options"].IsSequence()) {
                        for (auto opt : pval["options"]) {
                            pd.enum_options.push_back(opt.as<std::string>());
                        }
                    }
                } else if (ptype == "string") {
                    pd.type = ParamType::ENUM;
                    pd.default_i = 0;
                    pd.description = pval["description"].as<std::string>("");
                }
            } else {
                pd.description = "";
                if (pval.IsScalar()) {
                    has_default = true;
                    int int_val;
                    try { int_val = pval.as<int>(); pd.type = ParamType::INT; pd.default_i = int_val; }
                    catch (...) {
                        float flt_val = pval.as<float>();
                        pd.type = ParamType::FLOAT;
                        pd.default_f = flt_val;
                    }
                } else if (pval.IsSequence() && pval.size() == 3) {
                    has_default = true;
                    pd.type = ParamType::COLOR_RGB;
                    pd.buffer_size = 12;
                    pd.default_c3 = {pval[0].as<float>(), pval[1].as<float>(), pval[2].as<float>()};
                }
            }
            pd.buffer_size = param_buffer_size_for_type(pd.type);

            // ── Merge into an auto-generated standard param? ────────
            if ( auto *existing = desc.find(pname) ) {
                // Keep the auto-generated layout/type/category; apply
                // the YAML-provided default/range/description/options.
                spdlog::info("[scene] '{}' overrides standard param defaults", pname);
                if ( has_default ) {
                    // The YAML parse may infer a different scalar type
                    // (2 vs 2.0); convert numerically to the standard
                    // param's declared type.
                    switch ( existing->type ) {
                        case ParamType::INT:
                        case ParamType::ENUM:
                            existing->default_i =
                                (pd.type == ParamType::FLOAT) ? (int)pd.default_f : pd.default_i;
                            break;
                        case ParamType::FLOAT:
                            existing->default_f =
                                (pd.type == ParamType::FLOAT) ? pd.default_f : (float)pd.default_i;
                            break;
                        case ParamType::BOOL:
                            existing->default_b =
                                (pd.type == ParamType::BOOL) ? pd.default_b
                                                            : (pd.default_f != 0.0f || pd.default_i != 0);
                            break;
                        case ParamType::COLOR_RGB:
                        case ParamType::VEC3:
                            existing->default_c3 = pd.default_c3;
                            break;
                        case ParamType::COLOR_RGBA:
                            existing->default_c4 = {pd.default_c3[0], pd.default_c3[1],
                                                    pd.default_c3[2], 1.0f};
                            break;
                    }
                }
                if ( pd.has_range ) {
                    existing->has_range = true;
                    existing->range_min_f = pd.range_min_f;
                    existing->range_max_f = pd.range_max_f;
                    existing->range_min_i = pd.range_min_i;
                    existing->range_max_i = pd.range_max_i;
                }
                if ( !pd.description.empty() )
                    existing->description = pd.description;
                if ( !pd.enum_options.empty() )
                    existing->enum_options = pd.enum_options;
                continue;
            }

            desc.params.push_back(std::move(pd));
        }
    }

    // ── 4. Statistics (auto-generated + YAML-declared) ───────────────
    for (auto &s : auto_standard_stats())
        desc.stats.push_back(s);

    auto stats_node = root["statistics"];
    if (stats_node && stats_node.IsMap()) {
        for (auto it = stats_node.begin(); it != stats_node.end(); ++it) {
            std::string sname = it->first.as<std::string>();

            if (sname == "fps" || sname == "frame_time_ms" || sname == "spp" ||
                sname == "pixel_count" || sname == "device_memory_mb" ||
                sname == "host_memory_mb") {
                spdlog::warn("[scene] stat '{}' conflicts with auto-generated stat, skipping", sname);
                continue;
            }

            YAML::Node sval = it->second;
            StatDescriptor sd;
            sd.name = sname;
            sd.category = ParamCategory::Kernel;

            if (sval.IsMap() && sval["type"]) {
                std::string stype = sval["type"].as<std::string>();
                sd.type = parse_param_type(stype);
                sd.description = sval["description"].as<std::string>("");

                std::string sviz = sval["viz"].as<std::string>("auto");
                if (sviz == "text")          sd.viz = VisualizationHint::Text;
                else if (sviz == "gauge")    sd.viz = VisualizationHint::Gauge;
                else if (sviz == "graph")    sd.viz = VisualizationHint::Graph;
                else if (sviz == "counter")  sd.viz = VisualizationHint::Counter;
                else if (sviz == "flag")     sd.viz = VisualizationHint::Flag;
                else                         sd.viz = VisualizationHint::Auto;

                if (sval["range"] && sval["range"].size() == 2) {
                    sd.has_range = true;
                    sd.range_min_f = sval["range"][0].as<float>();
                    sd.range_max_f = sval["range"][1].as<float>();
                }
            } else {
                sd.description = "";
                if (sval.IsScalar()) {
                    try { sval.as<float>(); sd.type = ParamType::FLOAT; }
                    catch (...) { sd.type = ParamType::INT; }
                } else if (sval.IsSequence() && sval.size() == 3) {
                    sd.type = ParamType::VEC3;
                }
            }
            sd.buffer_size = param_buffer_size_for_type(sd.type);
            desc.stats.push_back(std::move(sd));
        }
    }

    spdlog::info("[scene] descriptor '{}' ({}D, {} params, {} stats)",
                 desc.name, desc.scene_type, desc.params.size(), desc.stats.size());
    return desc;
}

// ── SceneDescriptor members ─────────────────────────────────────────

void SceneDescriptor::build_layout() {
    PROFILER_FUNCTION();
    uint32_t offset = 0;

    for (auto &p : params) {
        p.buffer_size = param_buffer_size_for_type(p.type);
        p.buffer_offset = offset;
        offset += p.buffer_size;
    }

    size_t total_floats = offset / sizeof(float);
    default_buffer.resize(total_floats, 0.0f);

    for (auto &p : params) {
        float *buf = default_buffer.data() + p.buffer_offset / sizeof(float);
        switch (p.type) {
            case ParamType::FLOAT:
                buf[0] = p.default_f;
                break;
            case ParamType::INT:
            case ParamType::ENUM:
                buf[0] = (float)p.default_i;
                break;
            case ParamType::BOOL:
                buf[0] = p.default_b ? 1.0f : 0.0f;
                break;
            case ParamType::COLOR_RGB:
            case ParamType::VEC3:
                buf[0] = p.default_c3[0];
                buf[1] = p.default_c3[1];
                buf[2] = p.default_c3[2];
                break;
            case ParamType::COLOR_RGBA:
                buf[0] = p.default_c4[0];
                buf[1] = p.default_c4[1];
                buf[2] = p.default_c4[2];
                buf[3] = p.default_c4[3];
                break;
        }
    }

    lookup_entries.clear();
    for (auto &p : params) {
        rt::ParamEntry entry;
        entry.name_hash = rt::param_name_hash(p.name.c_str());
        entry.byte_offset = p.buffer_offset;
        entry.type = p.type;
        lookup_entries.push_back(entry);
    }

    current_buffer = default_buffer;

    // ── Build stat buffer layout ─────────────────────────────────────
    uint32_t stat_offset = 0;
    for (auto &s : stats) {
        s.buffer_size = param_buffer_size_for_type(s.type);
        s.buffer_offset = stat_offset;
        stat_offset += s.buffer_size;
    }

    size_t stat_floats = stat_offset / sizeof(float);
    default_stat_buffer.resize(stat_floats, 0.0f);

    stat_lookup_entries.clear();
    for (auto &s : stats) {
        rt::StatEntry entry;
        entry.name_hash = rt::stat_name_hash(s.name.c_str());
        entry.byte_offset = s.buffer_offset;
        entry.type = s.type;
        stat_lookup_entries.push_back(entry);
    }

    current_stat_buffer = default_stat_buffer;
}

const ParamDescriptor *SceneDescriptor::find(const std::string &name) const {
    for (auto &p : params) {
        if (p.name == name) return &p;
    }
    return nullptr;
}

ParamDescriptor *SceneDescriptor::find(const std::string &name) {
    for (auto &p : params) {
        if (p.name == name) return &p;
    }
    return nullptr;
}

void SceneDescriptor::reset_current_to_defaults() {
    current_buffer = default_buffer;
}

ParamRef SceneDescriptor::find_param_ref(const std::string &name) {
    auto *pd = find(name);
    if (!pd) return {};
    return {pd, current_buffer.data() + pd->buffer_offset / sizeof(float)};
}

ParamRef SceneDescriptor::find_param_ref(const std::string &name) const {
    const auto *pd = find(name);
    if (!pd) return {};
    return {pd, const_cast<float *>(current_buffer.data() + pd->buffer_offset / sizeof(float))};
}

StatRef SceneDescriptor::find_stat_ref(const std::string &name) {
    auto *sd = find_stat(name);
    if (!sd) return {};
    return {sd, current_stat_buffer.data() + sd->buffer_offset / sizeof(float)};
}

StatRef SceneDescriptor::find_stat_ref(const std::string &name) const {
    const auto *sd = find_stat(name);
    if (!sd) return {};
    return {sd, const_cast<float *>(current_stat_buffer.data() + sd->buffer_offset / sizeof(float))};
}

void SceneDescriptor::sync_buffer_to_lookup() {}

rt::ParamLookup SceneDescriptor::make_lookup() const {
    rt::ParamLookup lookup;
    lookup.set_buffer(current_buffer.data());
    lookup.set_entries(lookup_entries.data(), (int)lookup_entries.size());
    return lookup;
}

const StatDescriptor *SceneDescriptor::find_stat(const std::string &name) const {
    for (auto &s : stats) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

StatDescriptor *SceneDescriptor::find_stat(const std::string &name) {
    for (auto &s : stats) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

namespace {

// ── Helpers ──────────────────────────────────────────────────────────

bool is_ref(const std::string& s) {
    return !s.empty() && s[0] == '$';
}

std::string ref_name(const std::string& s) {
    return s.substr(1);
}

// ── RNG ──────────────────────────────────────────────────────────────

static std::mt19937& rng() {
    static std::mt19937 gen{42};
    return gen;
}

static float random_float(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng());
}

// ── YAML → Value ────────────────────────────────────────────────────

Value parse_scalar(const YAML::Node& node) {
    Value v;
    switch (node.Type()) {
        case YAML::NodeType::Scalar: {
            try {
                int int_val = node.as<int>();
                v.type = ValueType::Int;
                v.int_val = int_val;
                return v;
            } catch (...) {}
            try {
                float float_val = node.as<float>();
                v.type = ValueType::Float;
                v.float_val = float_val;
                return v;
            } catch (...) {}
            v.type = ValueType::String;
            v.string_val = node.as<std::string>();
            return v;
        }
        case YAML::NodeType::Sequence: {
            if (node.size() == 3) {
                v.type = ValueType::Vec3;
                v.vec3_val = {node[0].as<float>(), node[1].as<float>(), node[2].as<float>()};
                return v;
            }
            std::vector<float> arr;
            for (size_t i = 0; i < node.size(); i++) {
                arr.push_back(node[i].as<float>());
            }
            v.type = ValueType::FloatArray;
            v.float_arr = std::move(arr);
            return v;
        }
        default:
            break;
    }
    return v;
}

// ── Deep input resolver ─────────────────────────────────────────────

Value resolve_value(const YAML::Node& node, const ResolvedScene& config) {
    if (node.IsScalar()) {
        std::string s = node.as<std::string>();
        if (is_ref(s)) {
            auto* resolved = resolve_reference(ref_name(s), config);
            if (resolved) return *resolved;
            spdlog::warn("[scene_loader] unresolved reference '{}' in '{}'",
                         s, config.yaml_path);
        }
        return parse_scalar(node);
    }
    if (node.IsSequence()) {
        if (node.size() == 3) {
            bool all_scalar = true;
            for (size_t i = 0; i < 3 && all_scalar; i++) {
                all_scalar = all_scalar && node[i].IsScalar();
            }
            if (all_scalar) {
                auto v0 = resolve_value(node[0], config);
                auto v1 = resolve_value(node[1], config);
                auto v2 = resolve_value(node[2], config);
                Value result;
                result.type = ValueType::Vec3;
                result.vec3_val = {v0.float_val, v1.float_val, v2.float_val};
                return result;
            }
        }
        std::vector<float> arr;
        for (size_t i = 0; i < node.size(); i++) {
            auto elem = resolve_value(node[i], config);
            arr.push_back(elem.float_val);
        }
        Value result;
        result.type = ValueType::FloatArray;
        result.float_arr = std::move(arr);
        return result;
    }
    return {};
}

float resolve_float(const YAML::Node& node, const ResolvedScene& config) {
    return resolve_value(node, config).float_val;
}

int resolve_int(const YAML::Node& node, const ResolvedScene& config) {
    return resolve_value(node, config).int_val;
}

float3 resolve_vec3(const YAML::Node& node, const ResolvedScene& config) {
    return resolve_value(node, config).vec3_val;
}

// ── Data source generators ──────────────────────────────────────────

Value generate_random_range(const YAML::Node& inputs, const ResolvedScene& config) {
    int count = resolve_int(inputs["count"], config);
    float3 min_v = resolve_vec3(inputs["min"], config);
    float3 max_v = resolve_vec3(inputs["max"], config);

    bool is_vec3 = inputs["min"].IsSequence() && inputs["min"].size() == 3;

    if (is_vec3) {
        std::vector<float3> arr;
        arr.reserve(count);
        for (int i = 0; i < count; i++) {
            arr.push_back({
                random_float(min_v.x, max_v.x),
                random_float(min_v.y, max_v.y),
                random_float(min_v.z, max_v.z)
            });
        }
        Value v;
        v.type = ValueType::Vec3Array;
        v.vec3_arr = std::move(arr);
        return v;
    } else {
        std::vector<float> arr;
        arr.reserve(count);
        float mn = min_v.x;
        float mx = max_v.x;
        for (int i = 0; i < count; i++) {
            arr.push_back(random_float(mn, mx));
        }
        Value v;
        v.type = ValueType::FloatArray;
        v.float_arr = std::move(arr);
        return v;
    }
}

Value generate_weighted_choice(const YAML::Node& inputs, const ResolvedScene& config) {
    int count = resolve_int(inputs["count"], config);
    auto choices = inputs["choices"];

    struct Choice {
        YAML::Node value;
        float weight;
    };
    std::vector<Choice> choice_list;
    float total_weight = 0;
    for (size_t i = 0; i < choices.size(); i++) {
        float w = choices[i]["weight"].as<float>();
        choice_list.push_back({choices[i]["value"], w});
        total_weight += w;
    }

    bool is_string = !choice_list.empty() && choice_list[0].value.IsScalar();

    if (is_string) {
        std::vector<std::string> arr;
        arr.reserve(count);
        std::uniform_real_distribution<float> dist(0, total_weight);
        for (int i = 0; i < count; i++) {
            float r = dist(rng());
            float accum = 0;
            for (auto& ch : choice_list) {
                accum += ch.weight;
                if (r <= accum) {
                    arr.push_back(ch.value.as<std::string>());
                    break;
                }
            }
        }
        Value v;
        v.type = ValueType::StringArray;
        v.string_arr = std::move(arr);
        return v;
    } else {
        std::vector<int> arr;
        arr.reserve(count);
        std::uniform_real_distribution<float> dist(0, total_weight);
        for (int i = 0; i < count; i++) {
            float r = dist(rng());
            float accum = 0;
            for (auto& ch : choice_list) {
                accum += ch.weight;
                if (r <= accum) {
                    arr.push_back(ch.value.as<int>());
                    break;
                }
            }
        }
        Value v;
        v.type = ValueType::IntArray;
        v.int_arr = std::move(arr);
        return v;
    }
}

} // anonymous namespace

// ── load_and_resolve ────────────────────────────────────────────────

ResolvedScene load_and_resolve(const std::string &yaml_path) {
    PROFILER_FUNCTION();
    ResolvedScene config;
    config.yaml_path = yaml_path;

    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const std::exception &e) {
        spdlog::error("[scene] failed to load '{}': {}", yaml_path, e.what());
        return config;
    }

    config.name = root["name"].as<std::string>("Unnamed");

    // 1. Load data sources (generators)
    auto datasources = root["data"];
    if (datasources && datasources.IsMap()) {
        for (auto it = datasources.begin(); it != datasources.end(); ++it) {
            std::string ds_name = it->first.as<std::string>();
            YAML::Node ds = it->second;

            if (ds["random_range"]) {
                config.data_sources[ds_name] = generate_random_range(ds["random_range"], config);
            } else if (ds["weighted_choice"]) {
                config.data_sources[ds_name] = generate_weighted_choice(ds["weighted_choice"], config);
            } else {
                config.data_sources[ds_name] = parse_scalar(ds);
            }
        }
    }

    // 2. Load params (may reference data sources)
    auto params = root["params"];
    if (params && params.IsMap()) {
        for (auto it = params.begin(); it != params.end(); ++it) {
            YAML::Node pval = it->second;
            config.params[it->first.as<std::string>()] = parse_scalar(pval);
        }
    }

    return config;
}

// ── SceneBuilder helpers ────────────────────────────────────────────

using hittable_list = std::vector<rt::HitRecord>;

template<typename T>
T* alloc_hittable(hittable_list& list) {
    list.emplace_back();
    return reinterpret_cast<T*>(&list.back());
}

// ── build_scene ─────────────────────────────────────────────────────

void build_scene(rt::SceneBuilder& builder, const ResolvedScene& config) {
    PROFILER_FUNCTION();

    YAML::Node root;
    try {
        root = YAML::LoadFile(config.yaml_path);
    } catch (const std::exception& e) {
        spdlog::error("[scene] failed to load '{}': {}", config.yaml_path, e.what());
        return;
    }

    // Check for scene-level material
    auto mat_node = root["material"];
    std::string scene_material;
    auto mat_type_node = root["material_type"];
    std::string scene_material_type;
    if (mat_node) {
        if (mat_node.IsScalar()) {
            scene_material = mat_node.as<std::string>();
        } else if (mat_node.IsMap() && mat_node["type"]) {
            scene_material_type = mat_node["type"].as<std::string>();
        }
    }
    if (mat_type_node) {
        scene_material_type = mat_type_node.as<std::string>();
    }

    // Helper to build material from a material node
    auto build_material = [&](const YAML::Node& node) -> rt::Material {
        if (!node || !node.IsMap()) return lambertian(float3{1,1,1});
        std::string type = node["type"].as<std::string>("lambertian");

        // Resolve color (may be a reference)
        float3 col{0.8f, 0.8f, 0.8f};
        if (node["color"]) {
            auto col_val = resolve_value(node["color"], config);
            if (col_val.type == ValueType::Vec3)
                col = col_val.vec3_val;
        }

        if (type == "lambertian") {
            return lambertian(float3(col.x, col.y, col.z));
        } else if (type == "metal") {
            float roughness = node["roughness"] ? resolve_float(node["roughness"], config) : 0.0f;
            return metal(float3(col.x, col.y, col.z), roughness);
        } else if (type == "dielectric") {
            float index = node["index"] ? resolve_float(node["index"], config) : 1.5f;
            return dielectric(index);
        } else if (type == "diffuse_light") {
            float intensity = node["intensity"] ? resolve_float(node["intensity"], config) : 1.0f;
            return diffuse_light(scale(float3(col.x, col.y, col.z), intensity));
        } else if (type == "textured_lambertian") {
            // Texture-driven diffuse material.  The texture is named in
            // `texture` (currently "colorchecker" or "solid"); UVs come
            // from the hit point's parametric coordinates.  UV overflow
            // behavior is defined by the texture itself (the colorchecker
            // wraps and tiles infinitely).
            std::string texture = node["texture"].as<std::string>("colorchecker");
            float scale_u = node["scale_u"] ? resolve_float(node["scale_u"], config) : 1.0f;
            float scale_v = node["scale_v"] ? resolve_float(node["scale_v"], config) : 1.0f;
            if (texture == "colorchecker") {
                return textured_lambertian(rt::textures::ColorChecker(scale_u, scale_v));
            } else if (texture == "solid") {
                return textured_lambertian(rt::textures::SolidColor(float3(col.x, col.y, col.z)));
            }
            // Unknown texture: fall back to the colorchecker.
            return textured_lambertian(rt::textures::ColorChecker(scale_u, scale_v));
        }
        return lambertian(float3(col.x, col.y, col.z));
    };

    // ── Objects ──────────────────────────────────────────────────
    auto objects = root["objects"];
    if (objects && objects.IsSequence()) {
        for (size_t i = 0; i < objects.size(); i++) {
            auto obj = objects[i];
            std::string type = obj["type"].as<std::string>("sphere");

            // Portals are instanced as objects with a dummy material;
            // handle them before the material resolution below (an
            // optional `material:` node overrides the dummy, e.g.
            // diffuse_light for an emissive portal).
            if (type == "portal") {
                auto parse_shape = [&](const YAML::Node &n) -> rt::hittables::PortalShape {
                    if (!n || !n.IsMap()) {
                        spdlog::warn("[scene] portal shape missing, defaulting to unit quad");
                        return rt::hittables::Quad({0, 0, 0}, {1, 0, 0}, {0, 1, 0});
                    }
                    std::string stype = n["type"].as<std::string>("quad");
                    float3 center{0, 0, 0};
                    if (n["center"])
                        center = resolve_vec3(n["center"], config);
                    if (stype == "sphere") {
                        float radius = n["radius"] ? resolve_float(n["radius"], config) : 0.5f;
                        return rt::hittables::Sphere(center, radius);
                    } else if (stype == "triangle") {
                        float3 v0{0, 0, 0}, v1{1, 0, 0}, v2{0, 1, 0};
                        if (n["v0"]) v0 = resolve_vec3(n["v0"], config);
                        if (n["v1"]) v1 = resolve_vec3(n["v1"], config);
                        if (n["v2"]) v2 = resolve_vec3(n["v2"], config);
                        return rt::hittables::Triangle(v0, v1, v2);
                    }
                    float3 u{1, 0, 0}, v{0, 1, 0};
                    if (n["u"]) u = resolve_vec3(n["u"], config);
                    if (n["v"]) v = resolve_vec3(n["v"], config);
                    return rt::hittables::Quad(center, u, v);
                };

                if (obj["entry"] && obj["exit"]) {
                    auto entry = parse_shape(obj["entry"]);
                    auto exit = parse_shape(obj["exit"]);
                    if (obj["material"]) {
                        // Real material, e.g. diffuse_light for an
                        // emissive portal (glowing surface).
                        builder.add_portal(std::move(entry), std::move(exit),
                                           build_material(obj["material"]));
                    } else {
                        // Default: dummy white Lambertian, whose scatter()
                        // teleports the ray (pure window).
                        builder.add_portal(std::move(entry), std::move(exit));
                    }
                } else {
                    spdlog::warn("[scene] portal object #{} missing entry/exit, skipping", i);
                }
                continue;
            }

            // Resolve transform
            float3 center{0, 0, 0};
            if (obj["center"])
                center = resolve_vec3(obj["center"], config);

            // Build per-object material or use scene material
            rt::Material mat;
            if (obj["material"]) {
                mat = build_material(obj["material"]);
            } else if (!scene_material.empty()) {
                // Named material system - look up in materials section
                auto mats_node = root["materials"];
                if (mats_node && mats_node.IsMap() && mats_node[scene_material]) {
                    mat = build_material(mats_node[scene_material]);
                }
            } else if (!scene_material_type.empty()) {
                YAML::Node mat_node;
                mat_node["type"] = YAML::Node(scene_material_type);
                if (obj["color"]) mat_node["color"] = obj["color"];
                mat = build_material(mat_node);
            } else {
                mat = lambertian(float3(0.8f, 0.8f, 0.8f));
            }

            if (type == "sphere") {
                float radius = obj["radius"] ? resolve_float(obj["radius"], config) : 0.5f;
                builder.add(Sphere(center, radius), mat);
            } else if (type == "quad") {
                float3 u{1,0,0}, v{0,1,0};
                if (obj["u"]) u = resolve_vec3(obj["u"], config);
                if (obj["v"]) v = resolve_vec3(obj["v"], config);
                builder.add(Quad(center, u, v), mat);
            } else if (type == "box") {
                float3 size{1,1,1};
                if (obj["size"]) size = resolve_vec3(obj["size"], config);
                builder.add(Box(center.x, center.y, center.z, size.x, size.y, size.z), mat);
            } else if (type == "triangle") {
                float3 v0{0,0,0}, v1{1,0,0}, v2{0,1,0};
                if (obj["v0"]) v0 = resolve_vec3(obj["v0"], config);
                if (obj["v1"]) v1 = resolve_vec3(obj["v1"], config);
                if (obj["v2"]) v2 = resolve_vec3(obj["v2"], config);
                builder.add(Triangle(v0, v1, v2), mat);
            }
        }
    }

    // ── Lights (emissive) ─────────────────────────────────────────
    auto lights = root["lights"];
    if (lights && lights.IsSequence()) {
        for (size_t i = 0; i < lights.size(); i++) {
            auto light = lights[i];
            float3 center = resolve_vec3(light["center"], config);
            float3 color = resolve_vec3(light["color"], config);
            float intensity = light["intensity"] ? resolve_float(light["intensity"], config) : 1.0f;
            float radius = light["radius"] ? resolve_float(light["radius"], config) : 0.1f;

            auto mat = diffuse_light(scale(float3(color.x, color.y, color.z), intensity));
            builder.add(Sphere(center, radius), mat);
        }
    }

    spdlog::info("[scene] built from '{}': {} objects", config.yaml_path, builder.debug_aabb_count());
}

} // namespace scene_loader
