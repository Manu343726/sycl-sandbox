#pragma once
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/scene/data.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/param_types.h>
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <array>

/// Host-side scene system: typed parameter descriptors with documentation,
/// procedural data sources, declarative YAML scene geometry, and procedural
/// UI support.
///
/// Key concepts:
/// - Scene YAML is the SOLE source of truth for ALL parameters including
///   camera parameters (auto-generated from `scene_type: 2d|3d`)
/// - Every parameter has a name, type, description (documentation), default
///   value, and optional range
/// - `SceneDescriptor` is the host-side representation used for UI generation
///   and buffer layout
/// - `rt::ParamLookup` is the device-friendly table for kernel parameter reads
/// - Kernels read parameters by name via `reader.read<T>("name")` — the
///   underlying buffer layout is an implementation detail invisible to kernels
///
/// Usage:
/// @code
///   // Load scene
///   auto scene_desc = scene_loader::load_scene_descriptor("scenes/my.yaml");
///
///   // Host-side buffer management
///   scene_desc.build_layout();
///   float* buffer = scene_desc.default_buffer.data();
///
///   // Pass to kernel
///   kernel_set_param_lookup(scene_desc.lookup_entries, ...);
///   kernel_init(queue, w, h, buffer, scene_desc.buffer_size());
///
///   // UI (procedural from descriptors)
///   render_procedural_controls(scene_desc.params, buffer);
/// @endcode
namespace scene_loader {

// ── Value type enum ───────────────────────────────────────────────────

enum class ValueType : uint8_t {
    Int,
    Float,
    Vec3,
    Color,
    String,
    IntArray,
    FloatArray,
    Vec3Array,
    StringArray,
};

// ── Value ──────────────────────────────────────────────────────────────

/// A resolved value of any supported type.  Scalar and array variants
/// are stored in tagged unions (only one is active at a time).
struct Value {
    ValueType type = ValueType::Float;

    // Scalars
    int         int_val = 0;
    float       float_val = 0.0f;
    rt::float3  vec3_val = {0, 0, 0};
    std::string string_val;

    // Arrays
    std::vector<int>          int_arr;
    std::vector<float>        float_arr;
    std::vector<rt::float3>   vec3_arr;
    std::vector<std::string>  string_arr;

    size_t array_length() const;
};

// ── ResolvedScene ──────────────────────────────────────────────────────

/// Fully resolved scene configuration after evaluating all parameters
/// and data sources from a YAML file.  All `$name` references have been
/// resolved to concrete values.  Used by build_scene() for geometry.
struct ResolvedScene {
    std::string name;          ///< Human-readable scene name
    std::string kernel;        ///< Kernel shared library name
    std::string yaml_path;     ///< Source YAML file path
    std::string scene_type;    ///< "2d", "3d", or ""

    /// Parameter values keyed by name (scalar or small array values).
    std::map<std::string, Value> params;

    /// Data source arrays keyed by id.
    std::map<std::string, Value> data_sources;

    /// Raw YAML objects section — kept for build_scene() to parse.
    std::string objects_yaml;
};

// ── ParamDescriptor ────────────────────────────────────────────────────

/// Categories used to group parameters in the procedural UI.
enum class ParamCategory : uint8_t {
    Render,     ///< spp_frame, max_bounces, etc.
    Camera3D,   ///< cam_eye, cam_at, cam_fov, cam_aperture, cam_up
    Camera2D,   ///< center_x, center_y, zoom
    Kernel,     ///< Kernel-specific (scene-defined) parameters
};

/// Fully describes a single parameter: its metadata (name, type, doc),
/// default value, UI hint (range), and buffer position.
///
/// The YAML scene file is the source of truth.  Camera parameters are
/// auto-generated based on `scene_type` and NOT declared in YAML.
struct ParamDescriptor {
    std::string name;
    std::string description;   ///< Documentation string (shown as tooltip)
    ParamType type;            ///< Maps to ImGui widget type
    ParamCategory category = ParamCategory::Kernel;

    // ── Default value (union, which member depends on type) ──────────
    float       default_f    = 0.0f;
    int32_t     default_i    = 0;
    bool        default_b    = false;
    std::array<float, 3> default_c3 = {0, 0, 0};
    std::array<float, 4> default_c4 = {0, 0, 0, 0};

    // ── Optional range (for sliders) ─────────────────────────────────
    bool has_range = false;
    float range_min_f = 0.0f, range_max_f = 0.0f;
    int32_t range_min_i = 0, range_max_i = 0;

    // ── Buffer layout (filled by SceneDescriptor::build_layout()) ────
    uint32_t buffer_offset = 0; ///< byte offset into flat float buffer
    uint32_t buffer_size   = 4; ///< size in bytes (4 for int/float, 12 for vec3, etc.)

    // ── Camera integration tags ─────────────────────────────────────
    bool is_camera_eye     = false;
    bool is_camera_target  = false;
    bool is_camera_up      = false;
    bool is_camera_fov     = false;
    bool is_camera_aperture = false;
    bool is_camera_2d_center = false;
    bool is_camera_2d_zoom   = false;
};

// ── StatDescriptor ────────────────────────────────────────────────────

/// How a statistic should be visualized in the UI.
enum class VisualizationHint : uint8_t {
    Auto,       ///< Choose based on type (int/float → text, with history → graph)
    Text,       ///< Plain text value display
    Gauge,      ///< 0–1 gauge / progress bar
    Graph,      ///< Time-series line graph (requires history)
    Counter,    ///< Monotonically increasing counter (rate graph)
    Flag,       ///< Boolean status indicator (green/red)
};

/// Fully describes a single kernel-output statistic: its metadata, type,
/// documentation, visualization hint, and buffer position.
///
/// Statistics are the output counterpart of parameters.  Like parameters,
/// they have typed, named values identified by a hash-based lookup.
/// Unlike parameters, statistics are WRITTEN by the kernel and READ by
/// the host for display.  Standard statistics (FPS, frame_time) are
/// auto-generated; kernel-specific ones are declared in the YAML scene
/// file under a `statistics:` section.
struct StatDescriptor {
    std::string name;
    std::string description;   ///< Documentation string (shown as tooltip)
    ParamType type;            ///< FLOAT, INT, COLOR_RGB/VEC3, BOOL
    VisualizationHint viz = VisualizationHint::Auto;
    ParamCategory category = ParamCategory::Render; ///< grouping

    // ── Range (for gauges) ───────────────────────────────────────────
    bool has_range = false;
    float range_min_f = 0.0f, range_max_f = 0.0f;

    // ── Buffer layout (filled by SceneDescriptor::build_layout()) ────
    uint32_t buffer_offset = 0; ///< byte offset into flat float buffer
    uint32_t buffer_size   = 4; ///< size in bytes
};

// ── ParamRef — type-safe parameter accessor ───────────────────────────
//
// Wraps a ParamDescriptor (type metadata, documentation, range) together
// with a mutable pointer into the live parameter buffer.  All read/write
// operations go through the descriptor's declared type, ensuring that
// the UI and host code never accidentally reinterpret bits.
//
// Usage:
// @code
//   ParamRef ref = scene_desc.find_param_ref("max_bounces");
//   if (ref.valid()) {
//       int bounces = ref.as_int();       // type-checked read
//       ref.set(20);                      // type-checked write
//   }
// @endcode
struct ParamRef {
    const ParamDescriptor *descriptor = nullptr;
    float *value = nullptr;

    bool valid() const { return descriptor && value; }
    ParamType type() const { return descriptor ? descriptor->type : ParamType::FLOAT; }
    const std::string &name() const { static const std::string empty; return descriptor ? descriptor->name : empty; }

    // ── Type-safe reads ───────────────────────────────────────────
    float as_float() const {
        return valid() ? *value : 0.0f;
    }
    int as_int() const {
        return valid() ? static_cast<int>(*value) : 0;
    }
    bool as_bool() const {
        return valid() ? (*value != 0.0f) : false;
    }
    void as_vec3(float out[3]) const {
        if (valid()) { out[0] = value[0]; out[1] = value[1]; out[2] = value[2]; }
        else { out[0] = out[1] = out[2] = 0.0f; }
    }

    // ── Type-safe writes ──────────────────────────────────────────
    void set(float v) { if (valid()) *value = v; }
    void set(int v) { if (valid()) *value = static_cast<float>(v); }
    void set(bool v) { if (valid()) *value = v ? 1.0f : 0.0f; }
    void set(float r, float g, float b) {
        if (valid()) { value[0] = r; value[1] = g; value[2] = b; }
    }

    /// Mutable access to the underlying float pointer for ImGui widgets.
    float *ptr() { return value; }
    const float *ptr() const { return value; }

    /// Byte offset from the start of the buffer (for render-thread d_params mapping).
    uint32_t offset() const {
        return descriptor ? descriptor->buffer_offset : 0;
    }
};

// ── StatRef — type-safe statistic accessor ────────────────────────────
//
// Wraps a StatDescriptor (type metadata, visualization hint) together
// with a pointer into the live statistic buffer.  The kernel writes
// statistics through StatWriter; the host reads them through StatRef
// for display.
//
// Usage:
// @code
//   StatRef ref = scene_desc.find_stat_ref("num_objects");
//   if (ref.valid()) {
//       int count = ref.as_int();   // type-checked read
//   }
// @endcode
struct StatRef {
    const StatDescriptor *descriptor = nullptr;
    float *value = nullptr;

    bool valid() const { return descriptor && value; }
    ParamType type() const { return descriptor ? descriptor->type : ParamType::FLOAT; }
    const std::string &name() const { static const std::string empty; return descriptor ? descriptor->name : empty; }

    // ── Type-safe reads ───────────────────────────────────────────
    float as_float() const { return valid() ? *value : 0.0f; }
    int as_int() const { return valid() ? static_cast<int>(*value) : 0; }
    bool as_bool() const { return valid() ? (*value != 0.0f) : false; }
    void as_vec3(float out[3]) const {
        if (valid()) { out[0] = value[0]; out[1] = value[1]; out[2] = value[2]; }
        else { out[0] = out[1] = out[2] = 0.0f; }
    }

    // ── Type-safe writes (for host-computed stats like FPS) ───────
    void set(float v) { if (valid()) *value = v; }
    void set(int v) { if (valid()) *value = static_cast<float>(v); }
    void set(const float v[3]) {
        if (valid()) { value[0] = v[0]; value[1] = v[1]; value[2] = v[2]; }
    }

    const float *ptr() const { return value; }

    /// Byte offset from the start of the buffer.
    uint32_t offset() const {
        return descriptor ? descriptor->buffer_offset : 0;
    }
};

// ── SceneDescriptor ────────────────────────────────────────────────────

/// Complete host-side description of a scene: its parameters, defaults,
/// buffer layout, and lookup table for kernel parameter reads.
///
/// Built from a YAML scene file.  Camera parameters are auto-generated
/// from the `scene_type` field (2D or 3D).
struct SceneDescriptor {
    std::string name;
    std::string kernel_name;
    std::string yaml_path;

    /// "2d" or "3d" — controls auto-generation of camera params.
    std::string scene_type = "3d";

    /// All parameters (YAML-declared + auto-generated camera params).
    std::vector<ParamDescriptor> params;

    /// All statistics (auto-generated standard stats + YAML-declared kernel stats).
    std::vector<StatDescriptor> stats;

    // ── Param buffer state (filled by build_layout()) ────────────────
    std::vector<float> default_buffer;
    std::vector<rt::ParamEntry> lookup_entries;
    std::vector<float> current_buffer;  ///< live copy for UI edits

    // ── Stat buffer state (filled by build_layout()) ─────────────────
    std::vector<float> default_stat_buffer;    ///< initial/zero values
    std::vector<rt::StatEntry> stat_lookup_entries;
    std::vector<float> current_stat_buffer;   ///< live snapshot for UI display

    /// Compute buffer layout: assign offsets for params AND stats,
    /// build name-hash lookup tables, allocate buffer copies.
    void build_layout();

    /// Total param buffer size in bytes.
    size_t buffer_size() const {
        return default_buffer.size() * sizeof(float);
    }

    /// Total stat buffer size in bytes.
    size_t stat_buffer_size() const {
        return default_stat_buffer.size() * sizeof(float);
    }

    /// Find a parameter descriptor by name, or nullptr.
    const ParamDescriptor *find(const std::string &name) const;
    ParamDescriptor *find(const std::string &name);

    /// Find a stat descriptor by name, or nullptr.
    const StatDescriptor *find_stat(const std::string &name) const;
    StatDescriptor *find_stat(const std::string &name);

    /// Type-safe parameter accessor.  Returns a ParamRef that pairs the
    /// descriptor (type metadata, documentation, range) with a mutable
    /// pointer into the live buffer.  Returns an invalid ref if not found.
    ParamRef find_param_ref(const std::string &name);
    ParamRef find_param_ref(const std::string &name) const;

    /// Type-safe statistic accessor.  Returns a StatRef that pairs the
    /// descriptor (type metadata, documentation, viz hint) with a
    /// pointer into the live stat buffer.  Returns an invalid ref if not found.
    StatRef find_stat_ref(const std::string &name);
    StatRef find_stat_ref(const std::string &name) const;

    /// Write all param defaults into the current buffer.
    void reset_current_to_defaults();

    /// Copy current_buffer values into the lookup entries table
    /// (for kernel use — updates the hash→offset mapping).
    void sync_buffer_to_lookup();

    /// Return a ParamLookup pointing at current_buffer + entries.
    rt::ParamLookup make_lookup() const;

    /// Return a StatWriter pointing at current_stat_buffer + stat entries.
};

// ── SceneType helpers ─────────────────────────────────────────────────

/// Returns the list of camera parameter names for a 2D scene.
std::vector<ParamDescriptor> auto_camera_params_2d();

/// Returns the list of camera parameter names for a 3D scene.
std::vector<ParamDescriptor> auto_camera_params_3d();

/// Returns the list of standard auto-generated statistics (FPS, frame_time, etc.).
std::vector<StatDescriptor> auto_standard_stats();

// ── Parse helpers for scene_loader.cpp ────────────────────────────────

/// Parse a ParamType from a YAML type string.
ParamType parse_param_type(const std::string &type_str);

// ── API ────────────────────────────────────────────────────────────────

/// Load a YAML scene file and build a SceneDescriptor with all params
/// (including auto-generated camera params based on scene_type).
///
/// This does NOT resolve data sources or build geometry — use
/// load_and_resolve() + build_scene() for that.
///
/// @param yaml_path  Path to YAML scene file.
SceneDescriptor load_scene_descriptor(const std::string &yaml_path);

/// Load a YAML scene file and resolve all parameters and data sources.
/// (Legacy — used for geometry building).
///
/// Resolution steps:
///   1. Read all params (infer type from value if not declared).
///   2. Read all data_source definitions.
///   3. For each data source, resolve its inputs (recursively resolving
///      `$name` references to params or earlier data sources).
///   4. Evaluate each data source generator with resolved inputs.
///
/// @param yaml_path  Path to the YAML scene file.
/// @return Fully resolved configuration ready for build_scene().
ResolvedScene load_and_resolve(const std::string &yaml_path);

/// Build scene geometry from a resolved configuration.
///
/// Iterates the `objects` section of the YAML and populates the
/// SceneBuilder with hittable-material pairs.
///
/// @param builder  SceneBuilder to populate.
/// @param config   Resolved scene configuration from load_and_resolve().
void build_scene(rt::SceneBuilder &builder, const ResolvedScene &config);

} // namespace scene_loader
