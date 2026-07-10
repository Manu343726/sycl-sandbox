# Coding Guidelines

## Naming

### No prefixes — use namespaces

Never use `m_`, `g_`, `s_`, `k` prefixes for members, globals, statics,
or constants.  Group related names in namespaces instead.

```cpp
// Bad
int g_num_objects;
float m_speed;
static float s_cache;

// Good
namespace {
    int num_objects;
}
float speed;
```

### No prefixes for function or type names — use namespaces

Instead of prefixing functions/types with an abbreviation of their module,
put them in a namespace:

```cpp
// Bad
void rt_make_sphere(…);                 // prefix rt_
struct RNG_State;                        // prefix RNG_
float rt_math_length(…);

// Good
namespace rt::hittables { Sphere sphere(…); }
namespace rt { struct RNG { … }; }
namespace rt { float length(…); }        // rt::length()
```

### No abbreviations

Every name should be a properly spelled word or a well-known domain term.

```cpp
// Bad
float rnd();          // abbreviation
auto* o = new T[n];  // single-letter
float3 gc;           // abbreviation
void add_quad(…, int ax, float av, float b0, float b1, …);

// Good
float random_float();
auto* objects = new Object[count];
float3 ground_color;
void add_quad(…, Axis primary_axis, float axis_value, float min_second, float max_second, …);
```

Exception: loop counters (`i`, `k`) and well-known math symbols (`a`, `b`, `t`).

### Examples of good vs bad naming

| Bad               | Good                  | Reason              |
|-------------------|-----------------------|----------------------|
| `rnd()`           | `random_float()`      | abbreviation        |
| `g_objs`          | `scene_objects`       | `g_` prefix         |
| `g_n`             | `num_objects`         | `g_` prefix + abbrev|
| `c`               | `count`               | too short           |
| `p`               | `params`              | too short           |
| `lc, ls`          | `light_color`, `light_strength` | abbreviation |
| `ax, av, b0, b1`  | `primary_axis, axis_value, min_second, max_second` | abbreviation |
| `P_LC, P_LS`      | `PARAM_LIGHT_COLOR`, `PARAM_LIGHT_STRENGTH` | abbreviation |

## Namespaces

```
rt                          Top-level namespace for the raytracing lib.
rt::hittables               Geometry primitive types and factories.
rt::materials               Material types and factories.

No prefix is needed — just use the namespace.
```

Kernel code uses `using namespace rt;` and pulls in individual factories:

```cpp
using namespace rt;
using rt::hittables::sphere;
using rt::materials::lambertian;
```

## Enums

- **Plain `enum`** when values are used as array indices or bit flags
  and need implicit conversion to `int`.

  ```cpp
  enum rt_std_param { RT_SPP_FRAME = 0, RT_MAX_BOUNCES = 1, … };
  // Used as:  params[RT_SPP_FRAME]
  ```

- **`enum class`** when values represent a distinct choice that should
  not be used as an integer by accident.

  ```cpp
  enum class Axis : int { X = 0, Y = 1, Z = 2 };
  // Used as:  add_quad(…, Axis::Y, …)
  ```

## Function signatures

- **Return `std::optional<T>`** instead of taking an output reference
  and returning `bool`.

  ```cpp
  // Bad
  bool hit(const Ray&, float, float, HitRecord&) const;

  // Good
  std::optional<HitRecord> hit(const Ray&, float, float) const;
  ```

- **Bundle multiple output values into a struct.**
  `ScatterRecord` packs `attenuation` + `scattered` ray.

- **Free factory functions** instead of static methods.
  `sphere(c, r)` returns a `Sphere`, not `Sphere::create(c, r)`.

## CPU + GPU portability

All kernel code must compile and run correctly on both the `generic` (OpenMP
CPU) backend and the CUDA GPU backend.  This means:

- **No virtual functions, no function pointers, no `std::function`** in
  code that runs on the device (inside `parallel_for` or any function it
  calls).  These require vtables or indirect calls that CUDA doesn't support.
  Use `std::variant` + `visit_rt()` instead (compile-time dispatch).

- **No `dynamic_cast`, `typeid`, `std::any`** in device code.

- **No RTTI or exceptions** in device code.

- **No host-only pointers in device memory.**  Memory allocated with
  `sycl::malloc_device` is GPU-only; use `sycl::malloc_host` or
  `sycl::malloc_shared` when both sides need access.

- **No `std::vector`, `std::string`, or heap-allocating containers**
  inside kernel lambdas.  Prefer stack arrays or SYCL USM pointers.

- **No global/static constructors with side effects** in kernel shared
  libraries (`.so` loaded via `dlopen`).  The SYCL runtime must register
  device images during static init; don't interfere.

- **No platform-specific intrinsics or inline assembly.**  Stick to
  standard C++ and SYCL builtins (`sycl::sqrt`, `sycl::fabs`, etc.).

The `generic` backend runs kernel lambdas as regular C++ on the host
(via OpenMP), which is more permissive.  **Always build with the CUDA
target enabled** (or at least verify compilation) before committing,
because the CUDA backend is stricter and will catch portability issues
that `generic` silently accepts.

## AGENTS.md

`AGENTS.md` is the entry point for AI agents working on this repository.
It must only **reference** documentation files by path and order — never
write rules, summaries, or examples inline.  Rules live in
`docs/coding-guidelines.md`, architecture in `docs/architecture.md` and
`docs/raytracing.md`.  If a rule needs updating, update the doc file,
not `AGENTS.md`.

## Run clang-format

Always run clang-format after any code change, before committing:

```bash
clang-format -i src/*.cpp include/rt/**/*.h include/rt/hittables/*.h \
                include/rt/materials/*.h kernels/*/kernel.cpp
```

The CI equivalent `make format` can be set up as a phony target.  If
clang-format is not available, at minimum ensure `InsertBraces` and
`AllowShortIfStatementsOnASingleLine` are respected by hand.

## Keeping docs in sync

Architecture changes must update the corresponding docs in the same commit:
- **`docs/architecture.md`** — host, build system, kernel API.
- **`docs/raytracing.md`** — raytracing library internals.
- **`README.md`** — build instructions, usage, feature list.

Out-of-date docs are worse than no docs.

## Math code layout

Math-heavy functions must be broken into logical sections separated by
blank lines, each preceded by a `///` or `//` comment explaining the step.

```cpp
std::optional<HitRecord> hit(const Ray& ray, float t_min, float t_max) const {
    // Compute the quadratic coefficients using the geometric formulation
    float3 oc = sub(ray.orig, center);
    float a = dot(ray.dir, ray.dir);
    float half_b = dot(oc, ray.dir);
    float c_ = dot(oc, oc) - radius * radius;

    // Solve for t using the reduced quadratic formula
    float discriminant = half_b*half_b - a*c_;
    if (discriminant <= 0) {
        return std::nullopt;
    }
    float sqrt_d = sycl::sqrt(discriminant);
    float t = (-half_b - sqrt_d) / a;
    if (t < t_min || t > t_max) {
        t = (-half_b + sqrt_d) / a;
    }
    if (t < t_min || t > t_max) {
        return std::nullopt;
    }

    // Fill the HitRecord with the intersection point and surface normal
    HitRecord rec;
    rec.t = t;
    rec.p = add(ray.orig, scale(ray.dir, t));
    rec.normal = scale(sub(rec.p, center), 1.f / radius);
    rec.front_face = dot(ray.dir, rec.normal) < 0;
    if (!rec.front_face) rec.normal = scale(rec.normal, -1);
    return rec;
}
```

Each section should be one logical step — you should be able to read
the comments alone and understand the algorithm.

## Code organisation

- One class per file, named after the class (lowercase_snake_case).
- Files go in subdirectories matching the namespace.
- Each header must be self-contained (include what it uses).
- Prefer forward declarations over includes.

## Comments

- Public functions, classes, and enums must have `///` Doxygen-style
  documentation: what it does, parameters, return value, example usage.
- Comments explain *why*, not *what* (the code is the *what*).

```cpp
/// Creates a Lambertian material with the given albedo.
/// @param albedo  Surface colour (reflectance).
inline Lambertian lambertian(float3 albedo);
```

- Use `// ── section markers ──` sparingly for major sections in long files.

## Clang-format

A `.clang-format` file is checked into the repository root.  Run it
before committing:

```bash
clang-format -i src/*.cpp src/*.h include/**/*.h include/**/*.hpp kernels/*/kernel.cpp kernels/*/kernel.h
```

Key style rules enforced by `.clang-format`:
- Braces on the same line (Attach style).
- 4-space indent.
- 100-column line limit.
- Spaces inside `if (...)`, `for (...)`, `while (...)`.
- **`if`, `else`, `for`, `while` without braces are forbidden.**
- No tabs, spaces only.

If clang-format is not available, at minimum ensure no brace-less
control flow statements.

## Formatting

- `clang-format` with the provided `.clang-format` file.
- 4-space indentation, no tabs.
- Opening brace on the same line for functions, control flow, and classes.
- **`if`, `else`, `for`, `while` must always use braces** — no single-line
  unbraced statements.  This is enforced by `.clang-format`
  (`AllowShortIfStatementsOnASingleLine: Never`).
- Line length cap at 100 columns (prefer breaks at natural points).
- Spaces inside braces for initializer lists: `{1, 2, 3}` not `{1,2,3}`.
