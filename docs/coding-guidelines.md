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

## Keeping docs in sync

Architecture changes must update the corresponding docs in the same commit:
- **`docs/architecture.md`** — host, build system, kernel API.
- **`docs/raytracing.md`** — raytracing library internals.
- **`README.md`** — build instructions, usage, feature list.

Out-of-date docs are worse than no docs.

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

## Formatting

- 4-space indentation.
- Opening brace on the same line for functions, control flow, and classes.
- Line length cap at 100 columns (prefer breaks at natural points).
- Spaces inside braces for initializer lists: `{1, 2, 3}` not `{1,2,3}`.
