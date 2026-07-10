# Coding Guidelines

## Naming

- **No prefixes.**  Never use `m_`, `g_`, `s_`, `k` prefixes for members,
  globals, statics, or constants.  Use descriptive names and let scope
  speak for itself.

  ```cpp
  // Bad
  int g_num_objects;
  float m_speed;
  const int kMaxValue = 42;

  // Good
  int num_objects;
  float speed;
  const int max_value = 42;  // or just a plain enum
  ```

- **No abbreviations.**  Every name should be a properly spelled word or
  a well-known domain term.  `rnd` → `random`, `n` → `count`/`num_*`,
  `o` → `objects`, `c` → `count`, `p` → `params`, `gc` → `ground_color`.

  ```cpp
  // Bad
  float rnd();
  auto* o = new Object[n];
  float3 gc;

  // Good
  float random_float();
  auto* objects = new Object[count];
  float3 ground_color;
  ```

- **Use namespaces instead of prefixes.**
  `rt::hittables::sphere()` instead of `rt_make_sphere()`.
  `rt::materials::lambertian()` instead of `rt_lambertian_create()`.

  ```cpp
  namespace rt::hittables  { Sphere sphere(center, radius); }
  namespace rt::materials  { Lambertian lambertian(albedo); }
  ```

## Enums

- **Plain `enum` when the values are used as indices or flags** and need
  implicit conversion to `int`.  This applies to parameter index enums.

  ```cpp
  enum rt_std_param {
      RT_SPP_FRAME = 0,
      RT_MAX_BOUNCES = 1,
      // …
  };
  // Used as:  params[RT_SPP_FRAME]   ← implicit int conversion
  ```

- **`enum class` when the values represent a choice** that should not
  implicitly convert.  This prevents passing a random integer where an
  axis is expected.

  ```cpp
  enum class Axis : int { X = 0, Y = 1, Z = 2 };
  // Used as:  add_quad(…, Axis::Y, …)
  ```

## Function signatures

- **Return `std::optional<T>`** instead of taking an output reference
  and returning `bool`.  This makes the API self-documenting — the return
  type says "may or may not produce a result".

  ```cpp
  // Bad
  bool hit(const Ray&, float, float, HitRecord&) const;

  // Good
  std::optional<HitRecord> hit(const Ray&, float, float) const;
  ```

- **Bundle multiple output values into a struct.**
  `ScatterRecord` packs `attenuation` + `scattered` ray.

## Documentation

- Every public function, class, and enum must have a Doxygen-style
  `///` comment explaining what it does, its parameters, and return
  value.  Include a brief usage example where helpful.

  ```cpp
  /// Creates a Lambertian material with the given albedo.
  /// @param albedo  Surface colour (reflectance).
  inline Lambertian lambertian(float3 albedo);
  ```

- Comments should explain *why*, not *what* (the code is the *what*).

## Code organisation

- **One class per file**, named after the class (lowercase, snake_case).
  Files go in subdirectories matching the namespace.

- **Free factory functions** instead of static methods.
  `sphere(c, r)` returns a `Sphere`, not `Sphere::create(c, r)`.

- **Include what you use.**  Each header should be self-contained.
  Prefer forward declarations over includes where possible.

## Namespaces

```
rt                          Top-level namespace for everything.
rt::hittables               Geometry primitive types and factories.
rt::materials               Material types and factories.
```

Kernel code uses `using namespace rt;` and pulls in individual factories
as needed:

```cpp
using namespace rt;
using rt::hittables::sphere;
using rt::materials::lambertian;
```
