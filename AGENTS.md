# AI context for this repository

Before making changes, read the following files in order:

1. **README.md**: Project overview, build instructions, and usage.
2. **docs/architecture.md**: Architecture — new features always follow this.
3. **docs/raytracing.md**: Raytracing lib, variant dispatch, scene building, primitives.
4. **docs/coding-guidelines.md**: Coding style, workflow rules.

## Key API points

- `scene.h` provides only `Axis` enum, `quad_corner()`, and `add(objects, count, object)`.
- No `add_quad`/`add_box` helpers anywhere. Use factories directly:
  - `add(objects, count, {hittables::quad(axis, value, min_s, max_s, min_t, max_t), material})`
  - `add(objects, count, {hittables::box(cx, cy, cz, sx, sy, sz), material})`
- `Box` is a real hittable (composed of 6 `Quad` faces internally). Primitives can compose other primitives.
- `hittables::quad(axis, ...)` takes axis as int `0/1/2` (X/Y/Z), not `Axis` enum.

After finishing implementation, go back and re-read the docs to verify guidelines are followed.