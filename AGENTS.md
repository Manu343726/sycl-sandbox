# AI context for this repository

Before making changes, read the following files in order:

1. **README.md** — project overview, build instructions, and usage.
2. **docs/architecture.md** — host architecture, kernel loading, build system.
3. **docs/raytracing.md** — raytracing library: variant OOP, std::optional,
   standard params, kernel API, scene helpers.
4. **docs/coding-guidelines.md** — naming, enum rules, function signatures,
   documentation style, namespace conventions.

## Key rules to follow

- No prefixes (`m_`, `g_`, `k`, `s_`).  Use namespaces instead.
- No abbreviations in names.  Spell everything out.
- Use `std::optional<T>` return instead of `bool` + output reference.
- Use plain `enum` for parameter indices (implicit int conversion).
- Use `enum class` for choices like `Axis`.
- Each class in its own file under the matching subdirectory.
- Free factory functions, not static methods.
- Document all public APIs with `///` comments.
- **CPU + GPU portability:** no virtual functions, function pointers,
  RTTI, exceptions, or heap-allocating containers in device code.
  Use `std::variant` + `visit_rt()` for polymorphism.
  Verify against the CUDA backend before committing.
