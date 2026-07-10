# AI context for this repository

Before making changes, read the following files in order:

1. **README.md** — project overview, build instructions, and usage.
2. **docs/architecture.md** — how the raytracing system works (variant-based
   polymorphism, std::optional pattern, standard param layout, file structure).
3. **docs/coding-guidelines.md** — naming, enum rules, function signatures,
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
