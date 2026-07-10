#pragma once
#include "math.h"
#include "variant.h"
#include <variant>

namespace rt {

struct Ray { float3 orig, dir; };

struct HitRecord {
    float3   p;
    float3   normal;
    float    t;
    bool     front_face;
};

// ── Geometry classes ───────────────────────────────────────────────────
class Sphere {
public:
    float3 center;
    float  radius;
    Sphere() = default;
    Sphere(float3 c, float r) : center(c), radius(r) {}
    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
};

class Quad {
public:
    float3 a, b, c, normal;
    Quad() = default;
    Quad(float3 a_, float3 b_, float3 c_) : a(a_), b(b_), c(c_) {
        float3 ab = sub(b, a), ac = sub(c, a);
        normal = norm(cross(ab, ac));
    }
    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
};

using Hittable = std::variant<Sphere, Quad>;

// ── Material classes ───────────────────────────────────────────────────
class Lambertian {
public:
    float3 albedo;
    Lambertian() = default;
    explicit Lambertian(float3 a) : albedo(a) {}
    bool scatter(const Ray& in, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const;
    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

class Metal {
public:
    float3 albedo;
    float  fuzz;
    Metal() = default;
    Metal(float3 a, float f) : albedo(a), fuzz(f) {}
    bool scatter(const Ray& in, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const;
    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

class Dielectric {
public:
    float ir;
    Dielectric() = default;
    explicit Dielectric(float i) : ir(i) {}
    bool scatter(const Ray& in, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const;
    float3 emit(const HitRecord&) const { return {0,0,0}; }
};

class DiffuseLight {
public:
    float3 emit_color;
    DiffuseLight() = default;
    explicit DiffuseLight(float3 e) : emit_color(e) {}
    bool scatter(const Ray&, const HitRecord&, float3&, Ray&, RNG&) const { return false; }
    float3 emit(const HitRecord&) const { return emit_color; }
};

using Material = std::variant<Lambertian, Metal, Dielectric, DiffuseLight>;

// ── Object ─────────────────────────────────────────────────────────────
class Object {
public:
    Hittable hittable;
    Material material;

    Object() = default;
    Object(Hittable h, Material m) : hittable(std::move(h)), material(std::move(m)) {}

    static Object make_sphere(float3 center, float radius, Material mat) {
        return Object(Sphere(center, radius), std::move(mat));
    }
    static Object make_quad(float3 a, float3 b, float3 c, Material mat) {
        return Object(Quad(a, b, c), std::move(mat));
    }

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;
    bool scatter(const Ray& in, const HitRecord& rec, float3& attenuation,
                 Ray& scattered, RNG& rng) const;
    float3 emit(const HitRecord& rec) const;
};

} // namespace rt
