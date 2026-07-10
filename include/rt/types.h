#pragma once
#include "math.h"

namespace rt {

struct Ray {
    float3 orig;
    float3 dir;
};

struct HitRecord {
    float3   p;
    float3   normal;
    float    t;
    bool     front_face;
    class Material const* material;  // material of the hit object
};

// ── Material (reflection / refraction / emission) ─────────────────────
class Material {
public:
    virtual bool scatter(const Ray& in, const HitRecord& rec,
                          float3& attenuation, Ray& scattered, RNG& rng) const = 0;
    virtual float3 emit(const HitRecord&) const { return {0,0,0}; }
    virtual ~Material() = default;
};

class Lambertian final : public Material {
public:
    float3 albedo;
    explicit Lambertian(float3 a) : albedo(a) {}
    bool scatter(const Ray&, const HitRecord&, float3& attenuation, Ray& scattered, RNG& rng) const override;
};

class Metal final : public Material {
public:
    float3 albedo;
    float  fuzz;
    Metal(float3 a, float f) : albedo(a), fuzz(f) {}
    bool scatter(const Ray&, const HitRecord&, float3& attenuation, Ray& scattered, RNG& rng) const override;
};

class Dielectric final : public Material {
public:
    float ir;
    explicit Dielectric(float index) : ir(index) {}
    bool scatter(const Ray&, const HitRecord&, float3& attenuation, Ray& scattered, RNG& rng) const override;
};

class DiffuseLight final : public Material {
public:
    float3 emit_color;
    explicit DiffuseLight(float3 e) : emit_color(e) {}
    float3 emit(const HitRecord&) const override { return emit_color; }
    bool scatter(const Ray&, const HitRecord&, float3&, Ray&, RNG&) const override { return false; }
};

// ── Hittable (geometry) ───────────────────────────────────────────────
class Hittable {
public:
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
    virtual ~Hittable() = default;
};

class Sphere final : public Hittable {
public:
    float3 center;
    float radius;
    Sphere(float3 c, float r) : center(c), radius(r) {}
    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
};

class Quad final : public Hittable {
public:
    float3 a, b, c;
    float3 normal;
    Quad(float3 a_, float3 b_, float3 c_) : a(a_), b(b_), c(c_) {
        float3 ab = sub(b, a), ac = sub(c, a);
        normal = norm(cross(ab, ac));
    }
    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
};

// ── Object (geometry + material pair) ──────────────────────────────────
struct Object {
    Hittable const* geometry;
    Material const* material;
};

} // namespace rt
