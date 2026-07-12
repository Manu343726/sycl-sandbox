#pragma once
#include <cmath>

// ── OrbitCam ──────────────────────────────────────────────────────────
/// Spherical coordinate orbit camera (for 3D scenes).
///
/// Convention:
///   theta = azimuth (horizontal angle around the Y axis)
///   phi   = elevation (vertical angle from the XZ plane)
///   dist  = distance from the camera to the look-at target
///   roll  = rotation of the up vector around the forward (lookat) axis
struct OrbitCam {
    float theta = 1.35f;
    float phi = 1.05f;
    float dist = 12.f;
    float roll = 0.f;
    float target[3] = {0.f, 0.f, 0.f};
};

/// Convert spherical orbit coordinates to a Cartesian eye position.
///
///   eye = target + dist · (cos(phi)·sin(theta),  sin(phi),  cos(phi)·cos(theta))
///
/// This places the camera on a sphere of radius `dist` centred on the target,
/// with `theta` as azimuth and `phi` as elevation.
inline void orbit_to_eye(const OrbitCam &orbit, float eye[3]) {
    float cos_theta = cosf(orbit.theta), sin_theta = sinf(orbit.theta);
    float cos_phi = cosf(orbit.phi), sin_phi = sinf(orbit.phi);
    eye[0] = orbit.target[0] + orbit.dist * cos_phi * sin_theta;
    eye[1] = orbit.target[1] + orbit.dist * sin_phi;
    eye[2] = orbit.target[2] + orbit.dist * cos_phi * cos_theta;
}

/// Rotate the default up vector {0,1,0} around the forward (lookat) axis
/// by the camera's roll angle, using Rodrigues' rotation formula.
///
///   up_rotated = up · cos(roll) + (forward × up) · sin(roll)
///                + forward · (forward · up) · (1 − cos(roll))
///
/// The forward direction points from the camera toward the target
/// (in the orbit frame: forward = -lookat_direction_in_camera_frame).
inline void orbit_up(const OrbitCam &orbit, float up[3]) {
    float cos_theta = cosf(orbit.theta), sin_theta = sinf(orbit.theta);
    float cos_phi = cosf(orbit.phi), sin_phi = sinf(orbit.phi);
    float forward[3] = {-cos_phi * sin_theta, -sin_phi, -cos_phi * cos_theta};
    float default_up[3] = {0.f, 1.f, 0.f};
    float cos_roll = cosf(orbit.roll), sin_roll = sinf(orbit.roll);
    float dot_product =
        forward[0] * default_up[0] + forward[1] * default_up[1] + forward[2] * default_up[2];
    // Rodrigues' rotation:  up_rotated = up * cos(r) + (forward × up) * sin(r) + forward *
    // (forward·up) * (1-cos(r))
    up[0] = default_up[0] * cos_roll +
            (forward[1] * default_up[2] - forward[2] * default_up[1]) * sin_roll +
            forward[0] * dot_product * (1 - cos_roll);
    up[1] = default_up[1] * cos_roll +
            (forward[2] * default_up[0] - forward[0] * default_up[2]) * sin_roll +
            forward[1] * dot_product * (1 - cos_roll);
    up[2] = default_up[2] * cos_roll +
            (forward[0] * default_up[1] - forward[1] * default_up[0]) * sin_roll +
            forward[2] * dot_product * (1 - cos_roll);
}

/// Initialize an OrbitCam from a known camera position and target.
inline void orbit_from_lookat(OrbitCam &orbit,
                               const float eye[3],
                               const float target[3]) {
    float dx = eye[0] - target[0],
          dy = eye[1] - target[1],
          dz = eye[2] - target[2];
    orbit.dist = sqrtf(dx * dx + dy * dy + dz * dz);
    orbit.theta = atan2f(dx, dz);
    orbit.phi = asinf(dy / orbit.dist);
    orbit.target[0] = target[0];
    orbit.target[1] = target[1];
    orbit.target[2] = target[2];
    orbit.theta = std::fmod(orbit.theta, 2.f * 3.14159265f);
    orbit.phi = std::max(-1.5f, std::min(1.5f, orbit.phi));
}
