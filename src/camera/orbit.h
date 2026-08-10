#pragma once
#include <cmath>

// ── OrbitCam ──────────────────────────────────────────────────────────
/// Spherical coordinate orbit camera (for 3D scenes).
///
/// The camera position (eye) is fixed; yaw/pitch rotate the look-at point
/// around it.  This is a first-person orbit — the camera stays put and
/// looks around.  Pan moves both eye and look-at together; zoom changes
/// the focal distance.
///
/// Convention:
///   theta = yaw   (azimuth, horizontal angle around the Y axis)
///   phi   = pitch (elevation from the XZ plane)
///   dist  = focal distance from eye to look-at target
///   roll  = rotation of the up vector around the forward axis
struct OrbitCam {
    float theta = 1.35f;
    float phi = 1.05f;
    float dist = 12.f;
    float roll = 0.f;
    float eye[3] = {0.f, 0.f, 0.f};
};

/// Compute the look-at target from the fixed eye position and spherical
/// look direction.
///
///   lookat = eye + dist · (cos(phi)·sin(theta),  sin(phi),  cos(phi)·cos(theta))
///
/// The direction vector points from eye towards the target.
inline void orbit_to_lookat(const OrbitCam &orbit, float target[3]) {
    float cos_theta = cosf(orbit.theta), sin_theta = sinf(orbit.theta);
    float cos_phi = cosf(orbit.phi), sin_phi = sinf(orbit.phi);
    target[0] = orbit.eye[0] + orbit.dist * cos_phi * sin_theta;
    target[1] = orbit.eye[1] + orbit.dist * sin_phi;
    target[2] = orbit.eye[2] + orbit.dist * cos_phi * cos_theta;
}

/// Rotate the default up vector {0,1,0} around the forward (look) axis
/// by the camera's roll angle, using Rodrigues' rotation formula.
inline void orbit_up(const OrbitCam &orbit, float up[3]) {
    float cos_theta = cosf(orbit.theta), sin_theta = sinf(orbit.theta);
    float cos_phi = cosf(orbit.phi), sin_phi = sinf(orbit.phi);
    float forward[3] = {cos_phi * sin_theta, sin_phi, cos_phi * cos_theta};
    float default_up[3] = {0.f, 1.f, 0.f};
    float cos_roll = cosf(orbit.roll), sin_roll = sinf(orbit.roll);
    float dot_product =
        forward[0] * default_up[0] + forward[1] * default_up[1] + forward[2] * default_up[2];
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

/// Initialize an OrbitCam from a known eye position and look-at target.
inline void orbit_from_eye(OrbitCam &orbit,
                           const float eye[3],
                           const float target[3]) {
    float dx = target[0] - eye[0],
          dy = target[1] - eye[1],
          dz = target[2] - eye[2];
    orbit.dist = sqrtf(dx * dx + dy * dy + dz * dz);
    orbit.theta = atan2f(dx, dz);
    orbit.phi = asinf(dy / orbit.dist);
    orbit.eye[0] = eye[0];
    orbit.eye[1] = eye[1];
    orbit.eye[2] = eye[2];
    orbit.theta = std::fmod(orbit.theta, 2.f * 3.14159265f);
    orbit.phi = std::max(-1.5f, std::min(1.5f, orbit.phi));
}
