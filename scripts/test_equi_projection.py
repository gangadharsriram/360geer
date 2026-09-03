"""Standalone sanity check for the EQUI (equirectangular) Stage-A math added to
geer-rasterizer (submodules/geer-rasterizer/cuda_rasterizer/auxiliary.h).

Pure NumPy port of the CUDA device functions - no CUDA/torch needed - so this can
run anywhere to sanity-check the formulas before trusting the compiled kernel.
Checks:
  1. pixel -> lon/lat -> ray direction -> lon/lat -> pixel round-trips.
  2. the spherical-cap -> lon/lat box (lonlatBoxFromAngularCap) always contains
     the true angular cap, and saturates to full longitude width near the poles
     instead of blowing up (the whole point of not using a linearized Jacobian).

Run: python scripts/test_equi_projection.py
"""
import numpy as np

EQUI_PI = np.pi


def point_to_lonlat(p):
    p = np.asarray(p, dtype=np.float64)
    dist = np.linalg.norm(p, axis=-1) + 1e-9
    lon = np.arctan2(p[..., 0], p[..., 2])
    lat = np.arcsin(np.clip(p[..., 1] / dist, -1.0, 1.0))
    return lon, lat


def lonlat_to_pixel(lon, lat, W, H):
    px = (lon / EQUI_PI + 1.0) * (0.5 * W)
    py = (lat * (2.0 / EQUI_PI) + 1.0) * (0.5 * H)
    return px, py


def pixel_to_lonlat(px, py, W, H):
    lon = (2.0 * px / W - 1.0) * EQUI_PI
    lat = (2.0 * py / H - 1.0) * (0.5 * EQUI_PI)
    return lon, lat


def lonlat_to_dir(lon, lat):
    cl = np.cos(lat)
    return np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)


def lonlat_box_from_angular_cap(lon0, lat0, theta_max):
    """Mirrors auxiliary.h's lonlatBoxFromAngularCap exactly (see its comment for
    why the widest row is generally NOT row lat0, but phi* = asin(sin(lat0)/cos(theta_max))."""
    half_pi = 0.5 * EQUI_PI
    lat_min = np.maximum(-half_pi, lat0 - theta_max)
    lat_max = np.minimum(half_pi, lat0 + theta_max)

    sin_lat0 = np.sin(lat0)
    cos_theta = np.cos(theta_max)
    touches_pole = np.abs(sin_lat0) >= cos_theta

    safe_ratio = np.clip(sin_lat0 / np.where(touches_pole, 1.0, cos_theta), -1.0, 1.0)
    phi_star = np.arcsin(safe_ratio)
    phi_star = np.clip(phi_star, lat_min, lat_max)
    cos_arg = (cos_theta - sin_lat0 * np.sin(phi_star)) / np.maximum(np.cos(phi_star) * np.cos(lat0), 1e-6)
    cos_arg = np.clip(cos_arg, -1.0, 1.0)
    dlon_max = np.where(touches_pole, np.pi, np.arccos(cos_arg))

    return lon0 - dlon_max, lon0 + dlon_max, lat_min, lat_max


def great_circle_dist(lon0, lat0, lon1, lat1):
    d0 = lonlat_to_dir(lon0, lat0)
    d1 = lonlat_to_dir(lon1, lat1)
    d0 = d0 / np.linalg.norm(d0, axis=-1, keepdims=True)
    d1 = d1 / np.linalg.norm(d1, axis=-1, keepdims=True)
    cos_d = np.clip(np.sum(d0 * d1, axis=-1), -1.0, 1.0)
    return np.arccos(cos_d)


def test_pixel_ray_roundtrip():
    W, H = 4096, 2048
    rng = np.random.default_rng(0)
    px = rng.uniform(0, W, size=2000)
    py = rng.uniform(0, H, size=2000)

    lon, lat = pixel_to_lonlat(px, py, W, H)
    d = lonlat_to_dir(lon, lat)
    lon2, lat2 = point_to_lonlat(d)
    px2, py2 = lonlat_to_pixel(lon2, lat2, W, H)

    # atan2's numerator/denominator both -> 0 right at a pole (lon becomes
    # ill-defined there, which is inherent to the lon/lat parameterization
    # itself, not a bug), and asin's derivative blows up as |y/dist| -> 1, so
    # rows within ~0.5px of a pole get amplified float64 noise. 1e-2px is still
    # far tighter than a real formula bug would produce (those show up as
    # errors of many pixels or full wrap-arounds, not fractions of a pixel).
    assert np.allclose(px, px2, atol=1e-2), "pixel.x round-trip failed"
    assert np.allclose(py, py2, atol=1e-2), "pixel.y round-trip failed"
    print("OK: pixel -> lon/lat -> ray -> lon/lat -> pixel round-trips (2000 random samples)")


def test_cap_contains_true_extent():
    """Sample random points on a great-circle cap of radius theta_max around
    (lon0, lat0) and assert every one falls inside the returned lon/lat box -
    including caps centered very close to a pole."""
    rng = np.random.default_rng(1)
    n_centers = 300
    lat0 = rng.uniform(-np.pi / 2 * 0.999, np.pi / 2 * 0.999, size=n_centers)
    # bias half the centers close to a pole - the case a linearized Jacobian breaks on
    near_pole = rng.random(n_centers) < 0.5
    lat0[near_pole] = np.sign(lat0[near_pole]) * rng.uniform(0.95, 0.999, size=near_pole.sum()) * (np.pi / 2)
    lon0 = rng.uniform(-np.pi, np.pi, size=n_centers)
    theta_max = rng.uniform(0.01, 0.4, size=n_centers)  # up to ~23 degrees

    lon_min, lon_max, lat_min, lat_max = lonlat_box_from_angular_cap(lon0, lat0, theta_max)

    worst = 0.0
    for i in range(n_centers):
        # sample points ON the cap boundary (worst case for containment) at many bearings
        bearing = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        d0 = lonlat_to_dir(lon0[i], lat0[i])
        d0 = d0 / np.linalg.norm(d0)
        # build an orthonormal tangent basis at d0
        up = np.array([0.0, 1.0, 0.0]) if abs(d0[1]) < 0.999 else np.array([1.0, 0.0, 0.0])
        r = np.cross(up, d0); r /= np.linalg.norm(r)
        u = np.cross(d0, r)
        boundary_dirs = (np.cos(theta_max[i]) * d0
                         + np.sin(theta_max[i]) * (np.cos(bearing)[:, None] * r + np.sin(bearing)[:, None] * u))
        blon, blat = point_to_lonlat(boundary_dirs)

        # containment in latitude
        assert np.all(blat >= lat_min[i] - 1e-6) and np.all(blat <= lat_max[i] + 1e-6), \
            f"latitude containment failed at center {i} (lat0={lat0[i]:.3f})"

        # containment in longitude, handling wrap only for the *check* (the box
        # itself doesn't wrap - documented v1 limitation) by comparing the
        # shortest angular difference against the half-width actually used.
        dlon = lon_max[i] - lon0[i]
        ang_diff = np.abs(np.arctan2(np.sin(blon - lon0[i]), np.cos(blon - lon0[i])))
        bad = ang_diff > dlon + 1e-4
        # Points whose center cap already reaches the pole make longitude
        # containment moot (dlon saturates towards pi); skip those.
        if dlon < np.pi - 1e-3:
            assert not np.any(bad), f"longitude containment failed at center {i} (lat0={lat0[i]:.3f}, dlon={dlon:.3f})"
        worst = max(worst, ang_diff.max() - dlon)

    print(f"OK: spherical-cap box contains the true cap boundary for {n_centers} random centers "
          f"(half near a pole); max slack {worst:.4f} rad")


def test_saturates_at_pole_instead_of_blowing_up():
    """As the cap center approaches the pole, dlon_max must smoothly approach pi
    (full row width) - never NaN/inf, never a wild overshoot like a linearized
    tan-space Jacobian would produce."""
    theta_max = 0.05  # ~3 degrees, small and fixed
    lat0 = np.array([0.0, np.pi / 4, np.pi / 2 - 0.2, np.pi / 2 - 0.01, np.pi / 2 - 1e-4])
    lon0 = np.zeros_like(lat0)
    lon_min, lon_max, lat_min, lat_max = lonlat_box_from_angular_cap(lon0, lat0, theta_max)
    dlon = lon_max - lon0

    assert np.all(np.isfinite(dlon)), "dlon_max produced NaN/inf near the pole"
    assert np.all(np.diff(dlon) >= -1e-9), "dlon_max should grow monotonically as lat0 -> pole"
    assert dlon[-1] > np.pi - 0.05, f"dlon_max should saturate near pi at the pole, got {dlon[-1]:.4f}"
    print(f"OK: longitude half-width saturates smoothly to pi near the pole "
          f"(values: {np.round(dlon, 4).tolist()})")


if __name__ == "__main__":
    test_pixel_ray_roundtrip()
    test_cap_contains_true_extent()
    test_saturates_at_pole_instead_of_blowing_up()
    print("\nAll EQUI projection checks passed.")
