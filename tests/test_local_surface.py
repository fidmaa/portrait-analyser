import numpy as np

from portrait_analyser.local_surface import (
    SurfaceFeature,
    score_local_surface_feature,
)


def _tilted_surface(size=51):
    coordinates = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(coordinates, coordinates)
    z = 100.0 + 15.0 * x + 4.0 * y
    radial = np.hypot(x, y) / np.sqrt(2.0)
    return x, y, z, radial


def test_local_peak_ignores_camera_tilt_and_finds_chin_bump():
    x, y, z, radial = _tilted_surface()
    bump_x, bump_y = 0.18, -0.08
    z -= 8.0 * np.exp(-((x - bump_x) ** 2 + (y - bump_y) ** 2) / 0.025)

    result = score_local_surface_feature(
        x,
        y,
        z,
        np.ones(z.shape, dtype=bool),
        feature=SurfaceFeature.PEAK,
        radial_fraction=radial,
    )
    best_y, best_x = np.unravel_index(np.argmin(result.score), result.score.shape)

    assert abs(x[best_y, best_x] - bump_x) < 0.12
    assert abs(y[best_y, best_x] - bump_y) < 0.12
    assert best_x > 10  # absolute nearest-to-camera depth is at the left border


def test_local_valley_ignores_tilt_and_an_unrelated_protruding_bone():
    x, y, z, radial = _tilted_surface()
    valley_x, valley_y = -0.12, 0.05
    z += 7.0 * np.exp(-((x - valley_x) ** 2 + (y - valley_y) ** 2) / 0.02)
    z[y > 0.75] -= 5.0

    result = score_local_surface_feature(
        x,
        y,
        z,
        np.ones(z.shape, dtype=bool),
        feature=SurfaceFeature.VALLEY,
        radial_fraction=radial,
    )
    best_y, best_x = np.unravel_index(np.argmin(result.score), result.score.shape)

    assert abs(x[best_y, best_x] - valley_x) < 0.12
    assert abs(y[best_y, best_x] - valley_y) < 0.12
    assert y[best_y, best_x] < 0.75


def test_invalid_border_cannot_be_selected():
    x, y, z, radial = _tilted_surface(21)
    valid = np.ones(z.shape, dtype=bool)
    valid[:, :4] = False

    result = score_local_surface_feature(
        x,
        y,
        z,
        valid,
        feature=SurfaceFeature.PEAK,
        radial_fraction=radial,
    )

    assert np.isinf(result.score[:, :4]).all()
    assert np.isnan(result.residual[:, :4]).all()
