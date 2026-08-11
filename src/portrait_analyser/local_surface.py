"""Pose-invariant local surface feature scoring for anatomical landmarks."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np


class SurfaceFeature(str, Enum):
    """Local feature to rank after removing the patch's dominant tilt."""

    PEAK = "peak"
    VALLEY = "valley"


@dataclass(frozen=True)
class LocalSurfaceScores:
    """Lower scores are better candidates for the requested feature."""

    score: np.ndarray
    residual: np.ndarray
    baseline: np.ndarray
    valid: np.ndarray


def score_local_surface_feature(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    *,
    feature: SurfaceFeature,
    radial_fraction: np.ndarray | None = None,
    smoothing_size: int = 5,
    center_bias: float = 0.15,
) -> LocalSurfaceScores:
    """Score a peak or valley relative to a robust local baseline plane.

    Absolute camera depth is deliberately removed. A shallow head rotation is
    approximately planar within a small patch, so subtracting that plane leaves
    the anatomical bump or depression while discarding pose-induced tilt.

    ``radial_fraction`` may contain distances from the user's click divided by
    the patch radius. It adds only a weak tie-breaker towards the clicked area;
    sufficiently strong off-centre anatomy still wins.
    """
    x, y, z, valid = _validated_arrays(x, y, z, valid)
    feature = SurfaceFeature(feature)
    baseline = _robust_plane(x, y, z, valid)
    residual = z - baseline
    smoothed, smooth_valid = _masked_median(residual, valid, smoothing_size)

    score = smoothed.copy() if feature == SurfaceFeature.PEAK else -smoothed
    if radial_fraction is not None:
        radial_fraction = np.asarray(radial_fraction, dtype=np.float64)
        if radial_fraction.shape != z.shape:
            raise ValueError("radial_fraction must have the same shape as z")
        finite_residual = smoothed[smooth_valid]
        residual_scale = np.percentile(finite_residual, 90) - np.percentile(
            finite_residual, 10
        )
        residual_scale = max(float(residual_scale), np.finfo(np.float64).eps)
        score += center_bias * residual_scale * np.clip(radial_fraction, 0.0, 1.5) ** 2

    score[~smooth_valid] = np.inf
    residual[~valid] = np.nan
    baseline[~valid] = np.nan
    return LocalSurfaceScores(score, residual, baseline, smooth_valid)


def _validated_arrays(x, y, z, valid):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if x.shape != z.shape or y.shape != z.shape or valid.shape != z.shape:
        raise ValueError("x, y, z, and valid must have identical shapes")
    valid = valid & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(valid) < 6:
        raise ValueError("at least six valid surface points are required")
    return x, y, z, valid


def _robust_plane(x, y, z, valid, iterations=4):
    x_valid = x[valid]
    y_valid = y[valid]
    z_valid = z[valid]
    x_center = float(np.median(x_valid))
    y_center = float(np.median(y_valid))
    x_scale = max(float(np.std(x_valid)), np.finfo(np.float64).eps)
    y_scale = max(float(np.std(y_valid)), np.finfo(np.float64).eps)
    normalized_x = (x_valid - x_center) / x_scale
    normalized_y = (y_valid - y_center) / y_scale
    design = np.column_stack((normalized_x, normalized_y, np.ones_like(normalized_x)))
    weights = np.ones(len(z_valid), dtype=np.float64)

    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(iterations):
        square_root_weights = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * square_root_weights[:, None],
            z_valid * square_root_weights,
            rcond=None,
        )[0]
        errors = z_valid - design @ coefficients
        error_center = float(np.median(errors))
        absolute_deviation = np.abs(errors - error_center)
        mad = float(np.median(absolute_deviation))
        if mad <= np.finfo(np.float64).eps:
            break
        cutoff = 2.5 * 1.4826 * mad
        weights = np.minimum(1.0, cutoff / np.maximum(absolute_deviation, cutoff))

    full_x = (x - x_center) / x_scale
    full_y = (y - y_center) / y_scale
    return coefficients[0] * full_x + coefficients[1] * full_y + coefficients[2]


def _masked_median(values, valid, size):
    if size < 3 or size % 2 == 0:
        raise ValueError("smoothing_size must be an odd number >= 3")
    radius = size // 2
    masked = np.where(valid, values, np.nan)
    padded = np.pad(masked, radius, mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smoothed = np.nanmedian(windows, axis=(-2, -1))
    support = np.sum(np.isfinite(windows), axis=(-2, -1))
    smooth_valid = valid & np.isfinite(smoothed) & (support >= max(5, size * size // 3))
    return smoothed, smooth_valid
