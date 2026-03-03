"""FaceMesh-based mouth opening measurement (fallback).

When teethmap-based incisor detection fails (e.g. patient has no upper teeth),
this module measures vertical mouth opening using MediaPipe FaceMesh outer lip
landmarks (0 = upper lip outer center, 17 = lower lip outer center) and
converts to 3D distance using the depth map.

Outer lip landmarks are used instead of inner (13/14) because for edentulous
patients the inner landmarks point into the dark mouth cavity, producing
unreliable depth readings.
"""

from dataclasses import dataclass

from .face import sample_depth_at_point
from .incisor import compute_incisor_distance_3d


@dataclass
class MouthMeasurement:
    """Mouth opening measurement from FaceMesh outer lip landmarks."""

    upper_point: tuple[float, float]  # (x, y) landmark 0 in photo pixels
    lower_point: tuple[float, float]  # (x, y) landmark 17 in photo pixels
    upper_depth_raw: int | None = None
    lower_depth_raw: int | None = None
    upper_distance_cm: float | None = None
    lower_distance_cm: float | None = None
    distance_3d_mm: float | None = None


# FaceMesh landmark indices for outer lip center
_UPPER_LIP_OUTER = 0
_LOWER_LIP_OUTER = 17


def compute_mouth_measurement_from_facemesh(
    landmarks,
    depthmap,
    photo_w,
    photo_h,
    float_min,
    float_max,
):
    """Compute mouth opening from FaceMesh landmarks using depth map.

    :param landmarks: tuple of 478 (x, y) in photo-space pixels (FaceMeshDebug.landmarks)
    :param depthmap: PIL Image depth map
    :param photo_w: photo width in pixels
    :param photo_h: photo height in pixels
    :param float_min: EXIF FloatMinValue
    :param float_max: EXIF FloatMaxValue
    :returns: MouthMeasurement or None if computation fails
    """
    if len(landmarks) < max(_UPPER_LIP_OUTER, _LOWER_LIP_OUTER) + 1:
        return None

    upper_point = landmarks[_UPPER_LIP_OUTER]
    lower_point = landmarks[_LOWER_LIP_OUTER]

    upper_depth_raw = sample_depth_at_point(
        depthmap, upper_point[0], upper_point[1], photo_w, photo_h
    )
    lower_depth_raw = sample_depth_at_point(
        depthmap, lower_point[0], lower_point[1], photo_w, photo_h
    )

    distance_3d_mm = None
    upper_distance_cm = None
    lower_distance_cm = None

    if (
        upper_depth_raw is not None
        and lower_depth_raw is not None
        and float_min is not None
        and float_max is not None
    ):
        result_3d = compute_incisor_distance_3d(
            upper_point,
            lower_point,
            upper_depth_raw,
            lower_depth_raw,
            float(float_min),
            float(float_max),
        )
        if result_3d is not None:
            distance_3d_mm, upper_distance_cm, lower_distance_cm = result_3d

    return MouthMeasurement(
        upper_point=upper_point,
        lower_point=lower_point,
        upper_depth_raw=upper_depth_raw,
        lower_depth_raw=lower_depth_raw,
        upper_distance_cm=upper_distance_cm,
        lower_distance_cm=lower_distance_cm,
        distance_3d_mm=distance_3d_mm,
    )
