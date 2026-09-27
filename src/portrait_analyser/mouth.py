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

from .depth_map import LegacyDepthMap
from .incisor import distance_3d_from_cm


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
    *,
    camera=None,
    zero_is_invalid=None,
    depth=None,
):
    """Compute mouth opening from FaceMesh landmarks using depth map.

    :param landmarks: tuple of 478 (x, y) in photo-space pixels (FaceMeshDebug.landmarks)
    :param depthmap: PIL Image depth map
    :param photo_w: photo width in pixels
    :param photo_h: photo height in pixels
    :param float_min: EXIF FloatMinValue
    :param float_max: EXIF FloatMaxValue
    :param camera: optional :class:`portrait_analyser.camera.CameraModel`;
        ``None`` keeps the legacy calibration polynomial
    :param zero_is_invalid: ``portrait.depth_code_zero_is_invalid`` -- True
        for capture-app depth maps, where code 0 means "no depth" and is
        excluded from sampling; None = infer from ``camera``
    :param depth: optional :class:`portrait_analyser.depth_map.DepthMap`
        (``portrait.depth``); when given, depth is sampled from it (full
        precision for capture-app files) and ``depthmap``/``float_min``/
        ``float_max``/``zero_is_invalid`` are ignored. ``upper_depth_raw`` /
        ``lower_depth_raw`` are then None for float maps.
    :returns: MouthMeasurement or None if computation fails
    """
    if len(landmarks) < max(_UPPER_LIP_OUTER, _LOWER_LIP_OUTER) + 1:
        return None

    upper_point = landmarks[_UPPER_LIP_OUTER]
    lower_point = landmarks[_LOWER_LIP_OUTER]

    if depth is None:
        # Capture-app depth encodes "no depth" as code 0 (decided by the file
        # format; a camera alone implies a capture-app file).
        zero_invalid = (camera is not None) if zero_is_invalid is None else bool(zero_is_invalid)
        depth = LegacyDepthMap(
            depthmap, float_min, float_max, (photo_w, photo_h), zero_is_invalid=zero_invalid
        )
    upper = depth.sample(upper_point[0], upper_point[1])
    lower = depth.sample(lower_point[0], lower_point[1])
    upper_depth_raw = None if upper is None else upper.raw
    lower_depth_raw = None if lower is None else lower.raw

    distance_3d_mm = None
    upper_distance_cm = None
    lower_distance_cm = None

    if upper is not None and lower is not None:
        result_3d = distance_3d_from_cm(
            upper_point,
            lower_point,
            upper.distance_cm,
            lower.distance_cm,
            photo_w,
            photo_h,
            camera,
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
