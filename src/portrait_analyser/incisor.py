"""3D incisor distance measurement using depth map data.

Converts pixel coordinates and raw depth values into physical units (mm/cm)
and computes Euclidean 3D distance between upper and lower incisor centroids.

The calibration polynomial was fitted to Apple TrueDepth front camera data
at original image resolution (~2300x3000). See fidmaa-gui for calibration
source data and methodology.
"""

import math

from .camera import camera_axis

# Range over which the calibration polynomial below is trustworthy. It was
# fitted to measurements taken between roughly 20 and 70 cm; outside that
# span a 5th-degree fit has nothing to hold it down. It peaks and turns over
# past ~80 cm, crosses zero at ~105 cm and goes negative beyond that, which
# would silently yield negative millimetres-per-pixel. A depth map can easily
# contain such distances -- the disparity range of a typical portrait spans
# ~26 to ~197 cm -- so any point beyond this range is reported as
# unmeasurable rather than converted.
MIN_CALIBRATED_DISTANCE_CM = 15.0
MAX_CALIBRATED_DISTANCE_CM = 80.0

# Working range of the pinhole (file-intrinsics) conversion. The pinhole
# model itself has no such limit, but the TrueDepth sensor does: it is
# reliable at roughly 20-50 cm and a face is never measured closer than
# 10 cm or farther than 1.5 m. Anything outside is far more likely to be a
# decoding error (e.g. inverted depth putting the face at ~3 m) or an
# invalid "no depth" pixel than a real measurement, so it is refused.
MIN_PINHOLE_DISTANCE_CM = 10.0
MAX_PINHOLE_DISTANCE_CM = 150.0


def _pinhole_distance_ok(distance_cm):
    return (
        distance_cm is not None
        and MIN_PINHOLE_DISTANCE_CM <= distance_cm <= MAX_PINHOLE_DISTANCE_CM
    )


def _zero_is_invalid(zero_is_invalid, camera):
    # The file format decides (IOSPortrait.depth_code_zero_is_invalid); when
    # the caller does not say, a camera implies a capture-app file.
    return (camera is not None) if zero_is_invalid is None else bool(zero_is_invalid)


def raw_depth_to_distance_cm(value, float_min, float_max, camera=None, *, zero_is_invalid=None):
    """:func:`depth_raw_to_distance_cm`, honouring the capture-app encoding.

    In capture-app depth maps (``IOSPortrait.depth_code_zero_is_invalid``)
    code ``0`` means "no depth" and returns ``None``; in Camera-app maps it
    is a valid farthest depth, exactly as before. Pass
    ``zero_is_invalid=portrait.depth_code_zero_is_invalid``; when it is
    ``None`` a given ``camera`` implies a capture-app file.
    """
    if value is None:
        return None
    if _zero_is_invalid(zero_is_invalid, camera) and value == 0:
        return None
    return depth_raw_to_distance_cm(value, float_min, float_max)


def depth_raw_to_distance_cm(value, float_min, float_max):
    """Convert raw depth pixel value to physical distance in centimeters.

    Uses disparity-based conversion matching Apple TrueDepth camera format.
    The depth map stores disparity (inverse of distance), not linear depth.

    :param value: raw depth pixel value (0-255)
    :param float_min: EXIF FloatMinValue from depth metadata
    :param float_max: EXIF FloatMaxValue from depth metadata
    :returns: distance in centimeters, or None if disparity is zero
    """
    disparity = float_max * value / 255 + float_min * (1 - value / 255)
    if disparity == 0:
        return None
    return 100.0 / disparity


def pixels_per_mm_at_distance(distance_cm, *, focal_px=None):
    """How many pixels in the original image correspond to 1mm at a given distance.

    Without ``focal_px``: calibration polynomial fitted to TrueDepth camera
    data (constants from own calibration data, curve fitted by
    MyCurveFit.com), only trusted between MIN_CALIBRATED_DISTANCE_CM and
    MAX_CALIBRATED_DISTANCE_CM.

    With ``focal_px`` (a file's own focal length in photo pixels, see
    :class:`portrait_analyser.camera.CameraModel`): the pinhole model
    ``focal_px / distance_mm``, within MIN_PINHOLE_DISTANCE_CM ..
    MAX_PINHOLE_DISTANCE_CM (10-150 cm).

    :param distance_cm: distance from camera in centimeters
    :param focal_px: optional focal length in pixels (keyword-only)
    :returns: pixels per millimeter at the given distance, or None when the
        distance falls outside the calibrated range (polynomial) or the
        pinhole working range
    """
    if focal_px is not None:
        if not _pinhole_distance_ok(distance_cm):
            return None
        return focal_px / (distance_cm * 10.0)

    if not MIN_CALIBRATED_DISTANCE_CM <= distance_cm <= MAX_CALIBRATED_DISTANCE_CM:
        return None

    d = distance_cm
    return (
        30.79912
        - 1.346418 * d
        + 0.03009753 * d**2
        - 0.0003733656 * d**3
        + 0.000002521213 * d**4
        - 7.49986e-9 * d**5
    )


def pixel_to_mm(pixel_coord, distance_cm, image_dimension, *, focal_px=None, principal_px=None):
    """Convert a pixel coordinate to physical millimeters at a given distance.

    The pinhole camera model requires coordinates measured from the optical
    axis (the principal point), not from the top-left corner. Without this
    correction, points at different depths pick up a phantom lateral
    displacement proportional to their distance from the image centre and
    the depth difference between them.

    Legacy mode (``focal_px=None``): the calibration polynomial, with the
    principal point approximated as the image centre. The pixel coordinate
    must be in original (full-resolution) image space, matching the
    polynomial's expected resolution (~2300x3000).

    Camera mode (``focal_px`` given): ``(pixel - principal) * distance_mm /
    focal_px``, with ``principal_px`` defaulting to ``image_dimension / 2``,
    for distances within MIN_PINHOLE_DISTANCE_CM .. MAX_PINHOLE_DISTANCE_CM
    (10-150 cm; None outside).
    Both must be in the same pixel space as ``pixel_coord`` (the upright
    photo returned by ``load_image``).

    :param pixel_coord: coordinate in pixels (original image space, from top-left)
    :param distance_cm: distance from camera in centimeters
    :param image_dimension: full image width (for an x coordinate) or height
        (for a y coordinate) in pixels, used to locate the principal point
        when none is given
    :param focal_px: optional focal length for this axis, pixels
    :param principal_px: optional principal point for this axis, pixels
    :returns: physical distance in millimeters relative to the optical axis,
        or None if the distance is outside the calibrated range (legacy) or
        the pinhole working range (camera mode)
    """
    if focal_px is not None:
        if not _pinhole_distance_ok(distance_cm):
            return None
        principal = image_dimension / 2.0 if principal_px is None else principal_px
        return (pixel_coord - principal) * distance_cm * 10.0 / focal_px

    ppmm = pixels_per_mm_at_distance(distance_cm)
    if ppmm is None or ppmm <= 0:
        return None
    centred_coord = pixel_coord - image_dimension / 2.0
    return centred_coord / ppmm


def point_to_mm(x, y, distance_cm, image_width, image_height, camera=None):
    """Convert a photo-space point at ``distance_cm`` to ``(x_mm, y_mm)``.

    Thin wrapper over :func:`pixel_to_mm` for both axes; ``camera`` is an
    optional :class:`portrait_analyser.camera.CameraModel` (``None`` = legacy
    polynomial). Returns ``None`` when either axis cannot be converted.
    """
    focal_x, principal_x = camera_axis(camera, "x")
    focal_y, principal_y = camera_axis(camera, "y")
    x_mm = pixel_to_mm(x, distance_cm, image_width, focal_px=focal_x, principal_px=principal_x)
    y_mm = pixel_to_mm(y, distance_cm, image_height, focal_px=focal_y, principal_px=principal_y)
    if x_mm is None or y_mm is None:
        return None
    return x_mm, y_mm


def vector_length_3d(x1, y1, z1, x2, y2, z2):
    """Euclidean distance between two 3D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def compute_incisor_distance_3d(
    upper_centroid,
    lower_centroid,
    upper_depth_raw,
    lower_depth_raw,
    float_min,
    float_max,
    image_width,
    image_height,
    *,
    camera=None,
    zero_is_invalid=None,
):
    """Compute 3D Euclidean distance between upper and lower incisor centroids.

    Converts pixel coordinates and depth values to physical mm/cm,
    then computes the 3D vector length.

    :param upper_centroid: (x, y) in photo-space pixels
    :param lower_centroid: (x, y) in photo-space pixels
    :param upper_depth_raw: raw depth pixel value at upper centroid
    :param lower_depth_raw: raw depth pixel value at lower centroid
    :param float_min: EXIF FloatMinValue
    :param float_max: EXIF FloatMaxValue
    :param image_width: full photo width in pixels (principal point reference)
    :param image_height: full photo height in pixels (principal point reference)
    :param camera: optional :class:`portrait_analyser.camera.CameraModel`;
        ``None`` keeps the legacy calibration polynomial. With a camera, raw
        depth code 0 ("no depth" in capture-app files) yields None.
    :param zero_is_invalid: ``portrait.depth_code_zero_is_invalid``; None =
        infer from ``camera`` (see :func:`raw_depth_to_distance_cm`)
    :returns: (distance_3d_mm, upper_distance_cm, lower_distance_cm) or None
    """
    upper_z_cm = raw_depth_to_distance_cm(
        upper_depth_raw, float_min, float_max, camera, zero_is_invalid=zero_is_invalid
    )
    lower_z_cm = raw_depth_to_distance_cm(
        lower_depth_raw, float_min, float_max, camera, zero_is_invalid=zero_is_invalid
    )

    if upper_z_cm is None or lower_z_cm is None:
        return None

    upper_mm = point_to_mm(
        upper_centroid[0], upper_centroid[1], upper_z_cm, image_width, image_height, camera
    )
    lower_mm = point_to_mm(
        lower_centroid[0], lower_centroid[1], lower_z_cm, image_width, image_height, camera
    )
    if upper_mm is None or lower_mm is None:
        return None
    upper_x_mm, upper_y_mm = upper_mm
    lower_x_mm, lower_y_mm = lower_mm

    # Convert Z from cm to mm for consistent units
    upper_z_mm = upper_z_cm * 10
    lower_z_mm = lower_z_cm * 10

    distance = vector_length_3d(
        upper_x_mm, upper_y_mm, upper_z_mm,
        lower_x_mm, lower_y_mm, lower_z_mm,
    )

    return distance, upper_z_cm, lower_z_cm
