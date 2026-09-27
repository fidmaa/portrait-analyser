"""3D thyromental distance (TMD) measurement using depth map data.

Computes the physical distance between chin (mentum) and neck midpoint
using TrueDepth camera calibration data.
"""

from .incisor import point_to_mm, raw_depth_to_distance_cm, vector_length_3d


def compute_tmd_3d(
    chin_coord,
    neck_coord,
    chin_depth_raw,
    neck_depth_raw,
    float_min,
    float_max,
    image_width,
    image_height,
    *,
    camera=None,
    zero_is_invalid=None,
):
    """Compute 3D thyromental distance (chin to neck midpoint).

    :param chin_coord: (x, y) chin position in photo-space pixels
    :param neck_coord: (x, y) neck midpoint in photo-space pixels
    :param chin_depth_raw: raw depth pixel value at chin
    :param neck_depth_raw: raw depth pixel value at neck midpoint
    :param float_min: EXIF FloatMinValue
    :param float_max: EXIF FloatMaxValue
    :param image_width: full photo width in pixels (principal point reference)
    :param image_height: full photo height in pixels (principal point reference)
    :param camera: optional :class:`portrait_analyser.camera.CameraModel`;
        ``None`` keeps the legacy calibration polynomial. With a camera, raw
        depth code 0 ("no depth" in capture-app files) yields None.
    :param zero_is_invalid: ``portrait.depth_code_zero_is_invalid``; None =
        infer from ``camera``
    :returns: (distance_3d_mm, chin_z_cm, neck_z_cm) or None
    """
    chin_z_cm = raw_depth_to_distance_cm(
        chin_depth_raw, float_min, float_max, camera, zero_is_invalid=zero_is_invalid
    )
    neck_z_cm = raw_depth_to_distance_cm(
        neck_depth_raw, float_min, float_max, camera, zero_is_invalid=zero_is_invalid
    )

    if chin_z_cm is None or neck_z_cm is None:
        return None

    chin_mm = point_to_mm(
        chin_coord[0], chin_coord[1], chin_z_cm, image_width, image_height, camera
    )
    neck_mm = point_to_mm(
        neck_coord[0], neck_coord[1], neck_z_cm, image_width, image_height, camera
    )
    if chin_mm is None or neck_mm is None:
        return None
    chin_x_mm, chin_y_mm = chin_mm
    neck_x_mm, neck_y_mm = neck_mm

    chin_z_mm = chin_z_cm * 10
    neck_z_mm = neck_z_cm * 10

    distance = vector_length_3d(
        chin_x_mm, chin_y_mm, chin_z_mm,
        neck_x_mm, neck_y_mm, neck_z_mm,
    )

    return distance, chin_z_cm, neck_z_cm
