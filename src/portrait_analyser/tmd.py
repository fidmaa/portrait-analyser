"""3D thyromental distance (TMD) measurement using depth map data.

Computes the physical distance between chin (mentum) and neck midpoint
using TrueDepth camera calibration data.
"""

from .incisor import depth_raw_to_distance_cm, pixel_to_mm, vector_length_3d


def compute_tmd_3d(
    chin_coord,
    neck_coord,
    chin_depth_raw,
    neck_depth_raw,
    float_min,
    float_max,
):
    """Compute 3D thyromental distance (chin to neck midpoint).

    :param chin_coord: (x, y) chin position in photo-space pixels
    :param neck_coord: (x, y) neck midpoint in photo-space pixels
    :param chin_depth_raw: raw depth pixel value at chin
    :param neck_depth_raw: raw depth pixel value at neck midpoint
    :param float_min: EXIF FloatMinValue
    :param float_max: EXIF FloatMaxValue
    :returns: (distance_3d_mm, chin_z_cm, neck_z_cm) or None
    """
    chin_z_cm = depth_raw_to_distance_cm(chin_depth_raw, float_min, float_max)
    neck_z_cm = depth_raw_to_distance_cm(neck_depth_raw, float_min, float_max)

    if chin_z_cm is None or neck_z_cm is None:
        return None

    chin_x_mm = pixel_to_mm(chin_coord[0], chin_z_cm)
    chin_y_mm = pixel_to_mm(chin_coord[1], chin_z_cm)
    neck_x_mm = pixel_to_mm(neck_coord[0], neck_z_cm)
    neck_y_mm = pixel_to_mm(neck_coord[1], neck_z_cm)

    if any(v is None for v in (chin_x_mm, chin_y_mm, neck_x_mm, neck_y_mm)):
        return None

    chin_z_mm = chin_z_cm * 10
    neck_z_mm = neck_z_cm * 10

    distance = vector_length_3d(
        chin_x_mm, chin_y_mm, chin_z_mm,
        neck_x_mm, neck_y_mm, neck_z_mm,
    )

    return distance, chin_z_cm, neck_z_cm
