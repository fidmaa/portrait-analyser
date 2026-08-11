"""3D incisor distance measurement using depth map data.

Converts pixel coordinates and raw depth values into physical units (mm/cm)
and computes Euclidean 3D distance between upper and lower incisor centroids.

The calibration polynomial was fitted to Apple TrueDepth front camera data
at original image resolution (~2300x3000). See fidmaa-gui for calibration
source data and methodology.
"""

import math

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


def pixels_per_mm_at_distance(distance_cm):
    """How many pixels in the original image correspond to 1mm at a given distance.

    Calibration polynomial fitted to TrueDepth camera data.
    Constants from own calibration data and curve fitted by MyCurveFit.com.

    :param distance_cm: distance from camera in centimeters
    :returns: pixels per millimeter at the given distance, or None when the
        distance falls outside the calibrated range (see
        MIN_CALIBRATED_DISTANCE_CM / MAX_CALIBRATED_DISTANCE_CM)
    """
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


def pixel_to_mm(pixel_coord, distance_cm, image_dimension):
    """Convert a pixel coordinate to physical millimeters at a given distance.

    The pinhole camera model requires coordinates measured from the optical
    axis (the principal point, approximated here as the image centre), not
    from the top-left corner. Without this correction, points at different
    depths pick up a phantom lateral displacement proportional to their
    distance from the image centre and the depth difference between them.

    The pixel coordinate must be in original (full-resolution) image space,
    matching the calibration polynomial's expected resolution (~2300x3000).

    :param pixel_coord: coordinate in pixels (original image space, from top-left)
    :param distance_cm: distance from camera in centimeters
    :param image_dimension: full image width (for an x coordinate) or height
        (for a y coordinate) in pixels, used to locate the principal point
    :returns: physical distance in millimeters relative to the optical axis,
        or None if the distance is outside the calibrated range
    """
    ppmm = pixels_per_mm_at_distance(distance_cm)
    if ppmm is None or ppmm <= 0:
        return None
    centred_coord = pixel_coord - image_dimension / 2.0
    return centred_coord / ppmm


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
    :returns: (distance_3d_mm, upper_distance_cm, lower_distance_cm) or None
    """
    upper_z_cm = depth_raw_to_distance_cm(upper_depth_raw, float_min, float_max)
    lower_z_cm = depth_raw_to_distance_cm(lower_depth_raw, float_min, float_max)

    if upper_z_cm is None or lower_z_cm is None:
        return None

    upper_x_mm = pixel_to_mm(upper_centroid[0], upper_z_cm, image_width)
    upper_y_mm = pixel_to_mm(upper_centroid[1], upper_z_cm, image_height)
    lower_x_mm = pixel_to_mm(lower_centroid[0], lower_z_cm, image_width)
    lower_y_mm = pixel_to_mm(lower_centroid[1], lower_z_cm, image_height)

    if any(v is None for v in (upper_x_mm, upper_y_mm, lower_x_mm, lower_y_mm)):
        return None

    # Convert Z from cm to mm for consistent units
    upper_z_mm = upper_z_cm * 10
    lower_z_mm = lower_z_cm * 10

    distance = vector_length_3d(
        upper_x_mm, upper_y_mm, upper_z_mm,
        lower_x_mm, lower_y_mm, lower_z_mm,
    )

    return distance, upper_z_cm, lower_z_cm
