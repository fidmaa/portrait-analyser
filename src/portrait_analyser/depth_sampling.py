"""Depth-map sampling helpers for robust surface-distance measurement.

Complements incisor.py's single-point-pair 3D distance and face.py's
sample_depth_at_point(): instead of reading one raw depth pixel per point
(nearest-neighbour + integer kernel median), these helpers pre-filter the
whole depth map once with a same-size median filter and then sample it
bilinearly at fractional photo-space coordinates. This smooths out
TrueDepth sensor noise and correctly treats zero disparity as invalid,
which matters most when walking many points along a curved surface (e.g.
a neck arc): naive per-pixel sampling lets that noise accumulate into a
large measurement error, since each noisy step is summed as if it were
real surface displacement.
"""

import math

from PIL import ImageFilter

from .incisor import depth_raw_to_distance_cm, pixel_to_mm, vector_length_3d


def median_filter_depthmap(depthmap, size=3):
    """Return a same-size, single-channel median-filtered depth map."""
    if size < 3 or size % 2 == 0:
        raise ValueError("median filter size must be an odd number >= 3")
    if len(depthmap.getbands()) > 1:
        depthmap = depthmap.getchannel(0)
    else:
        depthmap = depthmap.convert("L")
    return depthmap.filter(ImageFilter.MedianFilter(size=size))


def bilinear_sample(image, x, y, invalid_value=None):
    """Sample a PIL image at fractional coordinates using bilinear interpolation.

    When ``invalid_value`` is provided, return ``None`` if a contributing pixel
    has that value.  This prevents interpolation across holes in a depth map.
    """
    if image.width == 0 or image.height == 0:
        return None

    x = min(max(float(x), 0.0), image.width - 1.0)
    y = min(max(float(y), 0.0), image.height - 1.0)
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = min(x0 + 1, image.width - 1)
    y1 = min(y0 + 1, image.height - 1)
    fraction_x = x - x0
    fraction_y = y - y0

    samples = (
        (image.getpixel((x0, y0)), (1.0 - fraction_x) * (1.0 - fraction_y)),
        (image.getpixel((x1, y0)), fraction_x * (1.0 - fraction_y)),
        (image.getpixel((x0, y1)), (1.0 - fraction_x) * fraction_y),
        (image.getpixel((x1, y1)), fraction_x * fraction_y),
    )

    weighted_value = 0.0
    for value, weight in samples:
        if weight == 0:
            continue
        if isinstance(value, tuple):
            value = value[0]
        if invalid_value is not None and value == invalid_value:
            return None
        weighted_value += value * weight
    return weighted_value


def sample_points_along_line(x1, y1, x2, y2, step):
    """Yield points separated by approximately ``step`` pixels on a 2D line.

    The first and last points are always included and all intervals have equal
    length.  This also makes the returned set independent of line direction.
    """
    if step <= 0:
        raise ValueError("step must be greater than zero")

    dist_x = x2 - x1
    dist_y = y2 - y1
    line_length = math.hypot(dist_x, dist_y)

    if line_length == 0:
        yield (x1, y1)
        return

    interval_count = max(1, round(line_length / step))
    for index in range(interval_count + 1):
        ratio = index / interval_count
        yield (x1 + dist_x * ratio, y1 + dist_y * ratio)


def sample_filtered_depth(filtered_depthmap, photo_x, photo_y, photo_width, photo_height):
    """Bilinearly sample raw depth at a photo-space point from a pre-filtered
    depth map, mapping into the depth map's native resolution.

    :param filtered_depthmap: single-channel depth map, typically produced by
        median_filter_depthmap()
    :param photo_x: x coordinate in photo-space pixels
    :param photo_y: y coordinate in photo-space pixels
    :param photo_width: full photo width in pixels
    :param photo_height: full photo height in pixels
    :returns: raw depth value, or None over invalid (zero) disparity
    """
    depth_width, depth_height = filtered_depthmap.size
    depth_x = photo_x * (depth_width - 1) / (photo_width - 1)
    depth_y = photo_y * (depth_height - 1) / (photo_height - 1)
    return bilinear_sample(filtered_depthmap, depth_x, depth_y, invalid_value=0)


def measure_filtered_surface_length(
    filtered_depthmap,
    points_photo,
    photo_width,
    photo_height,
    float_min,
    float_max,
):
    """Sum 3D Euclidean distance across consecutive photo-space points,
    reading depth from a pre-filtered map via bilinear sampling.

    :param filtered_depthmap: single-channel depth map, typically produced by
        median_filter_depthmap()
    :param points_photo: iterable of (x, y) photo-space pixel coordinates,
        e.g. from sample_points_along_line()
    :param photo_width: full photo width in pixels
    :param photo_height: full photo height in pixels
    :param float_min: EXIF FloatMinValue
    :param float_max: EXIF FloatMaxValue
    :returns: total length in millimeters, or None if fewer than 2 points
        were given, or any point falls on invalid depth, or any point lies
        outside the calibration polynomial's trustworthy distance range
    """
    points_3d = []
    for x, y in points_photo:
        raw_depth = sample_filtered_depth(
            filtered_depthmap, x, y, photo_width, photo_height
        )
        if raw_depth is None:
            return None

        z_cm = depth_raw_to_distance_cm(raw_depth, float_min, float_max)
        if z_cm is None:
            return None

        x_mm = pixel_to_mm(x, z_cm, photo_width)
        y_mm = pixel_to_mm(y, z_cm, photo_height)
        if x_mm is None or y_mm is None:
            # Beyond the calibrated range the conversion is meaningless; a
            # partial walk would silently report a shorter surface, so fail
            # the whole measurement instead.
            return None

        points_3d.append((x_mm, y_mm, z_cm * 10.0))

    if len(points_3d) < 2:
        return None

    return sum(
        vector_length_3d(*point_1, *point_2)
        for point_1, point_2 in zip(points_3d, points_3d[1:])
    )
