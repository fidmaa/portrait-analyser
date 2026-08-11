"""Regression tests for principal-point centring in pixel-to-mm conversion.

Mirrors fidmaa-gui's tests/test_principal_point.py: the pinhole camera model
requires pixel coordinates measured from the optical axis (approximated by
the image centre), not from the top-left corner. Without this correction,
points at different depths pick up a phantom lateral displacement
proportional to their distance from the image centre and the depth
difference between them.
"""

import math

import pytest

from portrait_analyser.incisor import (
    compute_incisor_distance_3d,
    pixel_to_mm,
    pixels_per_mm_at_distance,
)

IMAGE_WIDTH = 2320
IMAGE_HEIGHT = 3087


def test_image_centre_is_origin_at_every_depth():
    at_30_cm = pixel_to_mm(IMAGE_WIDTH / 2.0, 30.0, IMAGE_WIDTH)
    at_36_cm = pixel_to_mm(IMAGE_WIDTH / 2.0, 36.0, IMAGE_WIDTH)

    assert at_30_cm == pytest.approx(0.0)
    assert at_36_cm == pytest.approx(0.0)


def test_same_depth_distance_keeps_calibrated_pixel_scale():
    distance_cm = 30.0
    point_1 = pixel_to_mm(870.0, distance_cm, IMAGE_WIDTH)
    point_2 = pixel_to_mm(1450.0, distance_cm, IMAGE_WIDTH)

    measured_dx = point_2 - point_1
    ppmm = pixels_per_mm_at_distance(distance_cm)
    expected_dx = (1450.0 - 870.0) / ppmm

    assert measured_dx == pytest.approx(expected_dx)


def test_motion_along_optical_axis_has_no_phantom_xy_component():
    x_at_30 = pixel_to_mm(IMAGE_WIDTH / 2.0, 30.0, IMAGE_WIDTH)
    x_at_36 = pixel_to_mm(IMAGE_WIDTH / 2.0, 36.0, IMAGE_WIDTH)
    point_1 = (x_at_30, 0.0, 300.0)
    point_2 = (x_at_36, 0.0, 360.0)

    measured_distance = math.dist(point_1, point_2)

    assert measured_distance == pytest.approx(60.0)


def test_points_equidistant_from_centre_are_symmetric():
    left = pixel_to_mm(IMAGE_WIDTH / 2.0 - 500, 30.0, IMAGE_WIDTH)
    right = pixel_to_mm(IMAGE_WIDTH / 2.0 + 500, 30.0, IMAGE_WIDTH)

    assert left == pytest.approx(-right)


def test_incisor_distance_has_no_phantom_offset_from_depth_difference():
    """Two points on the optical axis but at different depths should
    measure the pure Z distance, with no lateral component introduced by
    their (large) absolute pixel offset from the top-left corner."""
    centre = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)
    result = compute_incisor_distance_3d(
        upper_centroid=centre,
        lower_centroid=centre,
        upper_depth_raw=200,
        lower_depth_raw=180,
        float_min=0.5,
        float_max=2.0,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )
    assert result is not None
    distance_3d_mm, upper_cm, lower_cm = result
    expected = abs(upper_cm - lower_cm) * 10.0
    assert distance_3d_mm == pytest.approx(expected, rel=1e-6)
