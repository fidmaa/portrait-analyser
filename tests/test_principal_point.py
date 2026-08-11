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


class TestCalibrationRangeGuard:
    """The calibration polynomial is only fitted between ~20 and ~70 cm.

    Past ~80 cm it departs sharply from the physical f/(Z+delta) behaviour and
    it crosses zero at ~104 cm, after which pixels-per-mm goes negative. A
    depth map routinely contains such distances (background), so conversion
    must refuse rather than return a nonsensical number.
    """

    def test_in_range_distances_convert(self):
        for distance_cm in (15.0, 25.0, 35.0, 50.0, 80.0):
            assert pixels_per_mm_at_distance(distance_cm) is not None
            assert pixel_to_mm(2000, distance_cm, 2320) is not None

    def test_too_close_is_rejected(self):
        assert pixels_per_mm_at_distance(14.9) is None
        assert pixel_to_mm(2000, 14.9, 2320) is None

    def test_too_far_is_rejected(self):
        for distance_cm in (80.1, 100.0, 110.0, 196.7):
            assert pixels_per_mm_at_distance(distance_cm) is None
            assert pixel_to_mm(2000, distance_cm, 2320) is None

    def test_polynomial_would_go_negative_beyond_range(self):
        """Guards the reason the upper bound exists, not just the bound."""
        d = 110.0
        raw = (
            30.79912
            - 1.346418 * d
            + 0.03009753 * d**2
            - 0.0003733656 * d**3
            + 0.000002521213 * d**4
            - 7.49986e-9 * d**5
        )
        assert raw < 0
        assert pixels_per_mm_at_distance(d) is None

    def test_surface_length_refuses_out_of_range_points(self):
        """A line crossing into the background must fail, not report a length."""
        from PIL import Image

        from portrait_analyser.depth_sampling import (
            measure_filtered_surface_length,
            median_filter_depthmap,
        )

        photo_width, photo_height = 480, 640
        # Depth 20 with this disparity range sits at ~150 cm -- background.
        depthmap = Image.new("L", (photo_width, photo_height), 200)
        for x in range(240, photo_width):
            for y in range(photo_height):
                depthmap.putpixel((x, y), 20)
        filtered = median_filter_depthmap(depthmap)

        near_only = measure_filtered_surface_length(
            filtered,
            [(50.0, 300.0), (100.0, 300.0), (150.0, 300.0)],
            photo_width,
            photo_height,
            float_min=0.5,
            float_max=4.0,
        )
        assert near_only is not None

        crossing = measure_filtered_surface_length(
            filtered,
            [(50.0, 300.0), (240.0, 300.0), (400.0, 300.0)],
            photo_width,
            photo_height,
            float_min=0.5,
            float_max=4.0,
        )
        assert crossing is None
