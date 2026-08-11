"""Tests for filtered/bilinear surface-distance sampling helpers."""

import math

import pytest
from PIL import Image

from portrait_analyser.depth_sampling import (
    bilinear_sample,
    measure_filtered_surface_length,
    median_filter_depthmap,
    sample_filtered_depth,
    sample_points_along_line,
)


class TestSamplePointsAlongLine:
    def test_uses_approximately_requested_step_and_keeps_endpoint(self):
        points = list(sample_points_along_line(0, 0, 10, 0, step=3))

        expected = [(0, 0), (10 / 3, 0), (20 / 3, 0), (10, 0)]
        assert len(points) == len(expected)
        for point, expected_point in zip(points, expected, strict=True):
            assert point == pytest.approx(expected_point)

    def test_diagonal_spacing_is_measured_along_line(self):
        points = list(sample_points_along_line(0, 0, 6, 8, step=5))

        assert points == [(0, 0), (3, 4), (6, 8)]

    def test_reversing_line_returns_same_points_in_reverse(self):
        forward = list(sample_points_along_line(2, 3, 15, 9, step=4))
        backward = list(sample_points_along_line(15, 9, 2, 3, step=4))

        assert len(forward) == len(backward)
        for point, reverse_point in zip(forward, reversed(backward), strict=True):
            assert point == pytest.approx(reverse_point)

    def test_rejects_non_positive_step(self):
        with pytest.raises(ValueError):
            list(sample_points_along_line(0, 0, 10, 0, step=0))


class TestFilteredDepthSampling:
    def test_median_filter_preserves_size_and_removes_impulse(self):
        depthmap = Image.new("L", (5, 5), 100)
        depthmap.putpixel((2, 2), 255)

        filtered = median_filter_depthmap(depthmap, size=3)

        assert filtered.size == depthmap.size
        assert filtered.getpixel((2, 2)) == 100

    def test_median_filter_uses_raw_first_channel(self):
        depthmap = Image.new("RGB", (3, 3), (100, 10, 200))

        filtered = median_filter_depthmap(depthmap, size=3)

        assert filtered.mode == "L"
        assert filtered.getpixel((1, 1)) == 100

    def test_bilinear_sample_interpolates_fractional_coordinate(self):
        image = Image.new("L", (2, 2))
        image.putdata([0, 10, 20, 30])

        assert bilinear_sample(image, 0.5, 0.5) == pytest.approx(15.0)

    def test_bilinear_sample_rejects_contributing_invalid_depth(self):
        image = Image.new("L", (2, 2))
        image.putdata([0, 10, 20, 30])

        assert bilinear_sample(image, 0.5, 0.5, invalid_value=0) is None

    def test_sample_filtered_depth_maps_photo_space_to_native_resolution(self):
        # 2x2 depth map covering a 480x640 photo: photo centre should read
        # the average of all four corners via bilinear interpolation.
        depthmap = Image.new("L", (2, 2))
        depthmap.putdata([100, 120, 140, 160])
        filtered = median_filter_depthmap(depthmap, size=3)

        value = sample_filtered_depth(filtered, 240, 320, 480, 640)

        assert value == pytest.approx(130.0, abs=5.0)

    def test_sample_filtered_depth_flags_zero_disparity_as_invalid(self):
        depthmap = Image.new("L", (4, 4), 0)
        filtered = median_filter_depthmap(depthmap, size=3)

        assert sample_filtered_depth(filtered, 2, 2, 4, 4) is None


class TestMeasureFilteredSurfaceLength:
    def test_stable_on_flat_surface(self):
        """A flat depth map measured at increasing sample density should
        converge to the straight-line pixel-to-mm distance, not inflate
        with sensor noise the way naive per-pixel walking does."""
        photo_width, photo_height = 480, 640
        depthmap = Image.new("L", (photo_width, photo_height), 100)
        filtered = median_filter_depthmap(depthmap)

        # float_max=4.0 puts depth 100 at ~53 cm. With the earlier float_max=2.0
        # it landed at ~92 cm, outside the calibration polynomial's trustworthy
        # range, so pixel_to_mm() now correctly refuses to convert it.
        lengths = [
            measure_filtered_surface_length(
                filtered,
                list(sample_points_along_line(10, 20, 41, 20, step)),
                photo_width,
                photo_height,
                float_min=0.5,
                float_max=4.0,
            )
            for step in range(2, 8)
        ]

        assert all(length is not None for length in lengths)
        assert lengths == pytest.approx([lengths[0]] * len(lengths), rel=0.05)

    def test_point_order_independence(self):
        photo_width, photo_height = 480, 640
        depthmap = Image.new("L", (photo_width, photo_height), 140)
        filtered = median_filter_depthmap(depthmap)

        forward = measure_filtered_surface_length(
            filtered,
            list(sample_points_along_line(50, 60, 200, 260, 5)),
            photo_width, photo_height,
            float_min=0.5, float_max=2.0,
        )
        backward = measure_filtered_surface_length(
            filtered,
            list(sample_points_along_line(200, 260, 50, 60, 5)),
            photo_width, photo_height,
            float_min=0.5, float_max=2.0,
        )

        assert forward == pytest.approx(backward)

    def test_invalid_depth_along_path_returns_none(self):
        photo_width, photo_height = 10, 10
        depthmap = Image.new("L", (photo_width, photo_height), 100)
        # A single dropout pixel would be smoothed away by the median
        # filter (that's the point of filtering) -- use a block large
        # enough to survive it and still read as invalid.
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                depthmap.putpixel((5 + dx, 5 + dy), 0)
        filtered = median_filter_depthmap(depthmap)

        result = measure_filtered_surface_length(
            filtered,
            [(0, 5), (5, 5), (9, 5)],
            photo_width, photo_height,
            float_min=0.5, float_max=2.0,
        )

        assert result is None

    def test_single_point_returns_none(self):
        photo_width, photo_height = 10, 10
        depthmap = Image.new("L", (photo_width, photo_height), 100)
        filtered = median_filter_depthmap(depthmap)

        result = measure_filtered_surface_length(
            filtered, [(5, 5)], photo_width, photo_height,
            float_min=0.5, float_max=2.0,
        )

        assert result is None
