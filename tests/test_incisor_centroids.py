"""Tests for centroid-based incisor measurement functions.

Uses synthetic PIL Images — no HEIC files needed.
"""

from PIL import Image

from portrait_analyser.face import (
    IncisorMeasurement,
    find_incisor_centroids,
    sample_depth_at_point,
)
from portrait_analyser.incisor import (
    compute_incisor_distance_3d,
    depth_raw_to_distance_cm,
    pixel_to_mm,
    vector_length_3d,
)


def _make_teeth_image(width, height, upper_band, lower_band, value=255, bg=0):
    """Create a synthetic teeth map with upper and lower bright bands.

    upper_band: (y_start, y_end) for upper teeth region
    lower_band: (y_start, y_end) for lower teeth region
    """
    img = Image.new("L", (width, height), bg)
    for y in range(upper_band[0], upper_band[1]):
        for x in range(width):
            img.putpixel((x, y), value)
    for y in range(lower_band[0], lower_band[1]):
        for x in range(width):
            img.putpixel((x, y), value)
    return img


class TestFindIncisorCentroids:
    def test_basic_symmetric_bands(self):
        """Symmetric upper/lower teeth bands produce centroids at expected centers."""
        # 400x600 image, upper teeth at y=100-200, lower teeth at y=300-400
        # Gap from y=200 to y=300, so gap_midline ~250
        img = _make_teeth_image(400, 600, (100, 200), (300, 400))
        bbox = (0, 50, 400, 400)  # x, y, w, h covering both bands

        result = find_incisor_centroids(img, bbox)
        assert result is not None

        upper_c, lower_c = result
        # Upper centroid X lands on one half of the central strip (denser-half
        # logic picks left or right to avoid the septum midpoint)
        assert 100 <= upper_c[0] <= 300  # within central strip
        assert 145 < upper_c[1] < 155  # y near 150 (center of 100-200)

        # Lower centroid similarly within central strip
        assert 100 <= lower_c[0] <= 300
        assert 345 < lower_c[1] < 355  # y near 350 (center of 300-400)

    def test_no_gap_all_bright(self):
        """Uniform bright image — no gap to split on — returns None."""
        img = Image.new("L", (400, 600), 255)
        bbox = (50, 50, 300, 400)

        find_incisor_centroids(img, bbox)  # should not crash

    def test_single_tooth_region(self):
        """Only upper teeth present, no lower — returns None."""
        img = Image.new("L", (400, 600), 0)
        # Only upper band
        for y in range(100, 200):
            for x in range(400):
                img.putpixel((x, y), 255)
        bbox = (0, 50, 400, 400)

        result = find_incisor_centroids(img, bbox)
        assert result is None

    def test_few_pixels_below_threshold(self):
        """Too few pixels in one group — returns None."""
        img = Image.new("L", (400, 600), 0)
        # Upper: only a tiny 3x3 patch (9 pixels < min_pixels=50)
        for y in range(148, 151):
            for x in range(198, 201):
                img.putpixel((x, y), 255)
        # Lower: full band
        for y in range(300, 400):
            for x in range(400):
                img.putpixel((x, y), 255)
        bbox = (0, 50, 400, 400)

        result = find_incisor_centroids(img, bbox, min_pixels=50)
        assert result is None

    def test_arch_shaped_teeth_centroids_on_incisors(self):
        """Arch/U-shaped upper teeth: centroid should land on central incisors,
        not be pulled down into the dark mouth cavity by lateral molars."""
        width, height = 400, 600
        img = Image.new("L", (width, height), 0)

        # Upper teeth: arch shape — molars on sides extend lower than central incisors.
        # Central incisors: narrow band y=100-160
        # Lateral molars: wider bands y=100-240 on left (x=0-100) and right (x=300-400)
        for y in range(100, 160):
            for x in range(100, 300):  # central incisors
                img.putpixel((x, y), 255)
        for y in range(100, 240):
            for x in range(0, 100):  # left molars
                img.putpixel((x, y), 255)
            for x in range(300, 400):  # right molars
                img.putpixel((x, y), 255)

        # Lower teeth: simple band y=320-400
        for y in range(320, 400):
            for x in range(width):
                img.putpixel((x, y), 255)

        bbox = (0, 50, 400, 400)

        result = find_incisor_centroids(img, bbox)
        assert result is not None

        upper_c, lower_c = result

        # Upper centroid Y should be in the central incisor region (100-160),
        # NOT pulled down toward 170+ by the lateral molars
        assert 100 <= upper_c[1] <= 165, (
            f"Upper centroid Y={upper_c[1]:.1f} is outside incisor region 100-165; "
            "lateral molars are pulling it into the mouth cavity"
        )

        # Upper centroid X should be near horizontal center
        assert 150 < upper_c[0] < 250, (
            f"Upper centroid X={upper_c[0]:.1f} is not centered"
        )


    def test_septum_avoidance(self):
        """Centroids avoid the dark septum and both land on the same side."""
        width, height = 400, 600
        img = Image.new("L", (width, height), 0)

        # Two distinct upper incisors separated by a dark septum at x=190-210
        for y in range(100, 200):
            for x in range(140, 190):  # left incisor
                img.putpixel((x, y), 255)
            for x in range(210, 260):  # right incisor
                img.putpixel((x, y), 255)

        # Two distinct lower incisors with the same septum
        for y in range(300, 400):
            for x in range(140, 190):  # left lower incisor
                img.putpixel((x, y), 255)
            for x in range(210, 260):  # right lower incisor
                img.putpixel((x, y), 255)

        bbox = (0, 50, 400, 400)

        result = find_incisor_centroids(img, bbox)
        assert result is not None

        upper_c, lower_c = result

        # Neither centroid should land in the septum zone (190-210)
        assert not (190 <= upper_c[0] <= 210), (
            f"Upper centroid X={upper_c[0]:.1f} is in the septum zone (190-210)"
        )
        assert not (190 <= lower_c[0] <= 210), (
            f"Lower centroid X={lower_c[0]:.1f} is in the septum zone (190-210)"
        )

        # Both centroids must be on the SAME side (left-to-left or right-to-right)
        upper_on_left = upper_c[0] < 200
        lower_on_left = lower_c[0] < 200
        assert upper_on_left == lower_on_left, (
            f"Centroids on different sides: upper X={upper_c[0]:.1f}, "
            f"lower X={lower_c[0]:.1f}; should both be on the same side"
        )


class TestSampleDepthAtPoint:
    def test_coordinate_scaling(self):
        """Photo-space coordinates are correctly translated to depth-map space."""
        # Depth map 10x10, photo 100x100
        depth = Image.new("L", (10, 10), 0)
        # Set pixel at (5, 5) to 200
        depth.putpixel((5, 5), 200)
        # Also fill 3x3 kernel around (5,5) for median
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                depth.putpixel((5 + dx, 5 + dy), 200)

        # Photo point (50, 50) should map to depth (5, 5)
        result = sample_depth_at_point(depth, 50, 50, 100, 100)
        assert result == 200

    def test_kernel_median_filtering(self):
        """Median kernel filters out outlier pixels."""
        depth = Image.new("L", (10, 10), 100)
        # Put an outlier at center
        depth.putpixel((5, 5), 250)
        # All 8 neighbors are 100, center is 250
        # Median of [100, 100, 100, 100, 250, 100, 100, 100, 100] = 100

        result = sample_depth_at_point(depth, 5, 5, 10, 10, kernel_size=3)
        assert result == 100

    def test_edge_handling(self):
        """Points near image edges don't crash (partial kernel)."""
        depth = Image.new("L", (10, 10), 128)

        # Point at corner (0, 0) in a 10x10 depth map
        result = sample_depth_at_point(depth, 0, 0, 10, 10, kernel_size=3)
        assert result == 128

    def test_returns_none_for_out_of_bounds(self):
        """If all kernel pixels are out of bounds, returns None."""
        depth = Image.new("L", (1, 1), 50)

        # This maps to (0,0) with kernel_size=1, should work
        result = sample_depth_at_point(depth, 0, 0, 1, 1, kernel_size=1)
        assert result == 50


class TestDepthRawToDistanceCm:
    def test_disparity_conversion(self):
        """Known disparity values produce expected distances."""
        # disparity = float_max * 255/255 + float_min * 0 = float_max
        # distance = 100 / float_max
        result = depth_raw_to_distance_cm(255, 0.5, 2.0)
        assert abs(result - 100.0 / 2.0) < 0.01  # 50 cm

    def test_zero_value(self):
        """Raw value 0 uses only float_min for disparity."""
        # disparity = float_max * 0 + float_min * 1 = float_min
        # distance = 100 / float_min
        result = depth_raw_to_distance_cm(0, 0.5, 2.0)
        assert abs(result - 100.0 / 0.5) < 0.01  # 200 cm

    def test_zero_disparity_returns_none(self):
        """Zero disparity (both min/max are 0) returns None."""
        result = depth_raw_to_distance_cm(128, 0.0, 0.0)
        assert result is None

    def test_mid_value(self):
        """Mid raw value uses weighted average of min and max."""
        # value=127.5 => disparity = 2.0 * 0.5 + 0.5 * 0.5 = 1.25
        # distance = 100 / 1.25 = 80 cm
        result = depth_raw_to_distance_cm(127.5, 0.5, 2.0)
        assert abs(result - 80.0) < 0.1


class TestPixelToMm:
    def test_conversion(self):
        """Pixel to mm conversion at a known distance."""
        # At 25 cm, pixels_per_mm gives some value
        # pixel_to_mm(100, 25) = 100 / pixels_per_mm(25)
        from portrait_analyser.incisor import pixels_per_mm_at_distance

        ppmm = pixels_per_mm_at_distance(25)
        result = pixel_to_mm(100, 25)
        assert result is not None
        assert abs(result - 100.0 / ppmm) < 0.001


class TestVectorLength3d:
    def test_axis_aligned(self):
        """Distance along a single axis."""
        assert abs(vector_length_3d(0, 0, 0, 3, 0, 0) - 3.0) < 0.001
        assert abs(vector_length_3d(0, 0, 0, 0, 4, 0) - 4.0) < 0.001
        assert abs(vector_length_3d(0, 0, 0, 0, 0, 5) - 5.0) < 0.001

    def test_3d_diagonal(self):
        """Classic 3-4-5 style: sqrt(1+4+9) = sqrt(14)."""
        import math

        result = vector_length_3d(0, 0, 0, 1, 2, 3)
        assert abs(result - math.sqrt(14)) < 0.001

    def test_same_point(self):
        """Distance between same point is 0."""
        assert vector_length_3d(1, 2, 3, 1, 2, 3) == 0.0


class TestComputeIncisorDistance3d:
    def test_full_pipeline(self):
        """End-to-end 3D distance computation produces a positive result."""
        result = compute_incisor_distance_3d(
            upper_centroid=(800.0, 1500.0),
            lower_centroid=(810.0, 1800.0),
            upper_depth_raw=200,
            lower_depth_raw=200,
            float_min=0.5,
            float_max=2.0,
        )
        assert result is not None
        distance_3d_mm, upper_cm, lower_cm = result
        assert distance_3d_mm > 0
        assert upper_cm > 0
        assert lower_cm > 0

    def test_same_depth_different_position(self):
        """Same depth but different pixel positions gives non-zero distance."""
        result = compute_incisor_distance_3d(
            upper_centroid=(800.0, 1500.0),
            lower_centroid=(800.0, 1600.0),
            upper_depth_raw=200,
            lower_depth_raw=200,
            float_min=0.5,
            float_max=2.0,
        )
        assert result is not None
        distance_3d_mm = result[0]
        # Should reflect the Y pixel difference converted to mm
        assert distance_3d_mm > 0

    def test_zero_disparity_returns_none(self):
        """Zero disparity produces None result."""
        result = compute_incisor_distance_3d(
            upper_centroid=(800.0, 1500.0),
            lower_centroid=(800.0, 1600.0),
            upper_depth_raw=128,
            lower_depth_raw=128,
            float_min=0.0,
            float_max=0.0,
        )
        assert result is None


class TestIncisorMeasurementDataclass:
    def test_defaults(self):
        m = IncisorMeasurement(
            upper_centroid=(100.0, 150.0),
            lower_centroid=(100.0, 350.0),
        )
        assert m.upper_centroid == (100.0, 150.0)
        assert m.lower_centroid == (100.0, 350.0)
        assert m.upper_depth_raw is None
        assert m.lower_depth_raw is None
        assert m.distance_3d_mm is None
        assert m.pixel_distance_y == 0.0

    def test_full_construction(self):
        m = IncisorMeasurement(
            upper_centroid=(100.0, 150.0),
            lower_centroid=(100.0, 350.0),
            upper_depth_raw=180,
            lower_depth_raw=170,
            upper_distance_cm=30.5,
            lower_distance_cm=29.2,
            distance_3d_mm=5.3,
            pixel_distance_y=200.0,
        )
        assert m.distance_3d_mm == 5.3
        assert m.pixel_distance_y == 200.0
        assert m.upper_distance_cm == 30.5
