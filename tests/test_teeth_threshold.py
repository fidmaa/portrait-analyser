"""Adaptive teeth-matte threshold and bounding box.

Synthetic mattes only.  Their intensities mimic what Apple's semantic teeth
matte looks like on real open-mouth captures: a faint (~20-40) lip/mouth
contour halo, and incisors whose confidence ranges from saturated (255) down
to ~100 on weak captures.
"""

import numpy
import pytest
from PIL import Image

from portrait_analyser import face
from portrait_analyser.face import (
    find_bounding_box_teeth,
    find_incisor_centroids,
    find_incisor_distance_teeth,
    sample_depth_at_point,
    teeth_threshold,
)

WIDTH, HEIGHT = 600, 800
# Upper incisors (rows 300-329) and lower incisors (rows 480-509), x 250-349.
UPPER_ROWS = (300, 330)
LOWER_ROWS = (480, 510)
TEETH_COLS = (250, 350)


def _halo(arr, value=30):
    """Draw a faint elliptical mouth contour like Apple's lip halo."""
    yy, xx = numpy.mgrid[0:HEIGHT, 0:WIDTH]
    r = ((xx - 300) / 120.0) ** 2 + ((yy - 405) / 130.0) ** 2
    arr[(r > 0.85) & (r < 1.15)] = value
    return arr


def _open_mouth(upper=255, lower=255, halo=30):
    arr = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
    if halo:
        _halo(arr, halo)
    arr[UPPER_ROWS[0] : UPPER_ROWS[1], TEETH_COLS[0] : TEETH_COLS[1]] = upper
    arr[LOWER_ROWS[0] : LOWER_ROWS[1], TEETH_COLS[0] : TEETH_COLS[1]] = lower
    return Image.fromarray(arr)


def _legacy_bounding_box(teethmap, margin_x=100, margin_y=100, min_value=200):
    """The pre-adaptive pure-Python implementation, kept as a reference."""
    min_x = min_y = max_x = max_y = None
    for y in range(margin_y, teethmap.size[1] - margin_y):
        for x in range(margin_x, teethmap.size[0] - margin_x):
            if teethmap.getpixel((x, y)) > min_value:
                min_x = x if min_x is None else min(min_x, x)
                max_x = x if max_x is None else max(max_x, x)
                min_y = y if min_y is None else min(min_y, y)
                max_y = y if max_y is None else max(max_y, y)
    if max_y == teethmap.size[1] - margin_y - 1 or min_x is None:
        return None
    if max_y - min_y < 200:
        return None
    return (min_x, min_y, max_x - min_x, max_y - min_y)


class TestTeethThreshold:
    def test_half_of_robust_peak(self):
        img = _open_mouth(upper=160, lower=160)
        assert teeth_threshold(img) == 80

    def test_floor_applies_to_weak_mattes(self):
        img = _open_mouth(upper=100, lower=100)
        assert teeth_threshold(img) == face.TEETH_THRESHOLD_FLOOR

    def test_cap_applies(self):
        img = _open_mouth()
        assert teeth_threshold(img, peak_fraction=1.0) == face.TEETH_THRESHOLD_CAP

    def test_few_saturated_pixels_do_not_set_the_peak(self):
        img = _open_mouth(upper=120, lower=120)
        arr = numpy.array(img)
        arr[100:103, 100:103] = 255  # 9 saturated speckle pixels
        assert teeth_threshold(Image.fromarray(arr)) == teeth_threshold(img)

    def test_empty_matte(self):
        blank = Image.new("L", (WIDTH, HEIGHT), 0)
        assert teeth_threshold(blank) == face.TEETH_THRESHOLD_FLOOR

    def test_invalid_arguments(self):
        img = _open_mouth()
        with pytest.raises(ValueError):
            teeth_threshold(img, floor=210, cap=200)
        with pytest.raises(ValueError):
            teeth_threshold(img, peak_fraction=0)


class TestWeakMatteDetected:
    def test_weak_matte_is_missed_by_fixed_200_but_found_adaptively(self):
        img = _open_mouth(upper=110, lower=100)

        assert find_bounding_box_teeth(img, min_value=200) is None

        bbox = find_bounding_box_teeth(img)
        assert bbox == (
            TEETH_COLS[0],
            UPPER_ROWS[0],
            TEETH_COLS[1] - TEETH_COLS[0] - 1,
            LOWER_ROWS[1] - UPPER_ROWS[0] - 1,
        )

        centroids = find_incisor_centroids(img, bbox)
        assert centroids is not None
        upper, lower = centroids
        assert upper[1] == UPPER_ROWS[1] - 1
        assert lower[1] == LOWER_ROWS[0]

        legacy = find_incisor_distance_teeth(img, bbox)
        assert legacy is not None
        # The legacy walk starts at the (fractional) bbox mid-row.
        assert UPPER_ROWS[1] - 1 <= legacy[1] < UPPER_ROWS[1]
        assert LOWER_ROWS[0] <= legacy[3] < LOWER_ROWS[0] + 1

    def test_single_weak_arch_gets_a_bounding_box(self):
        arr = numpy.array(_open_mouth(upper=100, lower=0))
        bbox = find_bounding_box_teeth(Image.fromarray(arr))
        assert bbox is not None
        assert bbox[1] == UPPER_ROWS[0]
        assert bbox[3] == UPPER_ROWS[1] - UPPER_ROWS[0] - 1

    def test_default_depth_support_uses_adaptive_threshold(self):
        img = _open_mouth(upper=110, lower=100)
        depth = Image.new("L", (WIDTH // 10, HEIGHT // 10), 90)
        upper_y = UPPER_ROWS[1] - 1
        assert sample_depth_at_point(depth, 300, upper_y, WIDTH, HEIGHT) == 90
        assert (
            sample_depth_at_point(
                depth, 300, upper_y, WIDTH, HEIGHT, support_mask=img, inward_y=-1
            )
            == 90
        )
        # The historical fixed cut rejects every support pixel of a weak matte.
        assert (
            sample_depth_at_point(
                depth,
                300,
                upper_y,
                WIDTH,
                HEIGHT,
                support_mask=img,
                support_threshold=200,
                inward_y=-1,
            )
            is None
        )


class TestNoiseRejected:
    def test_noise_only_matte(self):
        rng = numpy.random.default_rng(0)
        arr = rng.integers(0, 19, size=(HEIGHT, WIDTH), dtype=numpy.uint8)
        img = Image.fromarray(arr)
        assert teeth_threshold(img) == face.TEETH_THRESHOLD_FLOOR
        assert find_bounding_box_teeth(img) is None

    def test_halo_only_matte(self):
        arr = _halo(numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8), 37)
        assert find_bounding_box_teeth(Image.fromarray(arr)) is None

    def test_faint_blob_like_teethless_capture_is_rejected(self):
        """Mimics a matte without visible teeth: halo plus a faint top blob.

        The blob peaks at 59, below the adaptive threshold's floor.
        """
        arr = _halo(numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8), 25)
        arr[300:312, 270:330] = 40
        arr[304:308, 290:303] = 59
        img = Image.fromarray(arr)
        assert teeth_threshold(img) == face.TEETH_THRESHOLD_FLOOR
        assert find_bounding_box_teeth(img) is None

    def test_speckles_do_not_inflate_the_bounding_box(self):
        arr = numpy.array(_open_mouth(upper=200, lower=200, halo=0))
        arr[100, 100] = 255
        arr[700, 500] = 255
        bbox = find_bounding_box_teeth(Image.fromarray(arr))
        assert bbox[0] == TEETH_COLS[0]
        assert bbox[1] == UPPER_ROWS[0]
        assert bbox[1] + bbox[3] == LOWER_ROWS[1] - 1


class TestExplicitThresholdsUnchanged:
    def test_explicit_bbox_arguments_match_legacy_implementation(self):
        img = _open_mouth(upper=230, lower=210, halo=150)
        expected = _legacy_bounding_box(img, 100, 100, 200)
        assert expected is not None
        assert find_bounding_box_teeth(img, 100, 100, 200) == expected

    def test_explicit_min_value_is_strict(self):
        img = _open_mouth(upper=200, lower=200, halo=0)
        assert find_bounding_box_teeth(img, min_value=200) is None
        assert find_bounding_box_teeth(img, min_value=199) is not None

    def test_explicit_bbox_rejects_teeth_touching_bottom_margin(self):
        arr = numpy.array(_open_mouth(halo=0))
        arr[LOWER_ROWS[0] : HEIGHT, TEETH_COLS[0] : TEETH_COLS[1]] = 255
        img = Image.fromarray(arr)
        assert _legacy_bounding_box(img) is None
        assert find_bounding_box_teeth(img, 100, 100, 200) is None
        assert find_bounding_box_teeth(img) is None

    def test_explicit_threshold_ignores_weaker_teeth(self):
        img = _open_mouth(upper=255, lower=150)
        bbox = (200, 280, 200, 250)
        assert find_incisor_centroids(img, bbox, threshold=200) is None
        assert find_incisor_distance_teeth(img, bbox, threshold=200) is None
        assert find_incisor_centroids(img, bbox, threshold=150) is not None
        assert find_incisor_distance_teeth(img, bbox, threshold=150) is not None
