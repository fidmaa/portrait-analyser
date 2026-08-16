"""Tests for the colour-based upper lip bite test (ULBT) measurement.

The synthetic fixtures paint an idealised mouth: a skin-coloured face, a
vermilion-coloured lower lip, and an upper-lip band that is either skin
(the incisors have covered the vermilion) or vermilion (they have not).
"""

import math

import numpy
import pytest
from PIL import Image, ImageDraw

from portrait_analyser.ulbt import (
    LOWER_LIP_INNER,
    LOWER_LIP_OUTER,
    UPPER_LIP_INNER,
    UPPER_LIP_OUTER,
    compute_ulbt_from_facemesh,
)

SKIN_RGB = (151, 119, 95)
LIP_RGB = (161, 91, 88)
TEETH_RGB = (225, 215, 200)

IMAGE_SIZE = (440, 340)
LEFT_X, RIGHT_X, MID_Y = 120.0, 320.0, 220.0


def _arc(indices, amplitude):
    """Landmark ring: a sine bump between the two mouth corners."""
    points = {}
    count = len(indices)
    for position, index in enumerate(indices):
        t = position / (count - 1)
        points[index] = (
            LEFT_X + (RIGHT_X - LEFT_X) * t,
            MID_Y - amplitude * math.sin(math.pi * t),
        )
    return points


def _synthetic_landmarks():
    """478 landmarks with only the lip rings meaningfully placed."""
    points = [(0.0, 0.0)] * 478
    mapping = {}
    mapping.update(_arc(UPPER_LIP_OUTER, 34.0))
    mapping.update(_arc(UPPER_LIP_INNER, 12.0))
    mapping.update(_arc(LOWER_LIP_INNER, -12.0))
    mapping.update(_arc(LOWER_LIP_OUTER, -42.0))
    for index, coordinates in mapping.items():
        points[index] = coordinates
    return tuple(points)


def _band_polygon(points, outer_indices, inner_indices):
    return [points[i] for i in outer_indices] + [points[i] for i in reversed(inner_indices)]


def _synthetic_photo(points, upper_band_colour):
    """Skin face, vermilion lower lip, upper-lip band painted as asked."""
    photo = Image.new("RGB", IMAGE_SIZE, SKIN_RGB)
    draw = ImageDraw.Draw(photo)
    draw.polygon(_band_polygon(points, LOWER_LIP_INNER, LOWER_LIP_OUTER), fill=LIP_RGB)
    draw.polygon(_band_polygon(points, UPPER_LIP_OUTER, UPPER_LIP_INNER), fill=upper_band_colour)

    # Mild noise so the reference medians are not a degenerate single value.
    generator = numpy.random.default_rng(seed=1)
    noisy = numpy.asarray(photo).astype(numpy.int16)
    noisy += generator.integers(-4, 5, size=noisy.shape, dtype=numpy.int16)
    return Image.fromarray(numpy.clip(noisy, 0, 255).astype(numpy.uint8))


def _synthetic_teethmap(points):
    """Bright blob sitting in the mouth aperture."""
    teethmap = Image.new("L", IMAGE_SIZE, 0)
    draw = ImageDraw.Draw(teethmap)
    draw.polygon(_band_polygon(points, UPPER_LIP_INNER, LOWER_LIP_INNER), fill=255)
    return teethmap


@pytest.fixture
def landmarks():
    return _synthetic_landmarks()


class TestComputeULBTFromFacemesh:
    def test_returns_none_with_too_few_landmarks(self, landmarks):
        photo = _synthetic_photo(landmarks, SKIN_RGB)
        measurement, debug = compute_ulbt_from_facemesh(photo, landmarks[:100])
        assert measurement is None
        assert debug is None

    def test_covered_vermilion_reads_as_skin(self, landmarks):
        """Incisors covering the upper lip: the band holds no vermilion."""
        photo = _synthetic_photo(landmarks, SKIN_RGB)
        measurement, _ = compute_ulbt_from_facemesh(photo, landmarks)
        assert measurement.lip_share < 0.05
        assert measurement.skin_share > 0.95
        assert measurement.bite_coverage > 0.95

    def test_visible_vermilion_reads_as_lip(self, landmarks):
        """Upper lip not bitten: the band is vermilion throughout."""
        photo = _synthetic_photo(landmarks, LIP_RGB)
        measurement, _ = compute_ulbt_from_facemesh(photo, landmarks)
        assert measurement.lip_share > 0.90
        assert measurement.bite_coverage < 0.10

    def test_bite_coverage_complements_lip_share(self, landmarks):
        photo = _synthetic_photo(landmarks, SKIN_RGB)
        measurement, _ = compute_ulbt_from_facemesh(photo, landmarks)
        assert measurement.bite_coverage == pytest.approx(1.0 - measurement.lip_share)

    def test_mouth_width_matches_corner_distance(self, landmarks):
        photo = _synthetic_photo(landmarks, SKIN_RGB)
        measurement, _ = compute_ulbt_from_facemesh(photo, landmarks)
        assert measurement.mouth_width_px == pytest.approx(RIGHT_X - LEFT_X)

    def test_teeth_share_is_none_without_teethmap(self, landmarks):
        photo = _synthetic_photo(landmarks, SKIN_RGB)
        measurement, _ = compute_ulbt_from_facemesh(photo, landmarks)
        assert measurement.teeth_share is None

    def test_teeth_reference_used_when_teethmap_supplied(self, landmarks):
        photo = _synthetic_photo(landmarks, SKIN_RGB)
        measurement, _ = compute_ulbt_from_facemesh(
            photo, landmarks, teethmap=_synthetic_teethmap(landmarks)
        )
        assert measurement.teeth_share is not None
        assert measurement.lip_share + measurement.skin_share + measurement.teeth_share == (
            pytest.approx(1.0)
        )

    def test_debug_carries_masks_and_swatches(self, landmarks):
        photo = _synthetic_photo(landmarks, SKIN_RGB)
        _, debug = compute_ulbt_from_facemesh(photo, landmarks)
        assert debug.band_pixels > 0
        assert debug.band_mask.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0])
        assert debug.band_mask.sum() == debug.band_pixels
        assert set(debug.reference_swatches) == {"skin", "lip"}
        assert debug.reference_pixels["skin"] > 0

    def test_shares_sum_to_one(self, landmarks):
        photo = _synthetic_photo(landmarks, LIP_RGB)
        measurement, _ = compute_ulbt_from_facemesh(photo, landmarks)
        assert measurement.lip_share + measurement.skin_share == pytest.approx(1.0)
