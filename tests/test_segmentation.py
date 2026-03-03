"""Tests for segmentation-based neck/chin detection."""

import numpy as np

from portrait_analyser.pose import NeckMidpoint, PortraitPose
from portrait_analyser.segmentation import (
    _compute_width_profile,
    _detect_from_mask,
)


def _make_person_mask(
    h: int = 400,
    w: int = 300,
    head_width: int = 80,
    neck_width: int = 40,
    torso_width: int = 120,
    head_top: int = 20,
    head_bottom: int = 100,
    neck_bottom: int = 140,
    torso_bottom: int = 350,
) -> np.ndarray:
    """Create a synthetic person silhouette mask.

    The mask has three distinct regions: head (wide), neck (narrow),
    and torso (widest). This mimics a real person silhouette for testing
    the width-profile algorithm.
    """
    mask = np.zeros((h, w), dtype=bool)
    cx = w // 2

    # Head region
    for y in range(head_top, head_bottom):
        half = head_width // 2
        mask[y, cx - half : cx + half] = True

    # Neck region (narrower)
    for y in range(head_bottom, neck_bottom):
        half = neck_width // 2
        mask[y, cx - half : cx + half] = True

    # Torso region (widest)
    for y in range(neck_bottom, torso_bottom):
        half = torso_width // 2
        mask[y, cx - half : cx + half] = True

    return mask


class TestComputeWidthProfile:
    def test_basic_rectangle(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[3:7, 5:15] = True  # 4 rows, width 10, from x=5 to x=14

        widths, lefts, rights = _compute_width_profile(mask)

        assert len(widths) == 10
        assert widths[3] == 9  # 14 - 5
        assert widths[0] == 0  # empty row
        assert lefts[3] == 5
        assert rights[3] == 14

    def test_empty_mask(self):
        mask = np.zeros((10, 20), dtype=bool)
        widths, lefts, rights = _compute_width_profile(mask)

        assert np.all(widths == 0)
        assert np.all(lefts == 0)
        assert np.all(rights == 0)


class TestDetectFromMask:
    def test_person_with_neck(self):
        """A person silhouette with distinct head-neck-torso should detect neck."""
        mask = _make_person_mask()
        h, w = mask.shape

        result, debug = _detect_from_mask(mask, h, w, smoothing_window=1)

        assert result is not None
        assert isinstance(result, NeckMidpoint)
        assert result.pose == PortraitPose.EXTENDED_NECK
        assert result.neck_extended is True
        assert result.x is not None
        assert result.y is not None

        # Neck should be detected in the neck region (rows 100-140)
        assert 90 <= result.y <= 150, f"neck_y={result.y} not in expected range"

        # Midline should be near center
        assert abs(result.x - w / 2) < 10, f"midline_x={result.x} not near center"

        # Chin should be above neck
        assert result.chin[1] < result.y, "chin should be above neck"

        # Debug should be populated
        assert debug is not None
        assert debug.neck_y is not None
        assert debug.chin_y is not None

    def test_empty_mask_returns_none(self):
        """An empty mask should return None."""
        mask = np.zeros((400, 300), dtype=bool)

        result, debug = _detect_from_mask(mask, 400, 300)

        assert result is None
        assert debug is not None
        assert debug.neck_y is None

    def test_too_small_mask_returns_none(self):
        """A tiny mask (too few pixels) should return None."""
        mask = np.zeros((400, 300), dtype=bool)
        # Just a few pixels — less than 1% of image
        mask[200, 150] = True

        result, debug = _detect_from_mask(mask, 400, 300)

        assert result is None

    def test_short_person_returns_none(self):
        """A person spanning < 50 pixels vertically should return None."""
        mask = np.zeros((400, 300), dtype=bool)
        # 30 rows tall — below minimum
        mask[100:130, 120:180] = True

        result, debug = _detect_from_mask(mask, 400, 300)

        assert result is None

    def test_shoulder_fields_are_none(self):
        """Segmentation-based detection should not populate shoulder fields."""
        mask = _make_person_mask()
        h, w = mask.shape

        result, debug = _detect_from_mask(mask, h, w, smoothing_window=1)

        assert result is not None
        assert result.left_shoulder is None
        assert result.right_shoulder is None
        assert result.left_shoulder_visibility is None
        assert result.right_shoulder_visibility is None
        assert result.nose_visibility is None
        assert result.interpolation_ratio is None

    def test_face_landmark_estimates_present(self):
        """Nose and mouth estimates should be populated for GUI cursor placement."""
        mask = _make_person_mask()
        h, w = mask.shape

        result, _ = _detect_from_mask(mask, h, w, smoothing_window=1)

        assert result is not None
        assert result.nose is not None
        assert result.mouth_left is not None
        assert result.mouth_right is not None
        # Nose should be above chin
        assert result.nose[1] < result.chin[1]
        # Mouth should be above chin
        assert result.mouth_left[1] < result.chin[1]

    def test_narrow_neck_detected_correctly(self):
        """Verify neck is found at the narrowest point of the silhouette."""
        # Create a mask with an extremely narrow neck
        mask = _make_person_mask(neck_width=20, head_width=100, torso_width=150)
        h, w = mask.shape

        result, debug = _detect_from_mask(mask, h, w, smoothing_window=1)

        assert result is not None
        # The neck_y should be in the neck region (100-140)
        assert 95 <= debug.neck_y <= 145

    def test_debug_width_profile_populated(self):
        """SegmentationDebug should contain the width profile array."""
        mask = _make_person_mask()
        h, w = mask.shape

        _, debug = _detect_from_mask(mask, h, w, smoothing_window=1)

        assert debug is not None
        assert len(debug.width_profile) > 0
        assert debug.roi_top >= 0

    def test_face_flatness_ratio_is_none(self):
        """Segmentation-based detection should set face_flatness_ratio to None."""
        mask = _make_person_mask()
        h, w = mask.shape

        result, _ = _detect_from_mask(mask, h, w, smoothing_window=1)

        assert result is not None
        assert result.face_flatness_ratio is None
        assert result.mouth_open_ratio is None
