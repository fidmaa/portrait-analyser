"""Tests for neck midpoint detection via MediaPipe Pose."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from portrait_analyser.pose import NeckMidpoint, detect_neck_midpoint


# ---------------------------------------------------------------------------
# Helpers to build mock MediaPipe results
# ---------------------------------------------------------------------------


def _make_landmark(x, y, z=0.0, visibility=0.99):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _make_mock_mp(landmarks=None):
    """Build a mock mediapipe module with given landmarks (or None for no detection)."""
    mock_mp = MagicMock()

    # PoseLandmark enum indices
    mock_mp.solutions.pose.PoseLandmark.NOSE = 0
    mock_mp.solutions.pose.PoseLandmark.LEFT_SHOULDER = 11
    mock_mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER = 12

    # Pose context manager
    pose_instance = MagicMock()
    results = SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=landmarks) if landmarks else None,
    )
    pose_instance.process.return_value = results
    pose_instance.__enter__ = MagicMock(return_value=pose_instance)
    pose_instance.__exit__ = MagicMock(return_value=False)
    mock_mp.solutions.pose.Pose.return_value = pose_instance

    return mock_mp


# ---------------------------------------------------------------------------
# NeckMidpoint dataclass
# ---------------------------------------------------------------------------


class TestNeckMidpointDataclass:
    def test_construction(self):
        pt = NeckMidpoint(
            x=100.0,
            y=200.0,
            left_shoulder=(50.0, 300.0),
            right_shoulder=(150.0, 300.0),
            nose=(100.0, 100.0),
            left_shoulder_visibility=0.95,
            right_shoulder_visibility=0.93,
            nose_visibility=0.99,
            interpolation_ratio=0.35,
        )
        assert pt.x == 100.0
        assert pt.y == 200.0
        assert pt.interpolation_ratio == 0.35

    def test_frozen(self):
        pt = NeckMidpoint(
            x=1, y=2,
            left_shoulder=(0, 0), right_shoulder=(0, 0), nose=(0, 0),
            left_shoulder_visibility=0.9, right_shoulder_visibility=0.9,
            nose_visibility=0.9, interpolation_ratio=0.35,
        )
        with pytest.raises(AttributeError):
            pt.x = 999


# ---------------------------------------------------------------------------
# Interpolation math
# ---------------------------------------------------------------------------


class TestInterpolationMath:
    """Test neck midpoint interpolation with mocked MediaPipe."""

    IMAGE_W, IMAGE_H = 200, 400

    def _run_with_landmarks(self, nose, left_shoulder, right_shoulder, ratio=0.35):
        landmarks = {i: _make_landmark(0, 0) for i in range(33)}
        landmarks[0] = nose
        landmarks[11] = left_shoulder
        landmarks[12] = right_shoulder

        mock_mp = _make_mock_mp(landmarks)
        image = Image.new("RGB", (self.IMAGE_W, self.IMAGE_H))

        with patch.dict(sys.modules, {"mediapipe": mock_mp}):
            with patch("portrait_analyser.pose._import_mediapipe", return_value=mock_mp):
                return detect_neck_midpoint(image, interpolation_ratio=ratio)

    def test_ratio_zero_returns_shoulder_midpoint(self):
        """ratio=0 → result sits exactly at shoulder midpoint."""
        result = self._run_with_landmarks(
            nose=_make_landmark(0.5, 0.25),           # (100, 100) px
            left_shoulder=_make_landmark(0.25, 0.75),  # (50, 300) px
            right_shoulder=_make_landmark(0.75, 0.75), # (150, 300) px
            ratio=0.0,
        )
        assert result is not None
        assert pytest.approx(result.x) == 100.0  # shoulder mid x
        assert pytest.approx(result.y) == 300.0  # shoulder mid y

    def test_ratio_one_returns_nose(self):
        """ratio=1 → result sits exactly at nose."""
        result = self._run_with_landmarks(
            nose=_make_landmark(0.5, 0.25),
            left_shoulder=_make_landmark(0.25, 0.75),
            right_shoulder=_make_landmark(0.75, 0.75),
            ratio=1.0,
        )
        assert result is not None
        assert pytest.approx(result.x) == 100.0
        assert pytest.approx(result.y) == 100.0  # nose y

    def test_ratio_035_default(self):
        """Default ratio 0.35 → 35% from shoulder midpoint toward nose."""
        result = self._run_with_landmarks(
            nose=_make_landmark(0.5, 0.25),            # (100, 100) px
            left_shoulder=_make_landmark(0.25, 0.75),   # (50, 300) px
            right_shoulder=_make_landmark(0.75, 0.75),  # (150, 300) px
            ratio=0.35,
        )
        assert result is not None
        # shoulder mid = (100, 300), nose = (100, 100), diff_y = -200
        # neck_y = 300 + 0.35 * (-200) = 230
        assert pytest.approx(result.x) == 100.0
        assert pytest.approx(result.y) == 230.0
        assert result.interpolation_ratio == 0.35

    def test_asymmetric_shoulders(self):
        """Shoulders at different heights → midpoint is correct."""
        result = self._run_with_landmarks(
            nose=_make_landmark(0.5, 0.1),              # (100, 40)
            left_shoulder=_make_landmark(0.3, 0.6),     # (60, 240)
            right_shoulder=_make_landmark(0.7, 0.8),    # (140, 320)
            ratio=0.0,
        )
        assert result is not None
        assert pytest.approx(result.x) == 100.0   # (60+140)/2
        assert pytest.approx(result.y) == 280.0   # (240+320)/2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_returns_none_when_no_pose_detected(self):
        mock_mp = _make_mock_mp(landmarks=None)
        image = Image.new("RGB", (200, 400))

        with patch.dict(sys.modules, {"mediapipe": mock_mp}):
            with patch("portrait_analyser.pose._import_mediapipe", return_value=mock_mp):
                result = detect_neck_midpoint(image)

        assert result is None

    def test_returns_none_when_left_shoulder_visibility_low(self):
        landmarks = {i: _make_landmark(0, 0) for i in range(33)}
        landmarks[0] = _make_landmark(0.5, 0.25, visibility=0.99)
        landmarks[11] = _make_landmark(0.25, 0.75, visibility=0.1)  # low
        landmarks[12] = _make_landmark(0.75, 0.75, visibility=0.99)

        mock_mp = _make_mock_mp(landmarks)
        image = Image.new("RGB", (200, 400))

        with patch.dict(sys.modules, {"mediapipe": mock_mp}):
            with patch("portrait_analyser.pose._import_mediapipe", return_value=mock_mp):
                result = detect_neck_midpoint(image, min_visibility=0.5)

        assert result is None

    def test_returns_none_when_right_shoulder_visibility_low(self):
        landmarks = {i: _make_landmark(0, 0) for i in range(33)}
        landmarks[0] = _make_landmark(0.5, 0.25, visibility=0.99)
        landmarks[11] = _make_landmark(0.25, 0.75, visibility=0.99)
        landmarks[12] = _make_landmark(0.75, 0.75, visibility=0.1)  # low

        mock_mp = _make_mock_mp(landmarks)
        image = Image.new("RGB", (200, 400))

        with patch.dict(sys.modules, {"mediapipe": mock_mp}):
            with patch("portrait_analyser.pose._import_mediapipe", return_value=mock_mp):
                result = detect_neck_midpoint(image, min_visibility=0.5)

        assert result is None


# ---------------------------------------------------------------------------
# ImportError handling
# ---------------------------------------------------------------------------


class TestImportError:
    def test_raises_import_error_with_helpful_message(self):
        with patch("portrait_analyser.pose._import_mediapipe") as mock_import:
            mock_import.side_effect = ImportError(
                "mediapipe is required for pose estimation. "
                "Install it with: pip install portrait-analyser[pose]"
            )
            image = Image.new("RGB", (200, 400))
            with pytest.raises(ImportError, match="portrait-analyser\\[pose\\]"):
                detect_neck_midpoint(image)


# ---------------------------------------------------------------------------
# Integration test (skipped if mediapipe not installed)
# ---------------------------------------------------------------------------

try:
    import mediapipe  # noqa: F401

    _has_mediapipe = True
except ImportError:
    _has_mediapipe = False


@pytest.mark.skipif(not _has_mediapipe, reason="mediapipe not installed")
class TestIntegration:
    def test_blank_image_returns_none(self):
        """A blank white image should not detect any pose."""
        image = Image.new("RGB", (640, 480), color=(255, 255, 255))
        result = detect_neck_midpoint(image)
        assert result is None
