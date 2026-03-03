"""Tests for neck midpoint detection via MediaPipe Pose."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from portrait_analyser.pose import (
    FACE_FLATNESS_THRESHOLD,
    FACE_FLATNESS_THRESHOLD_POSE,
    NeckMidpoint,
    PortraitPose,
    detect_neck_midpoint,
)


# ---------------------------------------------------------------------------
# Helpers to build mock MediaPipe results
# ---------------------------------------------------------------------------


def _make_landmark(x, y, z=0.0, visibility=0.99):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _build_pose_result(landmark_dict):
    """Build a mock PoseLandmarker result from a dict of {index: landmark}.

    Fills all 33 landmark slots with defaults for any missing indices.
    """
    landmarks = [_make_landmark(0, 0) for _ in range(33)]
    for idx, lm in landmark_dict.items():
        landmarks[idx] = lm
    return SimpleNamespace(pose_landmarks=[landmarks])


def _run_detection(landmark_dict, image_size=(200, 400), ratio=0.35, min_visibility=0.5):
    """Run detect_neck_midpoint with mocked MediaPipe returning given landmarks.

    Returns (NeckMidpoint, ...) tuple or None.
    """
    result = _build_pose_result(landmark_dict)

    mock_mp = MagicMock()
    # Make mp.Image just return the mock_mp.Image(...) call result
    mock_mp.ImageFormat.SRGB = 1

    # Set up PoseLandmarker context manager
    mock_landmarker = MagicMock()
    mock_landmarker.detect.return_value = result
    mock_landmarker.__enter__ = MagicMock(return_value=mock_landmarker)
    mock_landmarker.__exit__ = MagicMock(return_value=False)
    mock_mp.tasks.vision.PoseLandmarker.create_from_options.return_value = mock_landmarker

    image = Image.new("RGB", image_size)

    with (
        patch("portrait_analyser.pose._get_model_path", return_value="/fake/model.task"),
        patch("portrait_analyser.pose._detect_chin_via_face_mesh", return_value=None),
        patch("portrait_analyser.pose.mp", mock_mp, create=True),
        patch.dict("sys.modules", {"mediapipe": mock_mp}),
    ):
        import portrait_analyser.pose as pose_module
        original_import = pose_module.__dict__.get("mp")

        # Temporarily replace mp in the module namespace
        pose_module.__dict__["mp"] = mock_mp
        try:
            return detect_neck_midpoint(
                image, interpolation_ratio=ratio, min_visibility=min_visibility,
            )
        finally:
            if original_import is not None:
                pose_module.__dict__["mp"] = original_import
            else:
                pose_module.__dict__.pop("mp", None)


def _run_detection_no_pose(image_size=(200, 400)):
    """Run detect_neck_midpoint with mocked MediaPipe returning no pose."""
    result = SimpleNamespace(pose_landmarks=[])

    mock_mp = MagicMock()
    mock_mp.ImageFormat.SRGB = 1
    mock_landmarker = MagicMock()
    mock_landmarker.detect.return_value = result
    mock_landmarker.__enter__ = MagicMock(return_value=mock_landmarker)
    mock_landmarker.__exit__ = MagicMock(return_value=False)
    mock_mp.tasks.vision.PoseLandmarker.create_from_options.return_value = mock_landmarker

    image = Image.new("RGB", image_size)

    import portrait_analyser.pose as pose_module
    original_mp = pose_module.__dict__.get("mp")
    pose_module.__dict__["mp"] = mock_mp
    try:
        with (
            patch("portrait_analyser.pose._get_model_path", return_value="/fake/model.task"),
            patch("portrait_analyser.pose._detect_chin_via_face_mesh", return_value=None),
            patch.dict("sys.modules", {"mediapipe": mock_mp}),
        ):
            return detect_neck_midpoint(image)
    finally:
        if original_mp is not None:
            pose_module.__dict__["mp"] = original_mp
        else:
            pose_module.__dict__.pop("mp", None)


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
            mouth_left=(80.0, 160.0),
            mouth_right=(120.0, 160.0),
            chin=(100.0, 180.0),
            left_shoulder_visibility=0.95,
            right_shoulder_visibility=0.93,
            nose_visibility=0.99,
            interpolation_ratio=0.35,
            neck_extended=False,
            face_flatness_ratio=1.1,
            pose=PortraitPose.NEUTRAL_NECK,
            mouth_open_ratio=None,
        )
        assert pt.x == 100.0
        assert pt.y == 200.0
        assert pt.interpolation_ratio == 0.35
        assert pt.face_flatness_ratio == 1.1

    def test_frozen(self):
        pt = NeckMidpoint(
            x=1, y=2,
            left_shoulder=(0, 0), right_shoulder=(0, 0), nose=(0, 0),
            mouth_left=(0, 0), mouth_right=(0, 0), chin=(0, 0),
            left_shoulder_visibility=0.9, right_shoulder_visibility=0.9,
            nose_visibility=0.9, interpolation_ratio=0.35,
            neck_extended=False, face_flatness_ratio=None,
            pose=PortraitPose.NEUTRAL_NECK, mouth_open_ratio=None,
        )
        with pytest.raises(AttributeError):
            pt.x = 999


# ---------------------------------------------------------------------------
# Interpolation math
# ---------------------------------------------------------------------------


class TestInterpolationMath:
    """Test neck midpoint interpolation with mocked MediaPipe."""

    IMAGE_W, IMAGE_H = 200, 400

    def test_ratio_zero_returns_shoulder_midpoint(self):
        """ratio=0 -> result sits exactly at shoulder midpoint."""
        result = _run_detection(
            {
                0: _make_landmark(0.5, 0.25),            # nose at (100, 100)
                2: _make_landmark(0.4, 0.20),             # left eye
                5: _make_landmark(0.6, 0.20),             # right eye
                9: _make_landmark(0.45, 0.30),            # mouth left
                10: _make_landmark(0.55, 0.30),           # mouth right
                11: _make_landmark(0.25, 0.75),           # left shoulder (50, 300)
                12: _make_landmark(0.75, 0.75),           # right shoulder (150, 300)
            },
            image_size=(self.IMAGE_W, self.IMAGE_H),
            ratio=0.0,
        )
        assert result is not None
        neck = result[0]
        assert pytest.approx(neck.x) == 100.0  # shoulder mid x
        assert pytest.approx(neck.y) == 300.0  # shoulder mid y

    def test_ratio_one_returns_nose(self):
        """ratio=1 -> result sits exactly at nose."""
        result = _run_detection(
            {
                0: _make_landmark(0.5, 0.25),
                2: _make_landmark(0.4, 0.20),
                5: _make_landmark(0.6, 0.20),
                9: _make_landmark(0.45, 0.30),
                10: _make_landmark(0.55, 0.30),
                11: _make_landmark(0.25, 0.75),
                12: _make_landmark(0.75, 0.75),
            },
            image_size=(self.IMAGE_W, self.IMAGE_H),
            ratio=1.0,
        )
        assert result is not None
        neck = result[0]
        assert pytest.approx(neck.x) == 100.0
        assert pytest.approx(neck.y) == 100.0  # nose y

    def test_ratio_035_default(self):
        """Default ratio 0.35 -> 35% from shoulder midpoint toward nose."""
        result = _run_detection(
            {
                0: _make_landmark(0.5, 0.25),             # (100, 100) px
                2: _make_landmark(0.4, 0.20),
                5: _make_landmark(0.6, 0.20),
                9: _make_landmark(0.45, 0.30),
                10: _make_landmark(0.55, 0.30),
                11: _make_landmark(0.25, 0.75),            # (50, 300) px
                12: _make_landmark(0.75, 0.75),            # (150, 300) px
            },
            image_size=(self.IMAGE_W, self.IMAGE_H),
            ratio=0.35,
        )
        assert result is not None
        neck = result[0]
        # shoulder mid = (100, 300), nose = (100, 100), diff_y = -200
        # neck_y = 300 + 0.35 * (-200) = 230
        assert pytest.approx(neck.x) == 100.0
        assert pytest.approx(neck.y) == 230.0
        assert neck.interpolation_ratio == 0.35

    def test_asymmetric_shoulders(self):
        """Shoulders at different heights -> midpoint is correct."""
        result = _run_detection(
            {
                0: _make_landmark(0.5, 0.1),               # (100, 40)
                2: _make_landmark(0.4, 0.05),
                5: _make_landmark(0.6, 0.05),
                9: _make_landmark(0.45, 0.15),
                10: _make_landmark(0.55, 0.15),
                11: _make_landmark(0.3, 0.6),              # (60, 240)
                12: _make_landmark(0.7, 0.8),              # (140, 320)
            },
            image_size=(self.IMAGE_W, self.IMAGE_H),
            ratio=0.0,
        )
        assert result is not None
        neck = result[0]
        assert pytest.approx(neck.x) == 100.0   # (60+140)/2
        assert pytest.approx(neck.y) == 280.0   # (240+320)/2


# ---------------------------------------------------------------------------
# Face-flattening algorithm
# ---------------------------------------------------------------------------


class TestFaceFlattening:
    """Test the face-flattening neck extension detection."""

    IMAGE_W, IMAGE_H = 200, 400

    def _standard_landmarks(self, eye_y=0.20, mouth_y=0.35, eye_vis=0.99):
        """Build landmarks with configurable eye/mouth Y for testing flatness.

        Eyes at x=0.4, 0.6 -> IPD = 0.2 * 200 = 40px (well above MIN_IPD_PIXELS).
        """
        return {
            0: _make_landmark(0.5, 0.25),                          # nose
            2: _make_landmark(0.4, eye_y, visibility=eye_vis),     # left eye
            5: _make_landmark(0.6, eye_y, visibility=eye_vis),     # right eye
            9: _make_landmark(0.45, mouth_y),                       # mouth left
            10: _make_landmark(0.55, mouth_y),                      # mouth right
            11: _make_landmark(0.25, 0.75),                         # left shoulder
            12: _make_landmark(0.75, 0.75),                         # right shoulder
        }

    def test_normal_face_not_extended(self):
        """Normal upright face (ratio ~1.1) -> neck_extended=False."""
        # eye_y=0.20, mouth_y=0.35 -> vertical = (0.35-0.20)*400 = 60px
        # IPD = (0.6-0.4)*200 = 40px -> ratio = 60/40 = 1.5
        result = _run_detection(
            self._standard_landmarks(eye_y=0.20, mouth_y=0.35),
            image_size=(self.IMAGE_W, self.IMAGE_H),
        )
        assert result is not None
        neck = result[0]
        assert neck.neck_extended is False
        assert neck.face_flatness_ratio is not None
        assert neck.face_flatness_ratio > FACE_FLATNESS_THRESHOLD

    def test_extended_neck_detected(self):
        """Extended neck (ratio ~0.7) -> neck_extended=True."""
        # eye_y=0.25, mouth_y=0.32 -> vertical = (0.32-0.25)*400 = 28px
        # IPD = 40px -> ratio = 28/40 = 0.7
        result = _run_detection(
            self._standard_landmarks(eye_y=0.25, mouth_y=0.32),
            image_size=(self.IMAGE_W, self.IMAGE_H),
        )
        assert result is not None
        neck = result[0]
        assert neck.neck_extended is True
        assert neck.face_flatness_ratio is not None
        assert neck.face_flatness_ratio < FACE_FLATNESS_THRESHOLD

    def test_low_ipd_guard(self):
        """IPD below minimum -> neck_extended defaults to False."""
        # Eyes very close together: x=0.49, 0.51 -> IPD = 0.02*200 = 4px < 20
        landmarks = self._standard_landmarks()
        landmarks[2] = _make_landmark(0.49, 0.20)
        landmarks[5] = _make_landmark(0.51, 0.20)

        result = _run_detection(
            landmarks,
            image_size=(self.IMAGE_W, self.IMAGE_H),
        )
        assert result is not None
        neck = result[0]
        assert neck.neck_extended is False
        assert neck.face_flatness_ratio is None

    def test_low_eye_visibility_still_computes_ratio(self):
        """Low eye visibility doesn't block face-flattening (no vis guard)."""
        result = _run_detection(
            self._standard_landmarks(eye_vis=0.1),
            image_size=(self.IMAGE_W, self.IMAGE_H),
            min_visibility=0.5,
        )
        assert result is not None
        neck = result[0]
        assert neck.neck_extended is False
        # Face-flattening ratio is computed regardless of eye visibility
        assert neck.face_flatness_ratio is not None

    def test_lateral_tilt_guard(self):
        """Lateral head tilt > 30% of IPD -> neck_extended defaults to False."""
        # left eye at y=0.20, right eye at y=0.25 -> tilt = 0.05*400 = 20px
        # IPD = 40px, 30% of IPD = 12px, 20 > 12 -> guard triggers
        landmarks = self._standard_landmarks()
        landmarks[2] = _make_landmark(0.4, 0.20)
        landmarks[5] = _make_landmark(0.6, 0.25)

        result = _run_detection(
            landmarks,
            image_size=(self.IMAGE_W, self.IMAGE_H),
        )
        assert result is not None
        neck = result[0]
        assert neck.neck_extended is False
        assert neck.face_flatness_ratio is None

    def test_ratio_above_threshold_not_extended(self):
        """Ratio above Pose threshold -> neck_extended=False.

        With Pose-only landmarks (FaceMesh mocked to None), the more lenient
        FACE_FLATNESS_THRESHOLD_POSE (1.2) is used instead of 0.85.
        """
        # eye_y=0.20, mouth_y=0.35 -> vertical = (0.35-0.20)*400 = 60px
        # IPD = 40px -> ratio = 60/40 = 1.5 > 1.2
        result = _run_detection(
            self._standard_landmarks(eye_y=0.20, mouth_y=0.35),
            image_size=(self.IMAGE_W, self.IMAGE_H),
        )
        assert result is not None
        neck = result[0]
        assert neck.neck_extended is False
        assert neck.face_flatness_ratio is not None
        assert neck.face_flatness_ratio > FACE_FLATNESS_THRESHOLD_POSE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_returns_none_neck_when_no_pose_detected(self):
        result = _run_detection_no_pose()
        neck, pose_debug, face_mesh_debug = result
        assert neck is None
        assert pose_debug is None

    def test_no_shoulders_when_left_shoulder_visibility_low(self):
        """Low left shoulder visibility -> neck returned but x/y are None."""
        result = _run_detection(
            {
                0: _make_landmark(0.5, 0.25, visibility=0.99),
                2: _make_landmark(0.4, 0.20),
                5: _make_landmark(0.6, 0.20),
                9: _make_landmark(0.45, 0.30),
                10: _make_landmark(0.55, 0.30),
                11: _make_landmark(0.25, 0.75, visibility=0.1),  # low
                12: _make_landmark(0.75, 0.75, visibility=0.99),
            },
            min_visibility=0.5,
        )
        neck, pose_debug, _ = result
        assert neck is not None
        assert pose_debug is not None
        assert neck.x is None
        assert neck.y is None
        assert neck.left_shoulder is None
        assert neck.interpolation_ratio is None

    def test_no_shoulders_when_right_shoulder_visibility_low(self):
        """Low right shoulder visibility -> neck returned but x/y are None."""
        result = _run_detection(
            {
                0: _make_landmark(0.5, 0.25, visibility=0.99),
                2: _make_landmark(0.4, 0.20),
                5: _make_landmark(0.6, 0.20),
                9: _make_landmark(0.45, 0.30),
                10: _make_landmark(0.55, 0.30),
                11: _make_landmark(0.25, 0.75, visibility=0.99),
                12: _make_landmark(0.75, 0.75, visibility=0.1),  # low
            },
            min_visibility=0.5,
        )
        neck, pose_debug, _ = result
        assert neck is not None
        assert pose_debug is not None
        assert neck.x is None
        assert neck.y is None
        assert neck.right_shoulder is None
        assert neck.interpolation_ratio is None


# ---------------------------------------------------------------------------
# ImportError handling
# ---------------------------------------------------------------------------


class TestImportError:
    def test_raises_import_error_with_helpful_message(self):
        with patch("portrait_analyser.pose._get_model_path") as mock_path:
            mock_path.side_effect = ImportError(
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
    def test_blank_image_returns_none_neck(self):
        """A blank white image should not detect any pose."""
        image = Image.new("RGB", (640, 480), color=(255, 255, 255))
        neck, pose_debug, _ = detect_neck_midpoint(image)
        assert neck is None
        assert pose_debug is None
