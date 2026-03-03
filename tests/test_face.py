"""Tests for face/eye detection via MediaPipe Face Detector."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from portrait_analyser.exceptions import MultipleFacesDetected, NoFacesDetected
from portrait_analyser.face import (
    Eye,
    Face,
    Rectangle,
    detect_eyes,
    get_face_parameters,
)


# ---------------------------------------------------------------------------
# Helpers to build mock MediaPipe detection results
# ---------------------------------------------------------------------------


def _make_keypoint(x, y):
    """Normalized keypoint (0..1)."""
    return SimpleNamespace(x=x, y=y)


def _make_bounding_box(origin_x, origin_y, width, height):
    return SimpleNamespace(
        origin_x=origin_x, origin_y=origin_y, width=width, height=height,
    )


def _make_detection(bbox, keypoints=None):
    return SimpleNamespace(bounding_box=bbox, keypoints=keypoints)


def _make_detection_result(detections):
    return SimpleNamespace(detections=detections)


def _patch_mediapipe(detection_result):
    """Return a mock mediapipe module that yields the given detection result."""
    mock_mp = MagicMock()

    # mp.Image constructor — just store data
    mock_mp.Image.return_value = MagicMock()
    mock_mp.ImageFormat.SRGB = "SRGB"

    # FaceDetector context manager
    detector_instance = MagicMock()
    detector_instance.detect.return_value = detection_result
    detector_instance.__enter__ = MagicMock(return_value=detector_instance)
    detector_instance.__exit__ = MagicMock(return_value=False)

    mock_mp.tasks.vision.FaceDetector.create_from_options.return_value = (
        detector_instance
    )

    return mock_mp


# ---------------------------------------------------------------------------
# Mocked unit tests — get_face_parameters
# ---------------------------------------------------------------------------


class TestGetFaceParametersMocked:
    IMAGE_W, IMAGE_H = 400, 600

    def _run(self, detections):
        result = _make_detection_result(detections)
        mock_mp = _patch_mediapipe(result)
        image = Image.new("RGB", (self.IMAGE_W, self.IMAGE_H))

        with patch("portrait_analyser.face._get_face_model_path", return_value="/fake"):
            with patch.dict("sys.modules", {"mediapipe": mock_mp}):
                # Re-import to pick up the mock
                import importlib
                import portrait_analyser.face as face_mod
                importlib.reload(face_mod)
                try:
                    return face_mod.get_face_parameters(image)
                finally:
                    importlib.reload(face_mod)

    def _run_simple(self, detections):
        """Run with direct patching (no reload needed for lazy import)."""
        result = _make_detection_result(detections)
        mock_mp = _patch_mediapipe(result)
        image = Image.new("RGB", (self.IMAGE_W, self.IMAGE_H))

        with patch("portrait_analyser.face._get_face_model_path", return_value="/fake"):
            with patch("portrait_analyser.face.mp", mock_mp, create=True):
                # Since mediapipe is imported lazily inside get_face_parameters,
                # we need to mock the import itself
                import sys
                with patch.dict(sys.modules, {"mediapipe": mock_mp}):
                    return get_face_parameters(image)

    def test_single_face_returns_face_object(self):
        det = _make_detection(
            bbox=_make_bounding_box(50, 80, 200, 250),
            keypoints=[
                _make_keypoint(0.30, 0.25),  # left eye
                _make_keypoint(0.55, 0.25),  # right eye
                _make_keypoint(0.42, 0.35),  # nose tip
                _make_keypoint(0.42, 0.45),  # mouth center
                _make_keypoint(0.15, 0.30),  # left ear
                _make_keypoint(0.70, 0.30),  # right ear
            ],
        )
        face = self._run_simple([det])
        assert isinstance(face, Face)
        assert face.x == 50
        assert face.y == 80
        assert face.width == 200
        assert face.height == 250
        assert len(face.eyes) == 2
        for eye in face.eyes:
            assert isinstance(eye, Eye)
            assert eye.face is face

    def test_no_detection_raises_no_faces(self):
        with pytest.raises(NoFacesDetected):
            self._run_simple([])

    def test_multiple_detections_raises_multiple_faces(self):
        det = _make_detection(bbox=_make_bounding_box(50, 80, 100, 100))
        with pytest.raises(MultipleFacesDetected):
            self._run_simple([det, det])

    def test_eye_center_coordinates(self):
        """Eye center_x/center_y should be face-relative from keypoint positions."""
        det = _make_detection(
            bbox=_make_bounding_box(100, 100, 200, 200),
            keypoints=[
                _make_keypoint(0.375, 0.333),  # left eye at (150, 200) abs
                _make_keypoint(0.625, 0.333),  # right eye at (250, 200) abs
                _make_keypoint(0.5, 0.5),
                _make_keypoint(0.5, 0.6),
                _make_keypoint(0.25, 0.4),
                _make_keypoint(0.75, 0.4),
            ],
        )
        face = self._run_simple([det])
        # Left eye abs=(150,200), face at (100,100), relative=(50,100)
        # eye_w=200*0.15=30, eye_h=200*0.08=16
        # eye box: (50-15, 100-8, 30, 16) = (35, 92, 30, 16)
        # center: (35+15, 92+8) = (50, 100)
        left_eye = face.eyes[0]
        assert abs(left_eye.center_x - 50) < 2
        assert abs(left_eye.center_y - 100) < 2


# ---------------------------------------------------------------------------
# Mocked unit tests — detect_eyes
# ---------------------------------------------------------------------------


class TestDetectEyesMocked:
    def test_returns_rectangles_for_detected_face(self):
        det = _make_detection(
            bbox=_make_bounding_box(50, 50, 200, 200),
            keypoints=[
                _make_keypoint(0.3, 0.3),  # left eye
                _make_keypoint(0.6, 0.3),  # right eye
                _make_keypoint(0.45, 0.5),
                _make_keypoint(0.45, 0.6),
                _make_keypoint(0.15, 0.4),
                _make_keypoint(0.75, 0.4),
            ],
        )
        result = _make_detection_result([det])
        mock_mp = _patch_mediapipe(result)
        image = Image.new("RGB", (400, 400))

        import sys
        with patch("portrait_analyser.face._get_face_model_path", return_value="/fake"):
            with patch.dict(sys.modules, {"mediapipe": mock_mp}):
                eyes = detect_eyes(image)

        assert len(eyes) == 2
        for eye in eyes:
            assert isinstance(eye, Rectangle)

    def test_returns_empty_for_no_detection(self):
        result = _make_detection_result([])
        mock_mp = _patch_mediapipe(result)
        image = Image.new("RGB", (400, 400))

        import sys
        with patch("portrait_analyser.face._get_face_model_path", return_value="/fake"):
            with patch.dict(sys.modules, {"mediapipe": mock_mp}):
                eyes = detect_eyes(image)

        assert eyes == []


# ---------------------------------------------------------------------------
# Face / Eye class tests (no MediaPipe needed)
# ---------------------------------------------------------------------------


class TestFaceClass:
    def test_face_with_eyes_parameter(self):
        image = Image.new("RGB", (400, 400))
        eye1 = Eye(None, 10, 20, 30, 16)
        eye2 = Eye(None, 60, 22, 30, 16)
        face = Face(image, 50, 50, 200, 200, eyes=[eye1, eye2])
        assert len(face.eyes) == 2
        assert face.eyes[0] is eye1
        assert face.eyes[1] is eye2

    def test_face_without_eyes_gets_empty_list(self):
        image = Image.new("RGB", (400, 400))
        face = Face(image, 50, 50, 200, 200)
        assert face.eyes == []

    def test_get_image_for_analysis_gray(self):
        image = Image.new("RGB", (400, 400), color=(100, 150, 200))
        face = Face(image, 10, 10, 50, 50)
        arr = face.get_image_for_analysis(mode="gray")
        assert arr.ndim == 2  # grayscale
        assert arr.shape == (49, 49)  # height-1, width-1

    def test_get_image_for_analysis_rgb(self):
        image = Image.new("RGB", (400, 400), color=(100, 150, 200))
        face = Face(image, 10, 10, 50, 50)
        arr = face.get_image_for_analysis(mode="rgb")
        assert arr.ndim == 3
        assert arr.shape == (49, 49, 3)

    def test_calculate_percentage_of_image(self):
        image = Image.new("RGB", (400, 400))
        face = Face(image, 0, 0, 200, 100)
        pw, ph = face.calculate_percentage_of_image()
        assert pw == 0.5
        assert ph == 0.25


# ---------------------------------------------------------------------------
# Integration tests (require actual HEIC files + mediapipe installed)
# ---------------------------------------------------------------------------


try:
    import mediapipe  # noqa: F401

    _has_mediapipe = True
except ImportError:
    _has_mediapipe = False


@pytest.mark.skipif(not _has_mediapipe, reason="mediapipe not installed")
class TestIntegration:
    def test_get_face_parameters_no_face(self, heic_image_path):
        from portrait_analyser.ios import load_image

        result = load_image(str(heic_image_path))
        with pytest.raises(NoFacesDetected):
            get_face_parameters(result.photo)

    def test_get_face_parameters_face(self, heic_face_image_path):
        from portrait_analyser.ios import load_image

        result = load_image(str(heic_face_image_path))
        res = get_face_parameters(result.photo)
        assert res
        assert isinstance(res, Face)
        assert len(res.eyes) == 2
