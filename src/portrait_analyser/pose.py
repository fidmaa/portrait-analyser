"""Neck midpoint detection via MediaPipe Pose estimation.

Requires the optional ``pose`` extra::

    pip install portrait-analyser[pose]
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/"
    "pose_landmarker_heavy.task"
)
_CACHE_DIR = Path.home() / ".cache" / "portrait-analyser"
_MODEL_FILENAME = "pose_landmarker_heavy.task"


@dataclass(frozen=True)
class MediaPipeDebug:
    """All raw MediaPipe pose landmarks for debug visualization."""

    landmarks: tuple[tuple[float, float], ...]  # 33 (x, y) in photo pixels


@dataclass(frozen=True)
class NeckMidpoint:
    """Detected neck midpoint with supporting landmarks.

    All coordinates are in absolute photo-space pixels.
    """

    x: float
    y: float
    left_shoulder: tuple[float, float]
    right_shoulder: tuple[float, float]
    nose: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]
    left_shoulder_visibility: float
    right_shoulder_visibility: float
    nose_visibility: float
    interpolation_ratio: float


def _get_model_path() -> str:
    """Return path to the PoseLandmarker .task model, downloading if needed.

    Raises:
        ImportError: If mediapipe is not installed.
    """
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        raise ImportError(
            "mediapipe is required for pose estimation. "
            "Install it with: pip install portrait-analyser[pose]"
        ) from None

    model_path = _CACHE_DIR / _MODEL_FILENAME
    if not model_path.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = model_path.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(_MODEL_URL, tmp_path)
            os.replace(tmp_path, model_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    return str(model_path)


def detect_neck_midpoint(
    image: Image.Image,
    interpolation_ratio: float = 0.35,
    min_detection_confidence: float = 0.5,
    min_visibility: float = 0.5,
) -> tuple[NeckMidpoint, MediaPipeDebug] | None:
    """Detect the neck midpoint from a portrait/bust photo.

    Uses MediaPipe PoseLandmarker (Tasks API) to locate shoulders and nose,
    then interpolates between the shoulder midpoint (neck base, ~C7/T1) and
    nose to approximate the mid-cervical level (~C3-C4).

    Args:
        image: PIL Image of the portrait.
        interpolation_ratio: How far from shoulder midpoint toward nose.
            0.0 = shoulder midpoint, 1.0 = nose. Default 0.35.
        min_detection_confidence: MediaPipe detection confidence threshold.
        min_visibility: Minimum landmark visibility to accept result.

    Returns:
        Tuple of (NeckMidpoint, MediaPipeDebug) with coordinates in photo
        pixels, or None if pose detection fails or landmark visibility is
        too low.

    Raises:
        ImportError: If mediapipe is not installed.
    """
    import mediapipe as mp

    model_path = _get_model_path()

    image_array = np.array(image)
    h, w = image_array.shape[:2]

    # Ensure RGB (3-channel) input
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        min_pose_detection_confidence=min_detection_confidence,
        num_poses=1,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]

    # Landmark indices (same as legacy PoseLandmark enum values)
    NOSE = 0
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    nose = landmarks[NOSE]
    mouth_left = landmarks[MOUTH_LEFT]
    mouth_right = landmarks[MOUTH_RIGHT]
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]

    if (
        left_shoulder.visibility < min_visibility
        or right_shoulder.visibility < min_visibility
    ):
        return None

    # Convert all 33 landmarks to absolute pixels for debug visualization
    all_landmarks = tuple((lm.x * w, lm.y * h) for lm in landmarks)
    debug = MediaPipeDebug(landmarks=all_landmarks)

    # Convert normalized coords to absolute pixels
    nose_px = (nose.x * w, nose.y * h)
    mouth_left_px = (mouth_left.x * w, mouth_left.y * h)
    mouth_right_px = (mouth_right.x * w, mouth_right.y * h)
    left_shoulder_px = (left_shoulder.x * w, left_shoulder.y * h)
    right_shoulder_px = (right_shoulder.x * w, right_shoulder.y * h)

    # Shoulder midpoint (neck base, ~C7/T1 vertebra level)
    shoulder_mid_x = (left_shoulder_px[0] + right_shoulder_px[0]) / 2
    shoulder_mid_y = (left_shoulder_px[1] + right_shoulder_px[1]) / 2

    # Interpolate toward nose to approximate mid-cervical (C3-C4)
    neck_x = shoulder_mid_x + interpolation_ratio * (nose_px[0] - shoulder_mid_x)
    neck_y = shoulder_mid_y + interpolation_ratio * (nose_px[1] - shoulder_mid_y)

    neck_midpoint = NeckMidpoint(
        x=neck_x,
        y=neck_y,
        left_shoulder=left_shoulder_px,
        right_shoulder=right_shoulder_px,
        nose=nose_px,
        mouth_left=mouth_left_px,
        mouth_right=mouth_right_px,
        left_shoulder_visibility=left_shoulder.visibility,
        right_shoulder_visibility=right_shoulder.visibility,
        nose_visibility=nose.visibility,
        interpolation_ratio=interpolation_ratio,
    )

    return neck_midpoint, debug
