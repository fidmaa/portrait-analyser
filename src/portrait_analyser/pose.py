"""Neck midpoint detection via MediaPipe Pose estimation.

Requires the optional ``pose`` extra::

    pip install portrait-analyser[pose]
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


class PortraitPose(Enum):
    NEUTRAL_NECK = "neutral_neck"
    OPEN_MOUTH = "open_mouth"
    EXTENDED_NECK = "extended_neck"


MOUTH_OPEN_THRESHOLD = 0.25

if TYPE_CHECKING:
    from PIL import Image

# Face-flattening detection: when the head tilts back (neck extension),
# vertical facial distances compress while horizontal IPD stays stable.
# Ratio below this threshold indicates neck extension.
FACE_FLATNESS_THRESHOLD = 0.85
FACE_FLATNESS_THRESHOLD_POSE = 1.2  # more lenient for Pose-only landmarks (less precise)
MIN_IPD_PIXELS = 20

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/"
    "pose_landmarker_heavy.task"
)
_CACHE_DIR = Path.home() / ".cache" / "portrait-analyser"
_MODEL_FILENAME = "pose_landmarker_heavy.task"

_FACE_MESH_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)
_FACE_MESH_MODEL_FILENAME = "face_landmarker.task"
_FACE_MESH_CHIN_INDEX = 152


@dataclass(frozen=True)
class MediaPipeDebug:
    """All raw MediaPipe pose landmarks for debug visualization."""

    landmarks: tuple[tuple[float, float], ...]  # 33 (x, y) in photo pixels


@dataclass(frozen=True)
class FaceMeshDebug:
    """All raw MediaPipe Face Mesh landmarks for debug visualization."""

    landmarks: tuple[tuple[float, float], ...]  # 478 (x, y) in photo pixels


@dataclass(frozen=True)
class FaceMeshAnalysis:
    chin: tuple[float, float]
    mouth_open_ratio: float | None
    nose: tuple[float, float]
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]
    debug: FaceMeshDebug


@dataclass(frozen=True)
class NeckMidpoint:
    """Detected neck midpoint with supporting landmarks.

    All coordinates are in absolute photo-space pixels.
    Shoulder-dependent fields (x, y, left_shoulder, right_shoulder,
    *_visibility, interpolation_ratio) are None when only FaceMesh
    detected the face but Pose landmarks were unavailable.
    """

    nose: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]
    chin: tuple[float, float]  # estimated chin (mentum) position
    neck_extended: bool  # True when neck appears maximally extended
    face_flatness_ratio: float | None  # debug: eye-mouth vertical / IPD ratio
    pose: PortraitPose
    mouth_open_ratio: float | None
    # Shoulder-dependent fields (None when only FaceMesh detected)
    x: float | None = None
    y: float | None = None
    left_shoulder: tuple[float, float] | None = None
    right_shoulder: tuple[float, float] | None = None
    left_shoulder_visibility: float | None = None
    right_shoulder_visibility: float | None = None
    nose_visibility: float | None = None
    interpolation_ratio: float | None = None


def _download_model(url: str, filename: str) -> str:
    """Return path to a cached model file, downloading if needed."""
    model_path = _CACHE_DIR / filename
    if not model_path.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = model_path.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.replace(tmp_path, model_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    return str(model_path)


def _get_model_path() -> str:
    """Return path to the PoseLandmarker .task model, downloading if needed."""
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        raise ImportError(
            "mediapipe is required for pose estimation. "
            "Install it with: pip install portrait-analyser[pose]"
        ) from None
    return _download_model(_MODEL_URL, _MODEL_FILENAME)


def _get_face_mesh_model_path() -> str:
    """Return path to the FaceLandmarker .task model, downloading if needed."""
    return _download_model(_FACE_MESH_MODEL_URL, _FACE_MESH_MODEL_FILENAME)


def _detect_chin_via_face_mesh(
    mp_image,
    w: int,
    h: int,
    min_detection_confidence: float,
) -> FaceMeshAnalysis | None:
    """Detect chin (mentum) and mouth openness using MediaPipe Face Mesh.

    Returns FaceMeshAnalysis with chin coords, mouth_open_ratio, and debug data,
    or None if detection fails.
    """
    import mediapipe as mp

    try:
        model_path = _get_face_mesh_model_path()
    except Exception:
        return None

    try:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            min_face_detection_confidence=min_detection_confidence,
            num_faces=1,
        )

        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]
        all_landmarks = tuple((lm.x * w, lm.y * h) for lm in face_landmarks)
        face_mesh_debug = FaceMeshDebug(landmarks=all_landmarks)

        chin = face_landmarks[_FACE_MESH_CHIN_INDEX]
        chin_px = (chin.x * w, chin.y * h)

        # Extract face landmarks from FaceMesh:
        # 1 = nose tip, 133 = left eye inner, 362 = right eye inner
        # 61 = mouth left corner, 291 = mouth right corner
        nose_lm = face_landmarks[1]
        nose_px = (nose_lm.x * w, nose_lm.y * h)
        left_eye_lm = face_landmarks[133]
        left_eye_px = (left_eye_lm.x * w, left_eye_lm.y * h)
        right_eye_lm = face_landmarks[362]
        right_eye_px = (right_eye_lm.x * w, right_eye_lm.y * h)
        mouth_left_lm = face_landmarks[61]
        mouth_left_px = (mouth_left_lm.x * w, mouth_left_lm.y * h)
        mouth_right_lm = face_landmarks[291]
        mouth_right_px = (mouth_right_lm.x * w, mouth_right_lm.y * h)

        # Compute mouth-open ratio using FaceMesh landmarks:
        # 13 = upper lip inner center, 14 = lower lip inner center
        # 78 = left mouth corner, 308 = right mouth corner (for width)
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        mouth_l_corner = face_landmarks[78]
        mouth_r_corner = face_landmarks[308]

        upper_lip_px = (upper_lip.x * w, upper_lip.y * h)
        lower_lip_px = (lower_lip.x * w, lower_lip.y * h)
        mouth_l_px = (mouth_l_corner.x * w, mouth_l_corner.y * h)
        mouth_r_px = (mouth_r_corner.x * w, mouth_r_corner.y * h)

        vertical_opening = abs(lower_lip_px[1] - upper_lip_px[1])
        mouth_width = abs(mouth_r_px[0] - mouth_l_px[0])

        mouth_open_ratio: float | None = None
        if mouth_width > 5:
            mouth_open_ratio = vertical_opening / mouth_width
            print(f"  Mouth open ratio: {mouth_open_ratio:.3f} (threshold: {MOUTH_OPEN_THRESHOLD})")

        return FaceMeshAnalysis(
            chin=chin_px,
            mouth_open_ratio=mouth_open_ratio,
            nose=nose_px,
            left_eye=left_eye_px,
            right_eye=right_eye_px,
            mouth_left=mouth_left_px,
            mouth_right=mouth_right_px,
            debug=face_mesh_debug,
        )
    except Exception:
        return None


def detect_neck_midpoint(
    image: Image.Image,
    interpolation_ratio: float = 0.35,
    min_detection_confidence: float = 0.5,
    min_visibility: float = 0.5,
) -> tuple[NeckMidpoint | None, MediaPipeDebug | None, FaceMeshDebug | None]:
    """Detect the neck midpoint from a portrait/bust photo.

    Uses MediaPipe PoseLandmarker (Tasks API) to locate shoulders and nose,
    then interpolates between the shoulder midpoint (neck base, ~C7/T1) and
    nose to approximate the mid-cervical level (~C3-C4).

    FaceMesh detection is always attempted independently of pose detection,
    so ``result[2]`` (FaceMeshDebug) may be populated even when pose fails.

    Args:
        image: PIL Image of the portrait.
        interpolation_ratio: How far from shoulder midpoint toward nose.
            0.0 = shoulder midpoint, 1.0 = nose. Default 0.35.
        min_detection_confidence: MediaPipe detection confidence threshold.
        min_visibility: Minimum landmark visibility to accept result.

    Returns:
        3-tuple of (NeckMidpoint | None, MediaPipeDebug | None,
        FaceMeshDebug | None).  First two elements are None when pose
        detection fails or landmark visibility is too low.

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

    # Always attempt FaceMesh — it may succeed even when pose detection fails
    face_mesh_result = _detect_chin_via_face_mesh(
        mp_image, w, h, min_detection_confidence
    )
    face_mesh_debug: FaceMeshDebug | None = None
    face_mesh_chin: tuple[float, float] | None = None
    mouth_open_ratio: float | None = None
    if face_mesh_result is not None:
        face_mesh_chin = face_mesh_result.chin
        mouth_open_ratio = face_mesh_result.mouth_open_ratio
        face_mesh_debug = face_mesh_result.debug
        print(f"  Chin detected via Face Mesh at: ({face_mesh_chin[0]:.1f}, {face_mesh_chin[1]:.1f})")
    else:
        print("  Face Mesh chin detection failed")

    # Use FaceMesh landmarks for face analysis (eyes, nose, mouth)
    # when available; they are more reliable than Pose for facial features.
    if face_mesh_result is not None:
        nose_px = face_mesh_result.nose
        left_eye_px = face_mesh_result.left_eye
        right_eye_px = face_mesh_result.right_eye
        mouth_left_px = face_mesh_result.mouth_left
        mouth_right_px = face_mesh_result.mouth_right
    else:
        nose_px = None
        left_eye_px = None
        right_eye_px = None
        mouth_left_px = None
        mouth_right_px = None

    # Attempt Pose detection for shoulders
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        min_pose_detection_confidence=min_detection_confidence,
        num_poses=1,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    pose_debug: MediaPipeDebug | None = None
    left_shoulder_px: tuple[float, float] | None = None
    right_shoulder_px: tuple[float, float] | None = None
    left_shoulder_vis: float | None = None
    right_shoulder_vis: float | None = None
    nose_vis: float | None = None
    has_shoulders = False

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # Landmark indices (same as legacy PoseLandmark enum values)
        NOSE = 0
        LEFT_EYE = 2
        RIGHT_EYE = 5
        MOUTH_LEFT = 9
        MOUTH_RIGHT = 10
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12

        # Convert all 33 landmarks to absolute pixels for debug visualization
        all_landmarks = tuple((lm.x * w, lm.y * h) for lm in landmarks)
        pose_debug = MediaPipeDebug(landmarks=all_landmarks)

        nose_vis = landmarks[NOSE].visibility

        # Fall back to Pose for face landmarks if FaceMesh failed
        if nose_px is None:
            nose_px = (landmarks[NOSE].x * w, landmarks[NOSE].y * h)
            left_eye_px = (landmarks[LEFT_EYE].x * w, landmarks[LEFT_EYE].y * h)
            right_eye_px = (landmarks[RIGHT_EYE].x * w, landmarks[RIGHT_EYE].y * h)
            mouth_left_px = (landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h)
            mouth_right_px = (landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h)

        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_shoulder_vis = left_shoulder.visibility
        right_shoulder_vis = right_shoulder.visibility

        if (
            left_shoulder.visibility >= min_visibility
            and right_shoulder.visibility >= min_visibility
        ):
            left_shoulder_px = (left_shoulder.x * w, left_shoulder.y * h)
            right_shoulder_px = (right_shoulder.x * w, right_shoulder.y * h)
            has_shoulders = True
        else:
            print(
                f"  Shoulder visibility too low"
                f" (L={left_shoulder.visibility:.2f},"
                f" R={right_shoulder.visibility:.2f})"
            )
    else:
        print("  Pose landmarks not detected")

    # If neither FaceMesh nor Pose detected a face, give up
    if nose_px is None:
        return None, pose_debug, face_mesh_debug

    # Face-flattening detection using eye/mouth landmarks (from FaceMesh or Pose)
    assert left_eye_px is not None and right_eye_px is not None
    assert mouth_left_px is not None and mouth_right_px is not None

    using_facemesh_landmarks = face_mesh_result is not None
    flatness_threshold = (
        FACE_FLATNESS_THRESHOLD if using_facemesh_landmarks else FACE_FLATNESS_THRESHOLD_POSE
    )
    print(
        f"  Face-flattening landmarks source: "
        f"{'FaceMesh' if using_facemesh_landmarks else 'Pose'}"
        f" (threshold: {flatness_threshold})"
    )

    mouth_center_x = (mouth_left_px[0] + mouth_right_px[0]) / 2
    mouth_center_y = (mouth_left_px[1] + mouth_right_px[1]) / 2
    nose_to_mouth = mouth_center_y - nose_px[1]

    ipd = abs(right_eye_px[0] - left_eye_px[0])
    eye_mid_y = (left_eye_px[1] + right_eye_px[1]) / 2
    eye_to_mouth_vertical = mouth_center_y - eye_mid_y
    lateral_tilt = abs(left_eye_px[1] - right_eye_px[1])

    print(
        f"  Eye L=({left_eye_px[0]:.1f}, {left_eye_px[1]:.1f}),"
        f" R=({right_eye_px[0]:.1f}, {right_eye_px[1]:.1f})"
        f" | IPD={ipd:.1f}px"
    )
    print(
        f"  Mouth L=({mouth_left_px[0]:.1f}, {mouth_left_px[1]:.1f}),"
        f" R=({mouth_right_px[0]:.1f}, {mouth_right_px[1]:.1f})"
        f" | center=({mouth_center_x:.1f}, {mouth_center_y:.1f})"
    )
    print(
        f"  eye_mid_y={eye_mid_y:.1f}, eye_to_mouth_vertical={eye_to_mouth_vertical:.1f},"
        f" lateral_tilt={lateral_tilt:.1f}"
    )

    neck_extended = False
    face_flatness_ratio: float | None = None

    if ipd < MIN_IPD_PIXELS:
        print(f"  IPD too small ({ipd:.1f}px < {MIN_IPD_PIXELS}px), skipping face-flattening")
    elif lateral_tilt > 0.3 * ipd:
        print(
            f"  Lateral head tilt detected ({lateral_tilt:.1f} > {0.3 * ipd:.1f}),"
            " skipping face-flattening"
        )
    else:
        face_flatness_ratio = eye_to_mouth_vertical / ipd
        neck_extended = face_flatness_ratio < flatness_threshold
        print(
            f"  Face flatness ratio: {face_flatness_ratio:.3f}"
            f" (threshold: {flatness_threshold},"
            f" extended={neck_extended})"
        )

    # Chin: prefer Face Mesh landmark 152, fall back to Pose estimation
    if face_mesh_chin is not None:
        chin_px = face_mesh_chin
    else:
        if neck_extended:
            chin_px = (mouth_center_x, mouth_center_y)
        else:
            chin_px = (mouth_center_x, mouth_center_y + 2.0 * nose_to_mouth)
        print(f"  Chin estimated via Pose at: ({chin_px[0]:.1f}, {chin_px[1]:.1f})")

    # Compute neck midpoint position (requires shoulders)
    neck_x: float | None = None
    neck_y: float | None = None
    if has_shoulders:
        assert left_shoulder_px is not None and right_shoulder_px is not None
        shoulder_mid_x = (left_shoulder_px[0] + right_shoulder_px[0]) / 2
        shoulder_mid_y = (left_shoulder_px[1] + right_shoulder_px[1]) / 2
        neck_x = shoulder_mid_x + interpolation_ratio * (nose_px[0] - shoulder_mid_x)
        neck_y = shoulder_mid_y + interpolation_ratio * (nose_px[1] - shoulder_mid_y)

    # Classify pose
    if mouth_open_ratio is not None and mouth_open_ratio > MOUTH_OPEN_THRESHOLD:
        pose = PortraitPose.OPEN_MOUTH
    elif neck_extended:
        pose = PortraitPose.EXTENDED_NECK
    else:
        pose = PortraitPose.NEUTRAL_NECK
    print(
        f"  Pose classification: {pose.value}"
        f" (mouth_open_ratio={mouth_open_ratio}, neck_extended={neck_extended},"
        f" has_shoulders={has_shoulders})"
    )

    neck_midpoint = NeckMidpoint(
        nose=nose_px,
        mouth_left=mouth_left_px,
        mouth_right=mouth_right_px,
        chin=chin_px,
        neck_extended=neck_extended,
        face_flatness_ratio=face_flatness_ratio,
        pose=pose,
        mouth_open_ratio=mouth_open_ratio,
        x=neck_x,
        y=neck_y,
        left_shoulder=left_shoulder_px,
        right_shoulder=right_shoulder_px,
        left_shoulder_visibility=left_shoulder_vis,
        right_shoulder_visibility=right_shoulder_vis,
        nose_visibility=nose_vis,
        interpolation_ratio=interpolation_ratio if has_shoulders else None,
    )

    return neck_midpoint, pose_debug, face_mesh_debug
