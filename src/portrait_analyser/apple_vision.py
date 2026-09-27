"""macOS-only Apple Vision body pose (``VNDetectHumanBodyPoseRequest``).

Used by :mod:`portrait_analyser.neck_width` for the bottom of the neck band:
Vision's ``neck_1_joint`` is the mid-point of the shoulder line (the neck
base), not the anatomical mid-neck.

**The padded-canvas trick.** On a close frontal portrait (the TrueDepth
capture app: head and neck fill the frame) ``VNDetectHumanBodyPoseRequest``
returns *no* observation -- the person is too large. Pasting the photo at
1/3 scale into a grey canvas of the photo's own size (horizontally centred,
a third of the free height from the top) makes it detect the head, shoulders
and neck reliably; the points are mapped back to photo pixels. Verified on
IMG_2389/IMG_2386 (neck_1 conf 0.71 / 0.60); the full frame gives 0
observations on both.

Like :mod:`portrait_analyser.apple_depth`, the pyobjc frameworks (Vision,
Quartz) are imported lazily inside :func:`detect_body_pose`, so the package
keeps importing on Linux; there :func:`detect_body_pose` raises
:class:`~portrait_analyser.exceptions.AppleVisionUnavailable` and callers
fall back to something else.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .exceptions import AppleVisionError, AppleVisionUnavailable

logger = logging.getLogger(__name__)

# Scale of the photo on the padded canvas (see the module docstring).
PADDED_CANVAS_PHOTO_SCALE = 1.0 / 3.0

# The canvas is the photo's size times this factor. Vision resizes its input
# to a small network resolution anyway, so a half-resolution canvas (photo at
# 1/6 of its pixels, same *relative* layout) gives the same joints and is
# ~4x cheaper to build and hand over than a full-size one.
PADDED_CANVAS_RESOLUTION = 0.5

# Grey of the padding.
PADDED_CANVAS_FILL = (128, 128, 128)

NECK_JOINT = "neck_1_joint"


@dataclass(frozen=True)
class BodyPose:
    """2-D body joints in photo pixels.

    :ivar joints: ``{joint name: (x, y, confidence)}`` for every joint Vision
        reported with confidence > 0 (Vision names, e.g. ``"neck_1_joint"``,
        ``"left_shoulder_1_joint"``). Points may lie outside the photo.
    :ivar source: how it was obtained, e.g. ``"vision-padded-canvas"``
    """

    joints: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    source: str = "vision-padded-canvas"

    def joint(self, name: str, min_confidence: float = 0.0):
        """``(x, y, confidence)`` of ``name`` if present with at least
        ``min_confidence``, else None."""
        value = self.joints.get(name)
        if value is None or value[2] < min_confidence:
            return None
        return value

    def neck(self, min_confidence: float = 0.0):
        """The ``neck_1_joint`` (neck base, mid-shoulders) or None."""
        return self.joint(NECK_JOINT, min_confidence)


def _import_backend():
    """Import Vision + Quartz lazily, or raise :class:`AppleVisionUnavailable`."""
    if sys.platform != "darwin":
        raise AppleVisionUnavailable(
            f"Apple Vision body pose requires macOS; running on sys.platform={sys.platform!r}."
        )
    try:
        import Quartz
        import Vision
    except ImportError as exc:
        raise AppleVisionUnavailable(
            "Apple Vision body pose requires the pyobjc-framework-Vision and "
            "pyobjc-framework-Quartz packages, which are not importable here."
        ) from exc
    return Vision, Quartz


def _pyobjc_error_types() -> tuple:
    """``(objc.error,)`` -- pyobjc's own bridge error -- or ``()`` without pyobjc."""
    try:
        import objc
    except ImportError:
        # No pyobjc-core means no bridge calls, so no objc.error can occur;
        # _import_backend has already raised AppleVisionUnavailable then.
        return ()
    return (objc.error,)


def padded_canvas(
    photo: Image.Image, scale=PADDED_CANVAS_PHOTO_SCALE, resolution=PADDED_CANVAS_RESOLUTION
):
    """The padded canvas and its mapping back to photo pixels.

    :returns: ``(canvas, (offset_x, offset_y), photo_scale)`` where a canvas
        pixel ``(u, v)`` is photo pixel ``((u - offset_x) / photo_scale,
        (v - offset_y) / photo_scale)``.
    """
    width, height = photo.size
    canvas_w = max(1, round(width * resolution))
    canvas_h = max(1, round(height * resolution))
    photo_scale = scale * resolution
    small = photo.convert("RGB").resize(
        (max(1, round(width * photo_scale)), max(1, round(height * photo_scale))),
        Image.BILINEAR,
    )
    canvas = Image.new("RGB", (canvas_w, canvas_h), PADDED_CANVAS_FILL)
    offset = ((canvas_w - small.width) // 2, (canvas_h - small.height) // 3)
    canvas.paste(small, offset)
    return canvas, offset, photo_scale


def _cgimage_from_rgb(Quartz, image: Image.Image):
    """A CGImage over the RGB bytes of a PIL image (no file, no PNG encode)."""
    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    height, width = rgb.shape[:2]
    data = rgb.tobytes()
    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    colour_space = Quartz.CGColorSpaceCreateDeviceRGB()
    cg_image = Quartz.CGImageCreate(
        width,
        height,
        8,
        24,
        width * 3,
        colour_space,
        Quartz.kCGImageAlphaNone,
        provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault,
    )
    if cg_image is None:
        raise AppleVisionError(f"CGImageCreate failed for a {width}x{height} RGB image")
    # Keep the bytes alive as long as the CGImage (the provider does not copy).
    return cg_image, data


def detect_body_pose(photo: Image.Image) -> BodyPose | None:
    """Run ``VNDetectHumanBodyPoseRequest`` on the padded canvas of ``photo``.

    :returns: :class:`BodyPose` in photo pixels, or None when Vision finds no
        person.
    :raises AppleVisionUnavailable: not macOS, or pyobjc Vision/Quartz missing.
    :raises AppleVisionError: Vision failed, or pyobjc returned data of an
        unexpected shape -- surfaced with context, never swallowed.
    """
    Vision, Quartz = _import_backend()
    unexpected = (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        *_pyobjc_error_types(),
    )
    try:
        return _detect_body_pose(Vision, Quartz, photo)
    except unexpected as exc:
        raise AppleVisionError(
            f"unexpected Apple Vision data during body pose detection: {type(exc).__name__}: {exc}"
        ) from exc


def _detect_body_pose(Vision, Quartz, photo):
    canvas, (offset_x, offset_y), photo_scale = padded_canvas(photo)
    canvas_w, canvas_h = canvas.size
    cg_image, _keepalive = _cgimage_from_rgb(Quartz, canvas)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
    # ``init`` is NS_UNAVAILABLE on Vision requests.
    request = Vision.VNDetectHumanBodyPoseRequest.alloc().initWithCompletionHandler_(None)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise AppleVisionError(f"VNDetectHumanBodyPoseRequest failed: {error}")
    observations = request.results() or []
    if not observations:
        return None
    if len(observations) > 1:
        logger.info("Vision found %d people; using the most confident", len(observations))
    observation = max(observations, key=lambda o: float(o.confidence()))
    points, error = observation.recognizedPointsForGroupKey_error_(
        Vision.VNHumanBodyPoseObservationJointsGroupNameAll, None
    )
    if points is None:
        raise AppleVisionError(f"recognizedPointsForGroupKey failed: {error}")
    joints = {}
    for name, point in dict(points).items():
        confidence = float(point.confidence())
        if confidence <= 0:
            continue
        location = point.location()
        # Vision: normalised, origin bottom-left.
        u = float(location.x) * canvas_w
        v = (1.0 - float(location.y)) * canvas_h
        joints[str(name)] = (
            (u - offset_x) / photo_scale,
            (v - offset_y) / photo_scale,
            confidence,
        )
    return BodyPose(joints=joints)
