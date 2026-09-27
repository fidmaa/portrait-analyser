"""Explicit pinhole camera model + EXIF-orientation geometry helpers.

Legacy (iPhone Camera-app) photos are measured with the calibration
polynomial in :mod:`portrait_analyser.incisor` (``pixels_per_mm_at_distance``),
fitted to iPhone 14 TrueDepth data. Photos from the dedicated TrueDepth
capture app instead carry absolute depth plus the camera's own intrinsic
matrix; for those, :class:`CameraModel` supplies a per-file focal length and
principal point (in pixels of the *upright photo* as returned by
``load_image``), and the metric conversion uses the plain pinhole model::

    X_mm = (u - cx) * Z_mm / fx
    Y_mm = (v - cy) * Z_mm / fy

Every metric function takes an optional keyword-only ``camera=None``; ``None``
keeps the legacy polynomial behaviour exactly.

Orientation helpers: the capture app stores the photo and mattes already
upright, but AVFoundation hands back the depth map (and reports the
calibration) in other frames -- see :func:`intrinsics_in_photo_space`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# EXIF orientations whose display transform swaps the image axes.
_AXIS_SWAPPING_ORIENTATIONS = frozenset({5, 6, 7, 8})

# Relative tolerance when comparing aspect ratios of two frames.
_ASPECT_TOLERANCE = 0.01


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics in pixels of the upright photo returned by ``load_image``.

    :ivar fx: focal length along the photo's x axis, pixels
    :ivar fy: focal length along the photo's y axis, pixels
    :ivar cx: principal point x, photo pixels (continuous coordinates:
        pixel ``i`` spans ``[i, i + 1)``)
    :ivar cy: principal point y, photo pixels
    :ivar width: width in pixels of the image these intrinsics refer to
        (the full-resolution upright photo, e.g. 3024), or None if unknown.
        Coordinates from a scaled display (e.g. the GUI's 480x640) must be
        converted to this pixel space before use.
    :ivar height: height counterpart of ``width``
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: float | None = None
    height: float | None = None

    def __post_init__(self):
        if not (self.fx > 0 and self.fy > 0):
            raise ValueError(
                f"focal lengths must be positive, got fx={self.fx}, fy={self.fy}"
            )

    @classmethod
    def from_portrait(cls, portrait) -> CameraModel | None:
        """Build the model from an ``IOSPortrait``'s file intrinsics.

        Returns ``None`` when the portrait carries no usable intrinsics --
        which, deliberately, includes every legacy Camera-app photo (their
        measurements keep using the calibration polynomial) -- or when its
        depth is not ``"absolute"`` (relative depth has an unknown scale, so
        a metric pinhole conversion would only look precise) or
        ``depth_plausible is False``.
        """
        focal = getattr(portrait, "focal_length_px", None)
        principal = getattr(portrait, "principal_point_px", None)
        if focal is None or principal is None:
            return None
        if getattr(portrait, "depth_accuracy", None) != "absolute":
            return None
        if getattr(portrait, "depth_plausible", None) is False:
            # Second line of defence: implausible (e.g. inverted) depth must
            # not be measured, even by a caller that forgot to check.
            return None
        photo = getattr(portrait, "photo", None)
        width, height = (None, None) if photo is None else photo.size
        return cls(
            fx=float(focal[0]),
            fy=float(focal[1]),
            cx=float(principal[0]),
            cy=float(principal[1]),
            width=width,
            height=height,
        )

    def axis(self, axis: str) -> tuple[float, float]:
        """Return ``(focal_px, principal_px)`` for ``"x"`` or ``"y"``."""
        if axis == "x":
            return self.fx, self.cx
        if axis == "y":
            return self.fy, self.cy
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def camera_axis(
    camera: CameraModel | None, axis: str
) -> tuple[float | None, float | None]:
    """``(focal_px, principal_px)`` for one axis, or ``(None, None)`` without a camera."""
    if camera is None:
        return None, None
    return camera.axis(axis)


def rotate_by_exif_orientation(
    array: np.ndarray, orientation: int | None
) -> np.ndarray:
    """Apply the EXIF ``Orientation`` display transform to a 2-D array.

    Row 0 of the input is the top of the *stored* image; the result is the
    image as it should be displayed. ``None``/``1`` return the input
    unchanged (same object). Orientations 2-8 follow the EXIF 2.3 spec
    (same semantics as ``PIL.ImageOps.exif_transpose``).
    """
    if orientation in (None, 1):
        return array
    if orientation == 2:
        out = np.fliplr(array)
    elif orientation == 3:
        out = np.rot90(array, 2)
    elif orientation == 4:
        out = np.flipud(array)
    elif orientation == 5:
        out = np.swapaxes(array, 0, 1)  # transpose
    elif orientation == 6:
        out = np.rot90(array, -1)  # 90 degrees clockwise
    elif orientation == 7:
        out = np.swapaxes(np.rot90(array, 2), 0, 1)  # transverse
    elif orientation == 8:
        out = np.rot90(array, 1)  # 90 degrees counter-clockwise
    else:
        raise ValueError(f"invalid EXIF orientation: {orientation!r}")
    return np.ascontiguousarray(out)


def map_point_by_exif_orientation(
    x: float, y: float, width: float, height: float, orientation: int | None
) -> tuple[float, float, float, float]:
    """Map a point through the same transform as :func:`rotate_by_exif_orientation`.

    Coordinates are continuous (pixel ``i`` spans ``[i, i + 1)``, so a pixel
    centre is ``i + 0.5``). Returns ``(x', y', width', height')`` in the
    transformed frame.
    """
    w, h = width, height
    if orientation in (None, 1):
        return x, y, w, h
    if orientation == 2:
        return w - x, y, w, h
    if orientation == 3:
        return w - x, h - y, w, h
    if orientation == 4:
        return x, h - y, w, h
    if orientation == 5:
        return y, x, h, w
    if orientation == 6:
        return h - y, x, h, w
    if orientation == 7:
        return h - y, w - x, h, w
    if orientation == 8:
        return y, w - x, h, w
    raise ValueError(f"invalid EXIF orientation: {orientation!r}")


def _is_landscape(width: float, height: float) -> bool:
    return width >= height


def _same_aspect(size_a, size_b) -> bool:
    return abs(size_a[0] / size_a[1] - size_b[0] / size_b[1]) <= _ASPECT_TOLERANCE * (
        size_b[0] / size_b[1]
    )


def intrinsics_in_photo_space(
    intrinsics: tuple[float, float, float, float],
    reference_size: tuple[float, float],
    sensor_size: tuple[int, int],
    exif_orientation: int | None,
    photo_size: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Express AVCameraCalibrationData intrinsics in upright-photo pixels.

    :param intrinsics: ``(fx, fy, cx, cy)`` at ``reference_size``
    :param reference_size: ``(width, height)`` of the intrinsics' frame
    :param sensor_size: ``(width, height)`` of the depth map as stored
        (sensor orientation, before :func:`rotate_by_exif_orientation`)
    :param exif_orientation: the file's EXIF ``Orientation``
    :param photo_size: ``(width, height)`` of the upright photo
    :returns: ``((fx, fy), (cx, cy))`` in photo pixels

    Which frame ``reference_size`` refers to is decided from its aspect:

    * Same aspect as the stored depth map (Apple's usual convention, e.g.
      every Camera-app file seen so far): the intrinsics live in the sensor
      frame, so the principal point goes through the same EXIF transform as
      the depth map, then gets scaled to the photo.
    * Transposed aspect (the capture app's files: depth 640x480 landscape,
      reference always 3024x4032 portrait, for EXIF orientation 6 *and* 3):
      the reference is taken to be the sensor frame rotated 90 degrees
      clockwise -- the EXIF-6 "portrait" frame -- so it is first mapped back
      to the sensor frame (90 degrees counter-clockwise) and then through
      the file's EXIF transform like the depth map. For orientation 6 this is
      the identity; for orientation 3 it is a 90-degree clockwise rotation.
      This assumption only moves the principal point by its offset from the
      image centre (~13 px of 3024 on the files seen), which changes a 3-D
      distance by ``offset * dZ / f`` -- about 0.05 mm for points 1 cm apart
      in depth.
    * If the mapped frame's aspect still does not match the photo, the
      principal point falls back to the photo centre (logged as a warning)
      and the focal length is scaled by the long-side ratio.
    """
    fx, fy, cx, cy = (float(v) for v in intrinsics)
    ref_w, ref_h = float(reference_size[0]), float(reference_size[1])
    photo_w, photo_h = float(photo_size[0]), float(photo_size[1])

    swapped = False
    if _is_landscape(ref_w, ref_h) != _is_landscape(*sensor_size):
        # Reference frame is the sensor frame rotated 90 deg clockwise; undo it.
        cx, cy, ref_w, ref_h = map_point_by_exif_orientation(cx, cy, ref_w, ref_h, 8)
        swapped = not swapped

    cx, cy, ref_w, ref_h = map_point_by_exif_orientation(
        cx, cy, ref_w, ref_h, exif_orientation
    )
    if exif_orientation in _AXIS_SWAPPING_ORIENTATIONS:
        swapped = not swapped
    if swapped:
        fx, fy = fy, fx

    if not _same_aspect((ref_w, ref_h), (photo_w, photo_h)):
        logger.warning(
            "Camera intrinsics reference frame %sx%s does not match the photo's "
            "aspect %sx%s after orientation mapping; using the photo centre as "
            "the principal point.",
            ref_w,
            ref_h,
            photo_w,
            photo_h,
        )
        scale = max(photo_w, photo_h) / max(ref_w, ref_h)
        if _is_landscape(ref_w, ref_h) != _is_landscape(photo_w, photo_h):
            fx, fy = fy, fx
        return (fx * scale, fy * scale), (photo_w / 2.0, photo_h / 2.0)

    scale_x = photo_w / ref_w
    scale_y = photo_h / ref_h
    return (fx * scale_x, fy * scale_y), (cx * scale_x, cy * scale_y)
