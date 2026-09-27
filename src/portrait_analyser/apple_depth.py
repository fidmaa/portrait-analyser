"""macOS-only reader for Apple's embedded depth/disparity data.

Some HEIC files -- notably ones produced by third-party TrueDepth capture
apps rather than the stock Camera app -- store the depth auxiliary image as
16-bit, JPEG-compressed disparity. Neither ``pyheif``/``pyheif-iplweb`` nor
the latest ``pillow-heif`` (1.8.0 / libheif 1.23.4) can decode that: pyheif
raises ``HeifError: ... "Unsupported JPEG data precision 16"`` and
pillow-heif raises "JPEG decoder plugin not built in". Both only know how to
hand the aux image's raw bytes to a generic JPEG decoder.

macOS's own ImageIO + AVFoundation frameworks read these files without
issue, because they never go through a generic JPEG decoder for the depth
channel: ``CGImageSourceCopyAuxiliaryDataInfoAtIndex`` hands back Apple's
whole aux-data dictionary (compressed bytes + description + XMP metadata),
and ``AVDepthData(fromDictionaryRepresentation:)`` decodes it directly,
independent of bit depth. Converting to ``kCVPixelFormatType_DepthFloat32``
then yields a plain metres-denominated pixel buffer.

This module is macOS-only and imports Quartz/AVFoundation/Foundation
lazily, inside :func:`read_apple_depth` -- so the rest of the package keeps
importing cleanly on Linux (where the main HEIC path already uses plain
``pyheif``). Call :func:`read_apple_depth` to get the depth map; nothing
else in the package is affected by this module's presence.

Orientation: the returned depth map stays in the *stored* (sensor) pixel
orientation exactly as ImageIO/AVFoundation decode it (e.g. 640x480) -- it
is NOT rotated according to the photo's EXIF orientation. This is the one
place in the pipeline that still needs it: the primary photo and semantic
mattes decoded via pyheif/pillow-heif come out already upright (e.g. a
640x480 sensor frame becomes a 3024x4032 upright photo), even though the
file's own EXIF ``Orientation``/``ImageWidth``/``ImageHeight`` tags still
describe the *un-rotated* sensor frame -- confirmed visually on real
captures (IMG_2346/IMG_2347: EXIF Orientation 6, pillow-heif photo already
3024x4032 upright even with ``apply_transformations=False``). Only the
depth buffer AVFoundation hands back is still sensor-oriented and needs
that EXIF rotation applied to line up with the upright photo/mattes -- do
**not** rotate the pillow-heif photo/mattes by EXIF again, or they will be
rotated twice. See :attr:`AppleDepthData.exif_orientation`.
``ios.load_image`` applies that rotation (via
:func:`portrait_analyser.camera.rotate_by_exif_orientation`) when it falls
back to this reader, and then re-encodes the metric map with
:func:`encode_depth_as_disparity_8bit` so the rest of the pipeline keeps
working on the familiar 8-bit disparity image.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from .exceptions import AppleDepthDecodeError, AppleDepthUnavailable

# Depth values above this are treated as invalid sentinels rather than real
# distances. Observed on a real capture (IMG_2346.HEIC, same TrueDepth
# capture app the fixtures in this module target): 96.3% of pixels decode
# to a plausible 0.29-13.9 m, but a handful of near-zero-disparity pixels
# decode to implausible values up to 9999.975 m -- Apple's disparity ->
# depth conversion (1 / disparity, in metres) blows up as disparity
# approaches zero, so "no usable measurement" ends up encoded as a huge
# fake distance instead of directly as non-finite. A front-facing TrueDepth
# portrait capture is never taken from more than a few metres away, so 20 m
# is a generous cap that rejects the observed sentinel/outlier values
# (22-9999 m) while never touching real data (max observed: 13.9 m).
MAX_PLAUSIBLE_DEPTH_M = 20.0

# Far limit for the 8-bit disparity re-encoding done by
# :func:`encode_depth_as_disparity_8bit`. The 8-bit code spreads 255 steps
# linearly in disparity (1/Z) between 1/Z_far and 1/Z_near, so the far limit
# directly sets the quantisation step at face distance: with Z_near ~0.25 m
# and Z_far = 3 m one step is ~0.0144 1/m, i.e. ~1.3 mm of depth at 30 cm
# (step_Z ~= Z**2 * step_disparity), ~3.6 mm at 50 cm. Letting background
# walls at 5-14 m (seen on real captures) stretch the range instead would
# barely change that (1/Z_far is already small), but a portrait subject is
# never beyond ~1 m, so anything past 3 m is simply treated as "no depth" --
# which the downstream code already ignores as zero disparity.
DISPARITY_FAR_CAP_M = 3.0

# Percentile of valid depths used as the near end of the 8-bit encoding.
NEAR_PERCENTILE = 0.1


@dataclass
class AppleDepthData:
    """Depth data decoded via macOS ImageIO + AVFoundation.

    :ivar depth_m: ``(height, width)`` float32 array, metres, in the stored
        (sensor) pixel orientation -- see the module docstring. ``NaN``
        marks pixels that AVFoundation reported as non-finite, pixels
        <= 0 m, and pixels above :data:`MAX_PLAUSIBLE_DEPTH_M` (a documented
        sentinel/outlier cap, see there) -- never silently left as garbage.
    :ivar accuracy: ``"absolute"`` or ``"relative"``
        (``AVDepthDataAccuracy``).
    :ivar filtered: whether AVFoundation reports the map as smoothed
        (``isDepthDataFiltered``).
    :ivar quality: ``"high"`` or ``"low"`` (``AVDepthDataQuality``).
    :ivar source_type: ``"disparity"`` or ``"depth"``, whichever aux data
        type the file actually carried (disparity is preferred; depth is
        the fallback when a file has no disparity aux image).
    :ivar intrinsics: ``(fx, fy, cx, cy)`` in pixels, expressed at
        ``intrinsics_reference_size``, or ``None`` when the file carries no
        camera calibration data.
    :ivar intrinsics_reference_size: ``(width, height)`` in pixels that
        ``intrinsics`` is expressed in. This is Apple's reference image size
        (typically the full-resolution photo), which is *not* necessarily
        ``depth_m``'s own width/height -- callers must rescale. ``None``
        alongside ``intrinsics=None``.
    :ivar lens_distortion_center: ``(x, y)`` in the same reference pixel
        space as ``intrinsics_reference_size``, or ``None`` when
        unavailable.
    :ivar exif_orientation: the primary image's EXIF ``Orientation`` value
        (1-8), or ``None`` if it could not be read. This is the rotation a
        later step must apply to ``depth_m`` alone to line it up with the
        photo/mattes as decoded by ``ios.load_image`` -- those are already
        upright and must NOT be rotated by this value too (see the module
        docstring). This module does not apply any rotation itself.
    :ivar aux_orientation: the ``Orientation`` key of the aux image's
        ``kCGImageAuxiliaryDataInfoDataDescription`` (describes the depth
        buffer), or ``None`` when absent. Prefer :attr:`depth_orientation`.
    """

    depth_m: "np.ndarray"
    accuracy: str
    filtered: bool
    quality: str
    source_type: str
    intrinsics: Optional[Tuple[float, float, float, float]]
    intrinsics_reference_size: Optional[Tuple[float, float]]
    lens_distortion_center: Optional[Tuple[float, float]] = None
    exif_orientation: Optional[int] = None
    aux_orientation: int | None = None

    @property
    def depth_orientation(self) -> int | None:
        """Orientation to apply to ``depth_m``: the aux data's own
        ``Orientation`` (it describes the depth buffer itself) when present,
        else the primary image's EXIF value. :func:`read_apple_depth`
        refuses files where both exist and disagree."""
        return self.aux_orientation if self.aux_orientation is not None else self.exif_orientation


def _import_backend():
    """Import the pyobjc frameworks this module needs, or raise a clear error.

    Imported lazily (only when :func:`read_apple_depth` is actually called)
    so the rest of the package keeps importing on non-macOS platforms where
    these frameworks don't exist / aren't installed.
    """
    if sys.platform != "darwin":
        raise AppleDepthUnavailable(
            "Reading Apple depth/disparity data requires macOS ImageIO and "
            f"AVFoundation; running on sys.platform={sys.platform!r}."
        )
    try:
        import Quartz
        import AVFoundation
        from Foundation import NSURL
    except ImportError as exc:
        raise AppleDepthUnavailable(
            "Reading Apple depth/disparity data requires the "
            "pyobjc-framework-Quartz and pyobjc-framework-AVFoundation "
            "packages, which are not importable here."
        ) from exc
    return Quartz, AVFoundation, NSURL


def _pixel_buffer_to_depth_m(Quartz, pixel_buffer) -> "np.ndarray":
    """Read a DepthFloat32 CVPixelBuffer into an ``(height, width)`` float32 array.

    Respects ``bytesPerRow`` (it can exceed ``width * 4`` due to row
    alignment padding) rather than assuming a tightly packed buffer.
    """
    Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, Quartz.kCVPixelBufferLock_ReadOnly)
    try:
        width = Quartz.CVPixelBufferGetWidth(pixel_buffer)
        height = Quartz.CVPixelBufferGetHeight(pixel_buffer)
        bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
        base_address = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
        if base_address is None:
            raise AppleDepthDecodeError(
                "CVPixelBufferGetBaseAddress returned NULL for the decoded "
                "depth buffer"
            )
        raw = base_address.as_buffer(bytes_per_row * height)
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row)
        # .copy(): raw is a live view of the (about-to-be-unlocked) pixel
        # buffer; we must not hand back an array that aliases it.
        depth_m = rows[:, : width * 4].copy().view(np.float32).reshape(height, width)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, Quartz.kCVPixelBufferLock_ReadOnly)

    invalid = ~np.isfinite(depth_m) | (depth_m <= 0) | (depth_m > MAX_PLAUSIBLE_DEPTH_M)
    depth_m[invalid] = np.nan
    return depth_m


def _accuracy_name(AVFoundation, value) -> str:
    if value == AVFoundation.AVDepthDataAccuracyAbsolute:
        return "absolute"
    if value == AVFoundation.AVDepthDataAccuracyRelative:
        return "relative"
    raise AppleDepthDecodeError(f"unrecognised AVDepthDataAccuracy value: {value!r}")


def _quality_name(AVFoundation, value) -> str:
    if value == AVFoundation.AVDepthDataQualityHigh:
        return "high"
    if value == AVFoundation.AVDepthDataQualityLow:
        return "low"
    raise AppleDepthDecodeError(f"unrecognised AVDepthDataQuality value: {value!r}")


def _read_exif_orientation(Quartz, source) -> Optional[int]:
    properties = Quartz.CGImageSourceCopyPropertiesAtIndex(source, 0, None)
    if not properties:
        return None
    orientation = properties.get(Quartz.kCGImagePropertyOrientation)
    if orientation is None:
        return None
    return int(orientation)


def read_apple_depth(path: Union[str, "Path"]) -> Optional[AppleDepthData]:
    """Read Apple's embedded depth/disparity data via macOS ImageIO + AVFoundation.

    Works regardless of the aux image's bit depth or JPEG precision (see
    module docstring for why this succeeds where ``pyheif``/``pillow-heif``
    fail on 16-bit disparity). Prefers the disparity aux type
    (``kCGImageAuxiliaryDataTypeDisparity``) and falls back to depth
    (``kCGImageAuxiliaryDataTypeDepth``) when a file has no disparity aux
    image.

    :param path: path to a HEIC/HEIF (or any other ImageIO-readable
        container) file, potentially carrying an Apple depth/disparity
        auxiliary image.
    :returns: an :class:`AppleDepthData`, or ``None`` if the file has
        neither a disparity nor a depth auxiliary image.
    :raises AppleDepthUnavailable: not running on macOS, or the pyobjc
        Quartz/AVFoundation frameworks are not importable.
    :raises AppleDepthDecodeError: the file has a depth/disparity aux image
        but ImageIO/AVFoundation failed to decode it, the pyobjc-bridged data
        had an unexpected shape, or the aux data's orientation disagrees with
        the EXIF one -- surfaced with context rather than swallowed.
    """
    Quartz, AVFoundation, NSURL = _import_backend()
    try:
        return _read_apple_depth(Quartz, AVFoundation, NSURL, str(path))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        # pyobjc bridges (e.g. intrinsicMatrix()'s nested tuples, CGSize
        # attributes) can change shape across macOS / pyobjc versions; any
        # such surprise is a decode failure of this file, reported with
        # context rather than as a bare AttributeError/IndexError.
        raise AppleDepthDecodeError(
            f"unexpected ImageIO/AVFoundation data while reading depth from "
            f"{str(path)!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _calibration_to_intrinsics(calibration):
    """``(intrinsics, reference_size, lens_distortion_center)`` from an
    AVCameraCalibrationData (pyobjc proxy)."""
    # matrix_float3x3, column-major: column0=(fx,0,0), column1=(0,fy,0),
    # column2=(cx,cy,1). pyobjc bridges it as a 1-tuple wrapping the
    # 3x3 tuple-of-columns.
    matrix = calibration.intrinsicMatrix()[0]
    intrinsics = (
        float(matrix[0][0]),
        float(matrix[1][1]),
        float(matrix[2][0]),
        float(matrix[2][1]),
    )
    reference_size = calibration.intrinsicMatrixReferenceDimensions()
    intrinsics_reference_size = (
        float(reference_size.width),
        float(reference_size.height),
    )
    lens_distortion_center = None
    center = calibration.lensDistortionCenter()
    if center is not None:
        lens_distortion_center = (float(center.x), float(center.y))
    return intrinsics, intrinsics_reference_size, lens_distortion_center


def _aux_orientation(aux_info) -> int | None:
    """The ``Orientation`` of the aux data's description dictionary, if any."""
    description = aux_info.get("kCGImageAuxiliaryDataInfoDataDescription")
    if description is None:
        return None
    orientation = description.get("Orientation")
    return None if orientation is None else int(orientation)


def _check_orientations_agree(exif_orientation, aux_orientation, path_str):
    if (
        exif_orientation is not None
        and aux_orientation is not None
        and exif_orientation != aux_orientation
    ):
        raise AppleDepthDecodeError(
            f"{path_str!r}: the depth aux data's Orientation ({aux_orientation}) "
            f"disagrees with the primary image's EXIF Orientation "
            f"({exif_orientation}); refusing to guess how the depth map aligns"
        )


def _read_apple_depth(Quartz, AVFoundation, NSURL, path_str) -> AppleDepthData | None:
    url = NSURL.fileURLWithPath_(path_str)
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    if source is None:
        raise AppleDepthDecodeError(f"ImageIO could not open {path_str!r} as an image source")

    aux_info = None
    source_type = None
    for candidate_type, aux_type_constant in (
        ("disparity", Quartz.kCGImageAuxiliaryDataTypeDisparity),
        ("depth", Quartz.kCGImageAuxiliaryDataTypeDepth),
    ):
        aux_info = Quartz.CGImageSourceCopyAuxiliaryDataInfoAtIndex(
            source, 0, aux_type_constant
        )
        if aux_info is not None:
            source_type = candidate_type
            break

    if aux_info is None:
        return None

    depth_data, error = AVFoundation.AVDepthData.depthDataFromDictionaryRepresentation_error_(
        aux_info, None
    )
    if depth_data is None:
        raise AppleDepthDecodeError(
            f"AVDepthData could not decode the {source_type} aux image in "
            f"{path_str!r}: {error}"
        )

    depth_data = depth_data.depthDataByConvertingToDepthDataType_(
        Quartz.kCVPixelFormatType_DepthFloat32
    )
    if depth_data is None:
        raise AppleDepthDecodeError(
            f"AVDepthData could not convert the {source_type} aux image in "
            f"{path_str!r} to DepthFloat32"
        )

    depth_m = _pixel_buffer_to_depth_m(Quartz, depth_data.depthDataMap())
    accuracy = _accuracy_name(AVFoundation, depth_data.depthDataAccuracy())
    quality = _quality_name(AVFoundation, depth_data.depthDataQuality())
    filtered = bool(depth_data.isDepthDataFiltered())

    intrinsics = None
    intrinsics_reference_size = None
    lens_distortion_center = None
    calibration = depth_data.cameraCalibrationData()
    if calibration is not None:
        intrinsics, intrinsics_reference_size, lens_distortion_center = (
            _calibration_to_intrinsics(calibration)
        )

    exif_orientation = _read_exif_orientation(Quartz, source)
    aux_orientation = _aux_orientation(aux_info)
    _check_orientations_agree(exif_orientation, aux_orientation, path_str)

    return AppleDepthData(
        depth_m=depth_m,
        accuracy=accuracy,
        filtered=filtered,
        quality=quality,
        source_type=source_type,
        intrinsics=intrinsics,
        intrinsics_reference_size=intrinsics_reference_size,
        lens_distortion_center=lens_distortion_center,
        exif_orientation=exif_orientation,
        aux_orientation=aux_orientation,
    )


def encode_depth_as_disparity_8bit(
    depth_m: np.ndarray, far_cap_m: float = DISPARITY_FAR_CAP_M
) -> tuple[Image.Image, float, float]:
    """Encode a metric depth map as the Camera-app style 8-bit disparity image.

    The rest of the library works on an 8-bit ``L`` disparity image plus
    ``FloatMinValue``/``FloatMaxValue`` (see
    :func:`portrait_analyser.incisor.depth_raw_to_distance_cm`, which decodes
    ``disparity = float_max * v / 255 + float_min * (1 - v / 255)`` and
    ``Z_cm = 100 / disparity``). This produces exactly that representation:

    * ``float_max = 1 / Z_near`` with ``Z_near`` the
      :data:`NEAR_PERCENTILE` percentile of valid depths (robust against a
      single stray near pixel; nearer pixels saturate at code 255);
    * ``float_min = 1 / Z_far`` with ``Z_far = min(largest valid depth,
      far_cap_m)``;
    * ``v = round(255 * (1/Z - float_min) / (float_max - float_min))``.

    Pixels that are ``NaN``, ``<= 0`` or farther than ``far_cap_m`` encode as
    ``0``, meaning "no depth". NOTE: ``depth_raw_to_distance_cm(0, ...)``
    still returns ``Z_far`` -- in Camera-app maps code 0 is simply the
    farthest depth, so the plain conversion cannot know. Callers must reject
    code 0 themselves for these maps: the ``camera=`` aware functions
    (``compute_incisor_distance_3d``, ``compute_tmd_3d``,
    ``raw_depth_to_distance_cm``), ``sample_depth_at_point(...,
    invalid_value=0)``, ``bilinear_sample(..., invalid_value=0)`` and the neck
    samplers do; ``IOSPortrait.depth_valid_mask`` exposes it. Valid pixels are
    clamped to ``1..255`` so ``0`` stays reserved for invalid ones: a valid
    pixel within half a step of ``Z_far`` is reported one quantisation step
    nearer.

    Quantisation: one code step is ``(float_max - float_min) / 255`` in
    disparity, i.e. ``~Z**2 * step`` in depth -- about 1.1-1.3 mm at 30 cm
    and 3-3.6 mm at 50 cm for the ranges seen on real captures
    (``Z_near`` 0.25-0.30 m); the round-trip error is at most half of that
    (one step for the clamped near-``Z_far`` pixels). The full-precision map
    should be kept alongside for anything that needs better.

    :param depth_m: 2-D float array of depths in metres, ``NaN`` = invalid
    :param far_cap_m: far limit in metres, see :data:`DISPARITY_FAR_CAP_M`
    :returns: ``(image, float_min, float_max)``
    :raises ValueError: when ``depth_m`` has no valid pixel within the cap
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be a 2-D array, got shape {depth.shape}")
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > 0) & (depth <= far_cap_m)
    if not valid.any():
        raise ValueError(
            f"depth map has no valid pixel between 0 and {far_cap_m} m to encode"
        )

    # A robust near end: one stray near pixel must not stretch the code range
    # (and so coarsen the quantisation) for the whole map. Pixels nearer than
    # this saturate at code 255 (they decode as Z_near).
    z_near = float(np.percentile(depth[valid], NEAR_PERCENTILE))
    z_far = min(float(depth[valid].max()), float(far_cap_m))
    float_max = 1.0 / z_near
    # A perfectly flat map has no range to spread; with float_min = 0 every
    # valid pixel encodes as 255 and decodes back to exactly Z_near.
    float_min = 1.0 / z_far if z_far > z_near else 0.0

    codes = np.zeros(depth.shape, dtype=np.uint8)
    disparity = 1.0 / depth[valid]
    scaled = np.rint(255.0 * (disparity - float_min) / (float_max - float_min))
    # np.clip also saturates pixels nearer than the robust Z_near at 255.
    codes[valid] = np.clip(scaled, 1, 255).astype(np.uint8)
    return Image.fromarray(codes), float_min, float_max
