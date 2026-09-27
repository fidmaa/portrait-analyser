import logging
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Union

import numpy as np
import piexif
import pyheif
from PIL import Image, ImageDraw

from . import const
from .apple_depth import MAX_PLAUSIBLE_DEPTH_M, read_apple_depth
from .camera import CameraModel, intrinsics_in_photo_space, rotate_by_exif_orientation
from .depth_map import FloatDepthMap, LegacyDepthMap
from .exceptions import (
    AppleDepthDecodeError,
    AppleDepthUnavailable,
    ExifValidationFailed,
    NoDepthMapFound,
    UnknownExtension,
)
from .face import (
    TEETH_MEASUREMENT_MIN_HEIGHT_FRACTION,
    IncisorMeasurement,
    detect_teeth_arches,
    find_incisor_centroids,
    find_incisor_distance_teeth,
    teeth_threshold,
)
from .incisor import depth_raw_to_distance_cm, distance_3d_from_cm

logger = logging.getLogger(__name__)

# Value used to test the binary teeth mask image (teeth 255, background 0).
MASK_ON = 128

# Facing incisal edges lie roughly in one coronal plane (overjet and head tilt
# give a few mm, <= ~1.1 cm on the labelled data).  A weak arch whose edge
# depth is further than this from the strong arch's was sampled on the
# depth discontinuity into the mouth cavity (seen at ~5.6 cm on real
# captures): the TrueDepth map does not resolve the thin, barely visible
# teeth, so the strong arch's depth is used for both edges instead.
MAX_INCISAL_DEPTH_DIFFERENCE_CM = 2.0


def _reconcile_weak_arch_depth(upper_raw, lower_raw, weak_side, float_min, float_max):
    """Return ``(upper_raw, lower_raw, depth_assumed_side)``.

    When the weak arch's depth sample disagrees implausibly with the strong
    arch's, replace it with the strong arch's sample and report which side's
    depth was assumed.  Otherwise return the samples unchanged.
    """
    if (
        weak_side is None
        or upper_raw is None
        or lower_raw is None
        or float_min is None
        or float_max is None
    ):
        return upper_raw, lower_raw, None
    upper_cm = depth_raw_to_distance_cm(upper_raw, float(float_min), float(float_max))
    lower_cm = depth_raw_to_distance_cm(lower_raw, float(float_min), float(float_max))
    if (
        upper_cm is not None
        and lower_cm is not None
        and abs(upper_cm - lower_cm) <= MAX_INCISAL_DEPTH_DIFFERENCE_CM
    ):
        return upper_raw, lower_raw, None
    if weak_side == "upper":
        return lower_raw, lower_raw, "upper"
    return upper_raw, upper_raw, "lower"


def _reconcile_weak_arch_samples(upper, lower, weak_side, calibrated=True):
    """:func:`_reconcile_weak_arch_depth` for ``DepthSample`` objects.

    Compares the samples' distances (for legacy 8-bit maps exactly the
    conversion the raw-code version uses) and, when the weak arch's depth is
    implausible, copies the strong arch's whole sample over it.
    """
    if weak_side is None or upper is None or lower is None or not calibrated:
        # Like _reconcile_weak_arch_depth without float_min/float_max: no
        # distances to compare, so nothing is replaced.
        return upper, lower, None
    upper_cm = upper.distance_cm
    lower_cm = lower.distance_cm
    if (
        upper_cm is not None
        and lower_cm is not None
        and abs(upper_cm - lower_cm) <= MAX_INCISAL_DEPTH_DIFFERENCE_CM
    ):
        return upper, lower, None
    if weak_side == "upper":
        return lower, lower, "upper"
    return upper, upper, "lower"


class IOSPortrait:
    """A loaded portrait photo with its depth map and semantic mattes.

    ``photo`` is RGB, upright. **Measure depth through** :attr:`depth`, a
    :class:`~portrait_analyser.depth_map.DepthMap`: for Camera-app files it
    wraps the 8-bit map exactly as before, for capture-app files it samples
    the full-precision float map ``depth_m``.

    ``depthmap`` is an 8-bit disparity image decoded with
    ``floatValueMin``/``floatValueMax`` via ``depth_raw_to_distance_cm``; its
    PIL mode differs by source: Camera-app files give mode ``"RGB"`` (use
    channel 0; e.g. 576x768 or 480x640) and it *is* their depth data.
    Capture-app files give mode ``"L"`` (e.g. 480x640 / 640x480) and it is a
    **display-only** re-encoding of ``depth_m`` (quantised, 3 m far cap) that
    no library measurement reads. Code 0 is a valid farthest depth in the
    former and "no depth" in the latter -- see
    :attr:`depth_code_zero_is_invalid` / :attr:`depth_valid_mask`.
    ``teethmap``/``skinmap``/``hairmap`` are ``"L"`` at photo size (or None).
    """

    def __init__(
        self,
        photo,
        depthmap=None,
        teethmap=None,
        skinmap=None,
        hairmap=None,
        floatValueMin=None,
        floatValueMax=None,
        teeth_bbox=None,
        incisor_distance=None,
        incisor_distance_3d_mm=None,
        incisor_measurement=None,
        teeth_threshold=None,
        teeth_arches=None,
        depth_m=None,
        depth_accuracy=None,
        depth_filtered=None,
        focal_length_px=None,
        principal_point_px=None,
        depth_plausible=None,
        depth=None,
        depth_repaired=None,
        depth_repair_check=None,
    ):
        self.photo = photo
        self.depthmap = depthmap
        self.teethmap = teethmap
        self.skinmap = skinmap
        self.hairmap = hairmap
        self.teeth_bbox = teeth_bbox
        self.incisor_distance = incisor_distance
        self.incisor_distance_3d_mm = incisor_distance_3d_mm
        self.incisor_measurement = incisor_measurement
        # Adaptive teeth-matte confidence threshold used for the detection
        # (pixels >= this value were treated as teeth), or None without matte.
        self.teeth_threshold = teeth_threshold
        # TeethArches (binary teeth mask, weak-arch side/threshold) or None.
        self.teeth_arches = teeth_arches
        self.floatValueMin = float(floatValueMin) if floatValueMin is not None else None
        self.floatValueMax = float(floatValueMax) if floatValueMax is not None else None
        # Full-precision depth in metres (float32, NaN = invalid), rotated to
        # the upright photo's orientation, at the depth sensor's resolution.
        # Only set for files whose depth was read via the macOS
        # ImageIO/AVFoundation reader (the 16-bit capture-app format), after
        # any repair (``depth_repaired``). ``depthmap`` is its display-only
        # 8-bit re-encoding; measurements sample this map via ``depth``.
        self.depth_m = depth_m
        # AVDepthData accuracy: "absolute", "relative", or None if unknown
        # (not macOS / pyobjc unavailable). "relative" depth (iPhone 17 Pro
        # Camera app) is known to underestimate distance by 7-28 %.
        self.depth_accuracy = depth_accuracy
        # AVDepthData isDepthDataFiltered, or None if unknown.
        self.depth_filtered = depth_filtered
        # File intrinsics in pixels of ``photo`` (upright): (fx, fy) and
        # (cx, cy). Only set for the capture-app format; legacy Camera-app
        # files keep None and are measured with the calibration polynomial.
        self.focal_length_px = focal_length_px
        self.principal_point_px = principal_point_px
        # Sanity check of capture-app depth against the skin/hair mattes:
        # True = face depth plausible (15-100 cm and nearer than the
        # background, possibly after the reciprocal repair -- see
        # ``depth_repaired``), False = implausible (e.g. inverted depth that
        # could not be repaired unambiguously) or depth not
        # alignable with the photo -- load_image then skips its automatic
        # 3-D measurements and callers should not measure either; None = not
        # checked (Camera-app files, or no skin matte).
        self.depth_plausible = depth_plausible
        # How the depth map was repaired on load, or None (never for
        # Camera-app files). See DEPTH_REPAIRED_RECIPROCAL.
        self.depth_repaired = depth_repaired
        # DepthRepairCheck when the repair was evaluated (absolute
        # capture-app depth that failed the plausibility check), else None.
        self.depth_repair_check = depth_repair_check
        self._depth = depth

    @property
    def depth(self):
        """The portrait's :class:`~portrait_analyser.depth_map.DepthMap`, or None.

        ``FloatDepthMap`` over ``depth_m`` for capture-app files,
        ``LegacyDepthMap`` over ``depthmap`` + ``floatValueMin``/``Max`` for
        Camera-app files; None without depth. Built by ``load_image`` (or on
        first access for hand-made portraits). Check
        ``depth_plausible is not False`` before measuring.
        """
        if self._depth is None:
            if self.photo is None:
                return None
            if self.depth_m is not None:
                self._depth = FloatDepthMap(self.depth_m, self.photo.size)
            elif self.depthmap is not None:
                self._depth = LegacyDepthMap(
                    self.depthmap,
                    self.floatValueMin,
                    self.floatValueMax,
                    self.photo.size,
                    zero_is_invalid=False,
                )
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value

    @property
    def depth_code_zero_is_invalid(self):
        """Meaning of raw ``depthmap`` code 0.

        ``True`` for capture-app files (``depth_m`` set): 0 = "no depth"
        (NaN, or beyond ``DISPARITY_FAR_CAP_M``) and must never be converted
        to a distance -- ``depth_raw_to_distance_cm(0, ...)`` would return
        ``Z_far``. ``False`` for Camera-app files: 0 is a real, farthest
        depth, as it always was.
        """
        return self.depth_m is not None

    @property
    def depth_valid_mask(self):
        """Boolean array (depthmap rows x cols) of 8-bit ``depthmap`` codes
        holding depth.

        Only for capture-app files (``depthmap != 0``); ``None`` for
        Camera-app files, where every code is a valid depth. This describes the
        display image (pixels beyond its 3 m cap are 0); for the measurement
        map use ``portrait.depth.valid_mask``.
        """
        if not self.depth_code_zero_is_invalid or self.depthmap is None:
            return None
        return np.asarray(self.depthmap) > 0

    @property
    def camera(self):
        """:class:`~portrait_analyser.camera.CameraModel` from the file intrinsics, or None.

        Only for capture-app files with *absolute*, plausible depth whose
        depth map could be aligned with the photo; ``None`` otherwise (all
        Camera-app files, relative depth, ``depth_plausible is False``),
        which keeps the calibration polynomial.

        Pass it as ``camera=`` to the metric functions (``pixel_to_mm`` via
        ``focal_px``/``principal_px``, ``compute_incisor_distance_3d``,
        ``measure_filtered_surface_length``, ...). ``None`` for legacy
        Camera-app files, which keeps the calibration polynomial.
        """
        return CameraModel.from_portrait(self)

    def teeth_bbox_translated(self, max_wi, max_he):
        if self.teeth_bbox is None:
            return
        x, y, wi, he = self.teeth_bbox
        return (
            x * max_wi / self.teethmap.size[0],
            y * max_he / self.teethmap.size[1],
            wi * max_wi / self.teethmap.size[0],
            he * max_he / self.teethmap.size[1],
        )


def _validate_exif(primary_image, use_exif):
    """Extract and validate TrueDepth EXIF metadata."""
    for exif_metadata in [
        metadata
        for metadata in primary_image.image.load().metadata
        if metadata.get("type", "") == "Exif"
    ]:
        exif = piexif.load(exif_metadata["data"])
        if use_exif:
            check_exif_data(exif)


def _parse_depth_metadata(depth_image):
    """Parse XML metadata from depth image for float min/max values."""
    ret = {}
    if depth_image.metadata:
        for metadata in depth_image.metadata:
            if metadata.get("type", "") == "mime":
                root = ET.fromstring(metadata.get("data"))
                for elem in root[0][0]:
                    ret[elem.tag] = elem.text

    float_min = ret.get(
        "{http://ns.apple.com/pixeldatainfo/1.0/}FloatMinValue", 0.0
    )
    float_max = ret.get(
        "{http://ns.apple.com/pixeldatainfo/1.0/}FloatMaxValue", 0.0
    )
    return float_min, float_max


def _decode_picture(raw_image):
    """Decode primary picture with device-specific dimension correction."""
    try:
        # iPhone 14
        return Image.frombytes(
            raw_image.mode,
            (raw_image.size[0] + 4, raw_image.size[1] - 1),
            raw_image.data,
        )
    except ValueError:
        # iPhone 12
        return Image.frombytes(
            raw_image.mode,
            (raw_image.size[0], raw_image.size[1]),
            raw_image.data,
        )


def _decode_semantic_map(raw_image):
    """Decode a semantic segmentation map (teeth/skin) using the Apple format.

    Camera-app mattes are decoded with the historical layout (RGB rows
    padded by 14 bytes, read as a 3x-wide "L" image that the caller resizes
    to the photo). When that layout does not fit -- capture-app files use
    other row paddings (e.g. none, or 8 bytes) -- the matte is decoded
    properly using the row stride and its first channel is returned.
    """
    loaded = raw_image.load()
    try:
        return Image.frombytes(
            "L",
            (loaded.size[0] * 3 + 14, loaded.size[1] - 1),
            loaded.data,
        )
    except ValueError:
        logger.debug(
            "semantic matte %s (stride %s) does not fit the Camera-app layout; "
            "decoding it by stride",
            loaded.size,
            loaded.stride,
            exc_info=True,
        )
    try:
        decoded = Image.frombytes(
            loaded.mode, loaded.size, loaded.data, "raw", loaded.mode, loaded.stride
        )
    except ValueError:
        logger.warning(
            "could not decode semantic matte %s mode %s stride %s",
            loaded.size,
            loaded.mode,
            loaded.stride,
            exc_info=True,
        )
        return None
    if len(decoded.getbands()) > 1:
        return decoded.getchannel(0)
    return decoded


def _load_pyheif_depth(primary_image):
    """Decode the depth aux image with pyheif (raises ``HeifError`` on 16-bit)."""
    return primary_image.depth_image.image.load()


def _read_apple_depth_info(fileName):
    """Best-effort AVDepthData metadata for a file pyheif decoded fine.

    Returns the :class:`AppleDepthData` or None when the macOS reader is not
    available here / fails; failures are logged, never raised, because the
    legacy pyheif path does not depend on them.
    """
    if sys.platform != "darwin":
        return None
    try:
        return read_apple_depth(fileName)
    except AppleDepthUnavailable:
        logger.info("macOS depth reader unavailable; depth accuracy unknown", exc_info=True)
    except AppleDepthDecodeError:
        logger.warning(
            "macOS depth reader failed on %s; depth accuracy unknown", fileName, exc_info=True
        )
    return None


def _load_depth_via_apple(fileName, pyheif_error):
    """Read a depth map pyheif could not decode (16-bit disparity) via macOS.

    :raises NoDepthMapFound: when the macOS reader is unavailable, fails or
        finds no depth -- chained to the underlying error.
    """
    try:
        apple = read_apple_depth(fileName)
    except AppleDepthUnavailable as exc:
        raise NoDepthMapFound(
            f"{fileName}: pyheif cannot decode the depth map ({pyheif_error}); "
            "this format (16-bit disparity) needs the macOS ImageIO/AVFoundation "
            f"reader, which is unavailable here: {exc}"
        ) from exc
    except AppleDepthDecodeError as exc:
        raise NoDepthMapFound(
            f"{fileName}: neither pyheif ({pyheif_error}) nor the macOS reader "
            f"could decode the depth map: {exc}"
        ) from exc
    if apple is None:
        raise NoDepthMapFound(
            f"{fileName}: pyheif cannot decode the depth map ({pyheif_error}) and "
            "ImageIO finds no depth/disparity data"
        ) from pyheif_error
    return apple


def _apple_depth_for_photo(apple, fileName, photo_size):
    """Upright depth (metres), photo intrinsics, alignment.

    The depth map comes in sensor orientation; the photo/mattes are already
    upright, so only the depth is rotated -- by the aux data's own
    Orientation (``AppleDepthData.depth_orientation``, EXIF as fallback).
    Returns ``(depth_m, focal, principal, aligned)``; when the orientation is
    unknown or the rotated map's aspect does not match the photo, ``aligned``
    is False and no intrinsics are returned.
    """
    orientation = apple.depth_orientation
    depth_m = rotate_by_exif_orientation(apple.depth_m, orientation)
    depth_h, depth_w = depth_m.shape
    photo_w, photo_h = photo_size
    aligned = True
    if orientation is None:
        logger.warning(
            "%s: the depth map's orientation is unknown (no aux or EXIF "
            "Orientation); not measuring with it",
            fileName,
        )
        aligned = False
    elif abs(depth_w / depth_h - photo_w / photo_h) > 0.01 * (photo_w / photo_h):
        logger.warning(
            "%s: rotated depth map %sx%s does not match the photo aspect %sx%s "
            "(orientation %s); depth is misaligned, not measuring with it",
            fileName,
            depth_w,
            depth_h,
            photo_w,
            photo_h,
            orientation,
        )
        aligned = False

    focal = principal = None
    if aligned and apple.intrinsics is not None and apple.intrinsics_reference_size is not None:
        sensor_h, sensor_w = apple.depth_m.shape
        focal, principal = intrinsics_in_photo_space(
            apple.intrinsics,
            apple.intrinsics_reference_size,
            (sensor_w, sensor_h),
            orientation,
            photo_size,
        )
    return np.ascontiguousarray(depth_m), focal, principal, aligned


# Plausible camera-to-face distance for the depth sanity check, cm.
PLAUSIBLE_FACE_DEPTH_CM = (15.0, 100.0)
# Matte confidence (0-255) counted as skin / hair for the sanity check.
_PLAUSIBILITY_MATTE_THRESHOLD = 128


def check_depth_plausibility(depth_m, skinmap, hairmap=None):
    """Sanity-check an upright metric depth map against the semantic mattes.

    The skin matte's median depth must lie within
    :data:`PLAUSIBLE_FACE_DEPTH_CM` and be nearer than the median depth of
    pixels that are neither skin nor hair (the background). Catches e.g. a
    capture whose depth came out inverted (face at ~2.7 m, background at
    ~0.6 m). Never "fixes" the map itself -- see :func:`repair_inverted_depth`.

    :param depth_m: ``(h, w)`` metres, NaN = invalid, or a
        :class:`~portrait_analyser.depth_map.FloatDepthMap`
    :param skinmap: PIL "L" skin matte (any size; resized to the depth map)
    :param hairmap: optional PIL "L" hair matte
    :returns: ``(plausible, skin_median_cm, background_median_cm)``;
        ``plausible`` is None when it cannot be judged (no skin matte / no
        valid depth on skin)
    """
    if skinmap is None or depth_m is None:
        return None, None, None
    if isinstance(depth_m, FloatDepthMap):
        depth_m = depth_m.depth_m
    height, width = depth_m.shape
    skin = np.asarray(skinmap.convert("L").resize((width, height))) >= (
        _PLAUSIBILITY_MATTE_THRESHOLD
    )
    person = skin.copy()
    if hairmap is not None:
        person |= np.asarray(hairmap.convert("L").resize((width, height))) >= (
            _PLAUSIBILITY_MATTE_THRESHOLD
        )
    skin_depths = depth_m[skin & np.isfinite(depth_m)]
    if skin_depths.size == 0:
        return None, None, None
    skin_cm = float(np.median(skin_depths)) * 100.0
    background = depth_m[~person & np.isfinite(depth_m)]
    background_cm = float(np.median(background)) * 100.0 if background.size else None
    low, high = PLAUSIBLE_FACE_DEPTH_CM
    plausible = low <= skin_cm <= high and (background_cm is None or skin_cm < background_cm)
    return plausible, skin_cm, background_cm


# ``IOSPortrait.depth_repaired`` value for a map repaired by taking 1/depth.
# Root cause (confirmed by the capture app's authors): iOS 26 on iPhone 17's
# front TrueDepth camera writes HEIC depth labelled "disparity" whose values
# are distances in metres; read as disparity it becomes 1/true_depth.
DEPTH_REPAIRED_RECIPROCAL = "metres stored under a disparity label (iOS 26)"

# Physical cross-check with the file intrinsics: the face's width (minor axis
# of the skin matte's largest blob) must be a human face width at the
# repaired depth and must not be one at the depth as read. Real captures:
# 15.8-16.6 cm at the correct depth, 96-125 cm at its reciprocal.
PLAUSIBLE_FACE_WIDTH_CM = (10.0, 25.0)
# The face-width blob analysis runs on the matte resized to 1/4 of the photo.
_FACE_WIDTH_DOWNSCALE = 4


@dataclass
class DepthRepairCheck:
    """Evidence behind :func:`repair_inverted_depth`'s decision.

    Distances in cm; widths computed as ``face_width_px * Z / focal_px`` at
    the skin median depth of each interpretation. Fields are None when that
    step was not reached.
    """

    repaired: bool
    reason: str
    as_read_skin_cm: float | None = None
    as_read_background_cm: float | None = None
    reciprocal_plausible: bool | None = None
    reciprocal_skin_cm: float | None = None
    reciprocal_background_cm: float | None = None
    face_width_px: float | None = None
    as_read_face_width_cm: float | None = None
    reciprocal_face_width_cm: float | None = None


def face_width_px(skinmap, photo_size=None):
    """Width of the face in photo pixels from the skin matte, or None.

    The minor axis (4 standard deviations) of the largest 8-connected blob of
    skin pixels (matte >= 128). Orientation-free, so it also works for faces
    lying sideways in a landscape frame, and not inflated by shoulders or
    hands the way whole skin rows are.

    :param skinmap: PIL "L" skin matte at any size
    :param photo_size: ``(width, height)`` of the photo; the matte is first
        resized to (a quarter of) it, since raw mattes need not share the
        photo's aspect. Default: the matte's own pixels.
    """
    import cv2

    matte = skinmap.convert("L")
    scale = 1.0
    if photo_size is not None:
        work = (
            max(1, photo_size[0] // _FACE_WIDTH_DOWNSCALE),
            max(1, photo_size[1] // _FACE_WIDTH_DOWNSCALE),
        )
        matte = matte.resize(work, Image.Resampling.BILINEAR)
        scale = photo_size[0] / work[0]
    skin = (np.asarray(matte) >= _PLAUSIBILITY_MATTE_THRESHOLD).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == largest)
    if xs.size < 3:
        return None
    minor_variance = float(np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(np.float64)))[0])
    return 4.0 * math.sqrt(max(minor_variance, 0.0)) * scale


def _reciprocal_depth(depth_m):
    with np.errstate(divide="ignore", invalid="ignore"):
        reciprocal = (1.0 / np.asarray(depth_m)).astype(np.float32)
        invalid = (
            ~np.isfinite(reciprocal) | (reciprocal <= 0) | (reciprocal > MAX_PLAUSIBLE_DEPTH_M)
        )
    reciprocal[invalid] = np.nan
    return reciprocal


def _width_ok(width_cm):
    low, high = PLAUSIBLE_FACE_WIDTH_CM
    return width_cm is not None and low <= width_cm <= high


def _interpretation_ok(skin_cm, width_cm):
    low, high = PLAUSIBLE_FACE_DEPTH_CM
    return skin_cm is not None and low <= skin_cm <= high and _width_ok(width_cm)


def repair_inverted_depth(depth_m, skinmap, hairmap=None, *, focal_px=None, photo_size=None):
    """Repair capture-app depth written as metres under a disparity label.

    iOS 26 on iPhone 17's front TrueDepth camera writes HEIC depth labelled
    "disparity" whose values are metres, so reading it as disparity yields
    ``1 / depth`` (face at ~2.4-2.9 m). Such a map fails
    :func:`check_depth_plausibility`.

    Both interpretations -- the map as read and its reciprocal -- are judged
    on the face alone, with the file's focal length:

    * the face (skin-matte median depth) must lie within
      :data:`PLAUSIBLE_FACE_DEPTH_CM`, and
    * the face width (:func:`face_width_px`, converted at that depth) must be
      a human face width (:data:`PLAUSIBLE_FACE_WIDTH_CM`).

    The reciprocal is returned only when the map as read fails
    :func:`check_depth_plausibility`, the reciprocal interpretation passes
    both face tests and the as-read one does not. Ambiguous (both pass) or
    hopeless (neither passes) maps are left alone, as is everything without
    ``focal_px`` or a face blob.

    The background comparison of :func:`check_depth_plausibility` is only
    recorded, never a veto: an object held in front of the face (e.g. a
    calibration board, IMG_2376) makes "everything that is not skin or hair"
    nearer than the face even in a correct map. A real far subject behind a
    near foreground is refused by the face-width test instead (its face is a
    few centimetres wide at the reciprocal depth).

    Never apply this to Camera-app (8-bit) depth.

    :param depth_m: upright metres, NaN = invalid
    :param skinmap: skin matte (any size)
    :param focal_px: horizontal focal length in photo pixels
    :param photo_size: ``(width, height)`` of the photo ``focal_px`` refers
        to (default: the matte's own size)
    :returns: ``(repaired_depth_m or None, DepthRepairCheck)``
    """
    plausible, skin_cm, background_cm = check_depth_plausibility(depth_m, skinmap, hairmap)
    check = DepthRepairCheck(
        repaired=False,
        reason="",
        as_read_skin_cm=skin_cm,
        as_read_background_cm=background_cm,
    )
    if plausible is not False:
        check.reason = "depth as read is not implausible"
        return None, check
    reciprocal = _reciprocal_depth(depth_m)
    r_plausible, r_skin_cm, r_background_cm = check_depth_plausibility(
        reciprocal, skinmap, hairmap
    )
    check.reciprocal_plausible = r_plausible
    check.reciprocal_skin_cm = r_skin_cm
    check.reciprocal_background_cm = r_background_cm
    if focal_px is None:
        check.reason = "no focal length for the face-width check"
        return None, check
    width_px = face_width_px(skinmap, photo_size)
    check.face_width_px = width_px
    if width_px is None or r_skin_cm is None:
        check.reason = "no face blob in the skin matte"
        return None, check
    check.as_read_face_width_cm = width_px * skin_cm / focal_px
    check.reciprocal_face_width_cm = width_px * r_skin_cm / focal_px
    as_read_ok = _interpretation_ok(skin_cm, check.as_read_face_width_cm)
    reciprocal_ok = _interpretation_ok(r_skin_cm, check.reciprocal_face_width_cm)
    if as_read_ok and reciprocal_ok:
        check.reason = "face distance and width are plausible both ways (ambiguous)"
        return None, check
    if as_read_ok:
        check.reason = "face distance and width are plausible as read; not repaired"
        return None, check
    if not reciprocal_ok:
        check.reason = "face distance or width is implausible both ways"
        return None, check
    check.repaired = True
    check.reason = "reciprocal face distance and width are plausible, as read they are not"
    return reciprocal, check


def load_image(fileName: str, use_exif=True) -> Union[IOSPortrait, None]:
    """Load HEIC/HEIF with depth data, return an IOSPortrait instance."""
    if not (fileName.lower().endswith("heic") or fileName.lower().endswith("heif")):
        raise UnknownExtension(
            "only supported extensions for filenames are: HEIF, HEIC"
        )

    with open(fileName, "rb") as f:
        heif_container = pyheif.open_container(f)

        primary_image = heif_container.primary_image
        _validate_exif(primary_image, use_exif)

        if primary_image.depth_image is None:
            raise NoDepthMapFound(f"{fileName} has no depth data")

        # Extract auxiliary semantic maps
        teeth_raw = skin_raw = hair_raw = None
        for aux in primary_image.auxiliary_images:
            aux_type = getattr(aux, "type", "")
            if aux_type == "urn:com:apple:photo:2019:aux:semanticteethmatte":
                teeth_raw = aux.image
            elif aux_type == "urn:com:apple:photo:2019:aux:semanticskinmatte":
                skin_raw = aux.image
            elif aux_type == "urn:com:apple:photo:2019:aux:semantichairmatte":
                hair_raw = aux.image

        # Decode depth map. Camera-app files carry 8-bit disparity pyheif can
        # decode; the TrueDepth capture app stores 16-bit disparity, which
        # pyheif rejects ("Unsupported JPEG data precision 16") -- those are
        # read via macOS ImageIO/AVFoundation instead (see apple_depth.py).
        apple_depth = None
        try:
            depth_loaded = _load_pyheif_depth(primary_image)
        except pyheif.error.HeifError as exc:
            logger.info(
                "%s: pyheif cannot decode the depth map (%s); using the macOS reader",
                fileName,
                exc,
            )
            apple_depth = _load_depth_via_apple(fileName, exc)

        depth_m = depth_accuracy = depth_filtered = None
        depth_image = float_min = float_max = None
        depth_aligned = True
        focal_length_px = principal_point_px = None
        if apple_depth is None:
            float_min, float_max = _parse_depth_metadata(depth_loaded)
            depth_image = Image.frombytes(
                depth_loaded.mode, depth_loaded.size, depth_loaded.data
            )
            depth_info = _read_apple_depth_info(fileName)
            if depth_info is not None:
                depth_accuracy = depth_info.accuracy
                depth_filtered = depth_info.filtered

        # Decode primary picture
        picture_image = _decode_picture(primary_image.image.load())

        if apple_depth is not None:
            depth_accuracy = apple_depth.accuracy
            depth_filtered = apple_depth.filtered
            depth_m, focal_length_px, principal_point_px, depth_aligned = (
                _apple_depth_for_photo(apple_depth, fileName, picture_image.size)
            )

        # Decode semantic maps
        teeth_image = _decode_semantic_map(teeth_raw) if teeth_raw else None
        skin_image = _decode_semantic_map(skin_raw) if skin_raw else None
        hair_image = _decode_semantic_map(hair_raw) if hair_raw else None

    # Capture-app depth: sanity checks (and the inverted-depth repair) before
    # anything is measured with it.
    depth_plausible = None
    depth_repaired = None
    depth_repair_check = None
    if depth_m is not None:
        if not depth_aligned:
            depth_plausible = False
        else:
            depth_plausible, skin_cm, background_cm = check_depth_plausibility(
                depth_m, skin_image, hair_image
            )
            if depth_plausible is False and depth_accuracy == "absolute":
                repaired, depth_repair_check = repair_inverted_depth(
                    depth_m,
                    skin_image,
                    hair_image,
                    focal_px=None if focal_length_px is None else focal_length_px[0],
                    photo_size=picture_image.size,
                )
                if repaired is None:
                    logger.info(
                        "%s: inverted-depth repair not applied: %s",
                        fileName,
                        depth_repair_check.reason,
                    )
                else:
                    c = depth_repair_check
                    logger.warning(
                        "%s: depth looks inverted (skin median %.1f cm, background "
                        "median %s cm, face width %.1f cm); its reciprocal is "
                        "plausible (skin %.1f cm, background %s cm, face width "
                        "%.1f cm) -- repaired as depth stored as disparity",
                        fileName,
                        skin_cm,
                        "n/a" if background_cm is None else f"{background_cm:.1f}",
                        c.as_read_face_width_cm,
                        c.reciprocal_skin_cm,
                        "n/a"
                        if c.reciprocal_background_cm is None
                        else f"{c.reciprocal_background_cm:.1f}",
                        c.reciprocal_face_width_cm,
                    )
                    depth_m = repaired
                    depth_repaired = DEPTH_REPAIRED_RECIPROCAL
                    depth_plausible = True
            if depth_plausible is False:
                logger.warning(
                    "%s: implausible depth (skin median %.1f cm, background median "
                    "%s cm; expected %s-%s cm and nearer than the background) -- "
                    "possibly inverted; skipping automatic 3-D measurements",
                    fileName,
                    skin_cm,
                    "n/a" if background_cm is None else f"{background_cm:.1f}",
                    *PLAUSIBLE_FACE_DEPTH_CM,
                )
        if depth_accuracy != "absolute":
            logger.warning(
                "%s: depth accuracy is %r, not absolute; measuring with the "
                "calibration polynomial, not the file intrinsics",
                fileName,
                depth_accuracy,
            )
        # Every measurement samples this float map; the 8-bit image is only
        # for display (and for pre-0.8 callers of depthmap/floatValueMin/Max).
        depth = FloatDepthMap(depth_m, picture_image.size)
        try:
            depth_image, float_min, float_max = depth.display_encoding()
        except ValueError as exc:
            raise NoDepthMapFound(f"{fileName}: depth map has no usable pixels: {exc}") from exc
    else:
        depth = LegacyDepthMap(
            depth_image, float_min, float_max, picture_image.size, zero_is_invalid=False
        )
    measure_depth = depth_plausible is not False
    depth_calibrated = depth.is_float or (
        depth.float_min is not None and depth.float_max is not None
    )

    # File intrinsics: only capture-app files with absolute, aligned depth.
    # None = calibration polynomial.
    camera = (
        CameraModel(
            *focal_length_px,
            *principal_point_px,
            width=picture_image.size[0],
            height=picture_image.size[1],
        )
        if focal_length_px is not None
        and principal_point_px is not None
        and depth_accuracy == "absolute"
        and depth_plausible is not False
        else None
    )

    # Process teeth map: resize and analyze
    teeth_bbox = None
    incisor_distance = None
    incisor_distance_3d_mm = None
    incisor_measurement = None
    teeth_cutoff = None
    teeth_arches = None
    if teeth_image is not None:
        teeth_image = teeth_image.resize(picture_image.size)

        # Neutralise white/noisy borders that some teethmaps have — paint a
        # 30-pixel black frame so edge pixels are never mistaken for teeth.
        draw = ImageDraw.Draw(teeth_image)
        border = 30
        tw, th = teeth_image.size
        draw.rectangle([0, 0, tw - 1, border - 1], fill=0)          # top
        draw.rectangle([0, th - border, tw - 1, th - 1], fill=0)    # bottom
        draw.rectangle([0, 0, border - 1, th - 1], fill=0)          # left
        draw.rectangle([tw - border, 0, tw - 1, th - 1], fill=0)    # right

        # Per-matte adaptive threshold for the strong arch(es), plus a weak
        # opposite arch with its own threshold.  The resulting binary teeth
        # mask drives every later step (bbox, legacy distance, incisal edges
        # and depth support), so the weak arch's edge comes from its own mask.
        teeth_cutoff = teeth_threshold(teeth_image)
        teeth_arches = detect_teeth_arches(teeth_image, threshold=teeth_cutoff)
        if teeth_arches is not None:
            teeth_bbox = teeth_arches.bbox
            teeth_mask = teeth_arches.mask_image()
        if teeth_bbox is not None and teeth_bbox[3] >= round(
            teeth_image.size[1] * TEETH_MEASUREMENT_MIN_HEIGHT_FRACTION
        ):
            incisor_distance = find_incisor_distance_teeth(
                teeth_mask, teeth_bbox, threshold=MASK_ON
            )

            # 3D distance for legacy edge-of-gap points
            photo_w, photo_h = picture_image.size
            if incisor_distance is not None and measure_depth:
                # Legacy format: (x, y1, x, y2)
                lx1, ly1, lx2, ly2 = incisor_distance
                ld_upper = depth.sample(
                    lx1,
                    ly1,
                    support_mask=teeth_mask,
                    support_threshold=MASK_ON,
                    inward_y=-1,
                )
                ld_lower = depth.sample(
                    lx2,
                    ly2,
                    support_mask=teeth_mask,
                    support_threshold=MASK_ON,
                    inward_y=1,
                )
                ld_upper, ld_lower, _ = _reconcile_weak_arch_samples(
                    ld_upper, ld_lower, teeth_arches.weak_side, depth_calibrated
                )
                if ld_upper is not None and ld_lower is not None:
                    legacy_3d = distance_3d_from_cm(
                        (float(lx1), float(ly1)),
                        (float(lx2), float(ly2)),
                        ld_upper.distance_cm,
                        ld_lower.distance_cm,
                        photo_w,
                        photo_h,
                        camera,
                    )
                    if legacy_3d is not None:
                        incisor_distance_3d_mm = legacy_3d[0]

            # Centroid-based measurement with depth integration
            centroids = find_incisor_centroids(
                teeth_mask, teeth_bbox, threshold=MASK_ON
            )
            if centroids is not None:
                upper_c, lower_c = centroids
                pixel_dist_y = abs(lower_c[1] - upper_c[1])

                upper_depth_raw = None
                lower_depth_raw = None
                upper_distance_cm = None
                lower_distance_cm = None
                distance_3d_mm = None
                depth_assumed = None

                if measure_depth:
                    upper_sample = depth.sample(
                        upper_c[0],
                        upper_c[1],
                        support_mask=teeth_mask,
                        support_threshold=MASK_ON,
                        inward_y=-1,
                    )
                    lower_sample = depth.sample(
                        lower_c[0],
                        lower_c[1],
                        support_mask=teeth_mask,
                        support_threshold=MASK_ON,
                        inward_y=1,
                    )
                    upper_sample, lower_sample, depth_assumed = (
                        _reconcile_weak_arch_samples(
                            upper_sample, lower_sample, teeth_arches.weak_side, depth_calibrated
                        )
                    )
                    # 8-bit codes (legacy maps only; None for float depth).
                    upper_depth_raw = None if upper_sample is None else upper_sample.raw
                    lower_depth_raw = None if lower_sample is None else lower_sample.raw

                    if upper_sample is not None and lower_sample is not None:
                        result_3d = distance_3d_from_cm(
                            upper_c,
                            lower_c,
                            upper_sample.distance_cm,
                            lower_sample.distance_cm,
                            photo_w,
                            photo_h,
                            camera,
                        )
                        if result_3d is not None:
                            distance_3d_mm, upper_distance_cm, lower_distance_cm = (
                                result_3d
                            )

                incisor_measurement = IncisorMeasurement(
                    upper_centroid=upper_c,
                    lower_centroid=lower_c,
                    upper_depth_raw=upper_depth_raw,
                    lower_depth_raw=lower_depth_raw,
                    upper_distance_cm=upper_distance_cm,
                    lower_distance_cm=lower_distance_cm,
                    distance_3d_mm=distance_3d_mm,
                    pixel_distance_y=pixel_dist_y,
                    weak_arch=teeth_arches.weak_side,
                    depth_assumed=depth_assumed,
                )

    # Process skin map: resize
    if skin_image is not None:
        skin_image = skin_image.resize(picture_image.size)

    # Process hair map: resize
    if hair_image is not None:
        hair_image = hair_image.resize(picture_image.size)

    return IOSPortrait(
        photo=picture_image,
        depthmap=depth_image,
        teethmap=teeth_image,
        skinmap=skin_image,
        hairmap=hair_image,
        floatValueMin=float_min,
        floatValueMax=float_max,
        teeth_bbox=teeth_bbox,
        incisor_distance=incisor_distance,
        incisor_distance_3d_mm=incisor_distance_3d_mm,
        incisor_measurement=incisor_measurement,
        teeth_threshold=teeth_cutoff,
        teeth_arches=teeth_arches,
        depth_m=depth_m,
        depth_accuracy=depth_accuracy,
        depth_filtered=depth_filtered,
        focal_length_px=focal_length_px,
        principal_point_px=principal_point_px,
        depth_plausible=depth_plausible,
        depth=depth,
        depth_repaired=depth_repaired,
        depth_repair_check=depth_repair_check,
    )


def check_exif_data(exif):
    data = exif.get("Exif", {})
    data = data.get(42036, "default")

    reason = ""

    if isinstance(data, str):
        ret = data.find(const.TRUEDEPTH_EXIF_ID)
    elif isinstance(data, bytes):
        ret = data.find(const.TRUEDEPTH_EXIF_ID.encode("ascii"))
        try:
            reason = data.decode("ascii")
        except Exception:
            reason = "cannot encode"
    else:
        ret = -1

    if ret == -1:
        raise ExifValidationFailed(reason)
