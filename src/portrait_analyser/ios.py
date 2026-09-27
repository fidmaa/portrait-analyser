import logging
import sys
import xml.etree.ElementTree as ET
from typing import Union

import numpy as np
import piexif
import pyheif
from PIL import Image, ImageDraw

from . import const
from .apple_depth import encode_depth_as_disparity_8bit, read_apple_depth
from .camera import CameraModel, intrinsics_in_photo_space, rotate_by_exif_orientation
from .exceptions import (
    AppleDepthDecodeError,
    AppleDepthUnavailable,
    ExifValidationFailed,
    NoDepthMapFound,
    UnknownExtension,
)
from .face import (
    TEETH_MEASUREMENT_MIN_HEIGHT_FRACTION,
    detect_teeth_arches,
    find_incisor_centroids,
    find_incisor_distance_teeth,
    IncisorMeasurement,
    sample_depth_at_point,
    teeth_threshold,
)
from .incisor import compute_incisor_distance_3d, depth_raw_to_distance_cm

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


class IOSPortrait:
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
        # ImageIO/AVFoundation reader (the 16-bit capture-app format);
        # ``depthmap`` is its 8-bit disparity re-encoding.
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

    @property
    def camera(self):
        """:class:`~portrait_analyser.camera.CameraModel` from the file intrinsics, or None.

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
    """Upright depth (metres), its 8-bit disparity encoding and photo intrinsics.

    The depth map comes in sensor orientation; the photo/mattes are already
    upright, so only the depth is rotated (by the file's EXIF orientation).
    """
    depth_m = rotate_by_exif_orientation(apple.depth_m, apple.exif_orientation)
    depth_h, depth_w = depth_m.shape
    photo_w, photo_h = photo_size
    if abs(depth_w / depth_h - photo_w / photo_h) > 0.01 * (photo_w / photo_h):
        logger.warning(
            "%s: rotated depth map %sx%s does not match the photo aspect %sx%s "
            "(EXIF orientation %s); depth may be misaligned",
            fileName,
            depth_w,
            depth_h,
            photo_w,
            photo_h,
            apple.exif_orientation,
        )
    try:
        depth_image, float_min, float_max = encode_depth_as_disparity_8bit(depth_m)
    except ValueError as exc:
        raise NoDepthMapFound(f"{fileName}: depth map has no usable pixels: {exc}") from exc

    focal = principal = None
    if apple.intrinsics is not None and apple.intrinsics_reference_size is not None:
        sensor_h, sensor_w = apple.depth_m.shape
        focal, principal = intrinsics_in_photo_space(
            apple.intrinsics,
            apple.intrinsics_reference_size,
            (sensor_w, sensor_h),
            apple.exif_orientation,
            photo_size,
        )
    return np.ascontiguousarray(depth_m), depth_image, float_min, float_max, focal, principal


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
            (
                depth_m,
                depth_image,
                float_min,
                float_max,
                focal_length_px,
                principal_point_px,
            ) = _apple_depth_for_photo(apple_depth, fileName, picture_image.size)

        # Decode semantic maps
        teeth_image = _decode_semantic_map(teeth_raw) if teeth_raw else None
        skin_image = _decode_semantic_map(skin_raw) if skin_raw else None
        hair_image = _decode_semantic_map(hair_raw) if hair_raw else None

    # File intrinsics (capture-app format only); None = calibration polynomial.
    camera = (
        CameraModel(*focal_length_px, *principal_point_px)
        if focal_length_px is not None and principal_point_px is not None
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
            if (
                incisor_distance is not None
                and depth_image is not None
                and float_min is not None
                and float_max is not None
            ):
                # Legacy format: (x, y1, x, y2)
                lx1, ly1, lx2, ly2 = incisor_distance
                photo_w, photo_h = picture_image.size
                ld_upper = sample_depth_at_point(
                    depth_image,
                    lx1,
                    ly1,
                    photo_w,
                    photo_h,
                    support_mask=teeth_mask,
                    support_threshold=MASK_ON,
                    inward_y=-1,
                )
                ld_lower = sample_depth_at_point(
                    depth_image,
                    lx2,
                    ly2,
                    photo_w,
                    photo_h,
                    support_mask=teeth_mask,
                    support_threshold=MASK_ON,
                    inward_y=1,
                )
                ld_upper, ld_lower, _ = _reconcile_weak_arch_depth(
                    ld_upper, ld_lower, teeth_arches.weak_side, float_min, float_max
                )
                if ld_upper is not None and ld_lower is not None:
                    legacy_3d = compute_incisor_distance_3d(
                        (float(lx1), float(ly1)),
                        (float(lx2), float(ly2)),
                        ld_upper,
                        ld_lower,
                        float(float_min),
                        float(float_max),
                        photo_w,
                        photo_h,
                        camera=camera,
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

                if depth_image is not None:
                    photo_w, photo_h = picture_image.size
                    upper_depth_raw = sample_depth_at_point(
                        depth_image,
                        upper_c[0],
                        upper_c[1],
                        photo_w,
                        photo_h,
                        support_mask=teeth_mask,
                        support_threshold=MASK_ON,
                        inward_y=-1,
                    )
                    lower_depth_raw = sample_depth_at_point(
                        depth_image,
                        lower_c[0],
                        lower_c[1],
                        photo_w,
                        photo_h,
                        support_mask=teeth_mask,
                        support_threshold=MASK_ON,
                        inward_y=1,
                    )
                    upper_depth_raw, lower_depth_raw, depth_assumed = (
                        _reconcile_weak_arch_depth(
                            upper_depth_raw,
                            lower_depth_raw,
                            teeth_arches.weak_side,
                            float_min,
                            float_max,
                        )
                    )

                    if (
                        upper_depth_raw is not None
                        and lower_depth_raw is not None
                        and float_min is not None
                        and float_max is not None
                    ):
                        result_3d = compute_incisor_distance_3d(
                            upper_c,
                            lower_c,
                            upper_depth_raw,
                            lower_depth_raw,
                            float(float_min),
                            float(float_max),
                            photo_w,
                            photo_h,
                            camera=camera,
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
