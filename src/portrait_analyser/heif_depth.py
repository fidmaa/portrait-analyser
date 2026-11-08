"""Cross-platform reader for Apple's 16-bit depth/disparity aux images.

Original iPhone portraits (e.g. iPhone 17 Pro, iOS 26) store the depth aux
image as a *16-bit lossless JPEG* (SOF3) item inside the HEIC. libheif hands
JPEG-coded items to its ``jpegdec`` plugin, which only drives libjpeg's 8-bit
``jpeg_read_scanlines`` path -- so no libheif version decodes these, and
neither ``pyheif`` nor ``pillow-heif`` can. :mod:`portrait_analyser.apple_depth`
solves this on macOS through ImageIO/AVFoundation; this module does the same
everywhere else:

1. ``pyheif`` locates the depth item (its id) and its XMP metadata -- that
   part needs no pixel decoding and works with any libheif;
2. the item's raw bytes are read straight from the file via the ``iloc`` box;
3. ``imagecodecs`` decodes the lossless JPEG to ``uint16``;
4. the XMP ``apdi:`` (pixel data info) and ``depthData:`` keys -- the very
   data AVFoundation's ``AVDepthData`` is built from -- turn the integers into
   metres and supply accuracy, quality, filtering and intrinsics.

The result is an :class:`~portrait_analyser.apple_depth.AppleDepthData`, so
``ios.load_image`` treats both readers identically.

Value mapping (``apdi``): stored integers ``v`` in ``[IntMinValue,
IntMaxValue]`` map linearly onto ``[FloatMinValue, FloatMaxValue]`` of the
native type (disparity in 1/m, or depth in m); ``IntInvalidValue`` marks
invalid pixels. Integer values are stored signed and are reinterpreted as
``uint16`` (``-1`` -> 65535, ``-2`` -> 65534). The native format of these
files is ``hdis`` (DisparityFloat16), so -- like AVFoundation -- disparity is
rounded to float16 before ``depth = 1 / disparity``.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .apple_depth import MAX_PLAUSIBLE_DEPTH_M, AppleDepthData
from .exceptions import AppleDepthDecodeError

APDI_NS = "http://ns.apple.com/pixeldatainfo/1.0/"
DEPTH_NS = "http://ns.apple.com/depthData/1.0/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# CoreVideo FourCCs that can appear as apdi:NativeFormat.
NATIVE_DISPARITY_FORMATS = {
    0x68646973: np.float16,  # 'hdis' kCVPixelFormatType_DisparityFloat16
    0x66646973: np.float32,  # 'fdis' kCVPixelFormatType_DisparityFloat32
}
NATIVE_DEPTH_FORMATS = {
    0x68646570: np.float16,  # 'hdep' kCVPixelFormatType_DepthFloat16
    0x66646570: np.float32,  # 'fdep' kCVPixelFormatType_DepthFloat32
}


# --------------------------------------------------------------------------
# ISOBMFF: raw bytes of one item
# --------------------------------------------------------------------------


def _boxes(data: bytes, start: int, end: int):
    """Yield ``(type, payload_start, box_end)`` for the boxes in ``data[start:end]``."""
    offset = start
    while offset + 8 <= end:
        size, box_type = struct.unpack(">I4s", data[offset : offset + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[offset + 8 : offset + 16])[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            raise AppleDepthDecodeError(f"malformed ISOBMFF box {box_type!r} at offset {offset}")
        yield box_type.decode("latin-1"), offset + header, offset + size
        offset += size


def _read_uint(data: bytes, offset: int, size: int) -> int:
    return int.from_bytes(data[offset : offset + size], "big") if size else 0


def _parse_iloc(data: bytes, start: int, end: int) -> dict:
    """``{item_id: (construction_method, [(offset, length), ...])}``."""
    version = data[start]
    pos = start + 4
    offset_size, length_size = data[pos] >> 4, data[pos] & 0x0F
    base_offset_size = data[pos + 1] >> 4
    index_size = data[pos + 1] & 0x0F if version in (1, 2) else 0
    pos += 2
    count_size = 2 if version < 2 else 4
    item_count = _read_uint(data, pos, count_size)
    pos += count_size
    items = {}
    for _ in range(item_count):
        item_id = _read_uint(data, pos, count_size)
        pos += count_size
        construction_method = 0
        if version in (1, 2):
            construction_method = _read_uint(data, pos, 2) & 0x0F
            pos += 2
        pos += 2  # data_reference_index
        base_offset = _read_uint(data, pos, base_offset_size)
        pos += base_offset_size
        extent_count = _read_uint(data, pos, 2)
        pos += 2
        extents = []
        for _ in range(extent_count):
            pos += index_size
            extent_offset = _read_uint(data, pos, offset_size)
            pos += offset_size
            extent_length = _read_uint(data, pos, length_size)
            pos += length_size
            extents.append((base_offset + extent_offset, extent_length))
        items[item_id] = (construction_method, extents)
    if pos > end:
        raise AppleDepthDecodeError("malformed iloc box (runs past its end)")
    return items


def read_item_data(data: bytes, item_id: int) -> bytes:
    """Raw (still compressed) payload of HEIF item ``item_id``.

    Supports ``iloc`` construction methods 0 (file offsets) and 1 (offsets
    into the ``idat`` box) -- the ones still images use.

    :raises AppleDepthDecodeError: no ``meta``/``iloc``, unknown item,
        unsupported construction method, or extents outside the file
    """
    meta = next(((s, e) for t, s, e in _boxes(data, 0, len(data)) if t == "meta"), None)
    if meta is None:
        raise AppleDepthDecodeError("HEIF file has no meta box")
    iloc = idat = None
    for box_type, start, end in _boxes(data, meta[0] + 4, meta[1]):
        if box_type == "iloc":
            iloc = _parse_iloc(data, start, end)
        elif box_type == "idat":
            idat = (start, end)
    if iloc is None or item_id not in iloc:
        raise AppleDepthDecodeError(f"HEIF item {item_id} has no iloc entry")
    construction_method, extents = iloc[item_id]
    if construction_method == 0:
        base, limit = 0, len(data)
    elif construction_method == 1 and idat is not None:
        base, limit = idat
    else:
        raise AppleDepthDecodeError(
            f"HEIF item {item_id}: unsupported iloc construction method {construction_method}"
        )
    chunks = []
    for offset, length in extents:
        begin = base + offset
        finish = limit if length == 0 else begin + length
        if finish > limit:
            raise AppleDepthDecodeError(f"HEIF item {item_id}: extent runs past the data")
        chunks.append(data[begin:finish])
    return b"".join(chunks)


# --------------------------------------------------------------------------
# XMP
# --------------------------------------------------------------------------


def parse_depth_xmp(xmp: bytes) -> dict:
    """Flatten the ``apdi:``/``depthData:`` keys of a depth item's XMP.

    Returns ``{"apdi:Key": "value", "depthData:Seq": ["v1", ...], ...}``.
    Accepts both XMP serialisations: simple values as child elements and as
    attributes of ``rdf:Description``.
    """
    text = xmp.decode("utf-8", errors="replace").strip("\x00 \n")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise AppleDepthDecodeError(f"depth XMP is not valid XML: {exc}") from exc
    prefixes = {APDI_NS: "apdi", DEPTH_NS: "depthData"}
    result = {}

    def key_of(tag):
        if not tag.startswith("{"):
            return None
        namespace, local = tag[1:].split("}", 1)
        prefix = prefixes.get(namespace)
        return f"{prefix}:{local}" if prefix else None

    for description in root.iter(f"{{{RDF_NS}}}Description"):
        for attribute, value in description.attrib.items():
            key = key_of(attribute)
            if key:
                result[key] = value
        for child in description:
            key = key_of(child.tag)
            if not key:
                continue
            items = child.findall(f".//{{{RDF_NS}}}li")
            result[key] = [li.text or "" for li in items] if items else (child.text or "").strip()
    return result


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def _decode_lossless_jpeg(blob: bytes) -> np.ndarray:
    try:
        import imagecodecs
    except ImportError as exc:  # pragma: no cover - hard dependency
        raise AppleDepthDecodeError("decoding 16-bit depth needs the imagecodecs package") from exc
    errors = []
    for name in ("ljpeg_decode", "jpeg8_decode"):
        decoder = getattr(imagecodecs, name, None)
        if decoder is None:
            continue
        try:
            array = np.asarray(decoder(blob))
        except Exception as exc:  # imagecodecs raises codec-specific errors
            errors.append(f"{name}: {exc}")
            continue
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim != 2:
            raise AppleDepthDecodeError(f"depth JPEG decoded to unexpected shape {array.shape}")
        return array
    raise AppleDepthDecodeError("imagecodecs could not decode the depth JPEG: " + "; ".join(errors))


def _as_uint16(value: str) -> int:
    """apdi integers are stored signed (``-1`` = 65535)."""
    return int(value) & 0xFFFF


def stored_to_depth_m(stored: np.ndarray, xmp: dict) -> tuple[np.ndarray, str]:
    """Map the stored integer image to metres using the ``apdi:`` keys.

    :returns: ``(depth_m, source_type)``; ``depth_m`` is float32 with ``NaN``
        for invalid, non-positive and implausibly far pixels (same rule as
        :func:`portrait_analyser.apple_depth._pixel_buffer_to_depth_m`).
    """
    try:
        int_min = _as_uint16(xmp["apdi:IntMinValue"])
        int_max = _as_uint16(xmp["apdi:IntMaxValue"])
        float_min = float(xmp["apdi:FloatMinValue"])
        float_max = float(xmp["apdi:FloatMaxValue"])
        native_format = int(xmp["apdi:NativeFormat"])
    except (KeyError, ValueError) as exc:
        raise AppleDepthDecodeError(f"depth XMP lacks the apdi value mapping: {exc}") from exc
    invalid_value = xmp.get("apdi:IntInvalidValue")
    if int_max == int_min:
        raise AppleDepthDecodeError("depth XMP has IntMinValue == IntMaxValue")

    if native_format in NATIVE_DISPARITY_FORMATS:
        source_type, native_dtype = "disparity", NATIVE_DISPARITY_FORMATS[native_format]
    elif native_format in NATIVE_DEPTH_FORMATS:
        source_type, native_dtype = "depth", NATIVE_DEPTH_FORMATS[native_format]
    else:
        raise AppleDepthDecodeError(f"unsupported apdi:NativeFormat {native_format:#x}")

    values = stored.astype(np.float64)
    native = float_min + (values - int_min) / (int_max - int_min) * (float_max - float_min)
    native = native.astype(native_dtype).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_m = (1.0 / native) if source_type == "disparity" else native
    depth_m = depth_m.astype(np.float32)
    invalid = ~np.isfinite(depth_m) | (depth_m <= 0) | (depth_m > MAX_PLAUSIBLE_DEPTH_M)
    if invalid_value is not None:
        invalid |= stored == _as_uint16(invalid_value)
    depth_m[invalid] = np.nan
    return depth_m, source_type


def _intrinsics(xmp: dict):
    matrix = xmp.get("depthData:IntrinsicMatrix")
    width = xmp.get("depthData:IntrinsicMatrixReferenceWidth")
    height = xmp.get("depthData:IntrinsicMatrixReferenceHeight")
    if not isinstance(matrix, list) or len(matrix) != 9 or width is None or height is None:
        return None, None, None
    # Column-major like AVCameraCalibrationData.intrinsicMatrix:
    # (fx, 0, 0), (0, fy, 0), (cx, cy, 1).
    m = [float(v) for v in matrix]
    intrinsics = (m[0], m[4], m[6], m[7])
    reference_size = (float(width), float(height))
    center = None
    cx = xmp.get("depthData:LensDistortionCenterOffsetX")
    cy = xmp.get("depthData:LensDistortionCenterOffsetY")
    if cx is not None and cy is not None:
        center = (float(cx), float(cy))
    return intrinsics, reference_size, center


def _exif_orientation(primary_image) -> Optional[int]:
    import piexif

    for block in primary_image.image.metadata or []:
        if block.get("type") != "Exif":
            continue
        try:
            exif = piexif.load(block["data"])
        except Exception:
            return None
        value = exif.get("0th", {}).get(piexif.ImageIFD.Orientation)
        return int(value) if value is not None else None
    return None


def read_heif_depth(path: Union[str, Path]) -> Optional[AppleDepthData]:
    """Read the depth/disparity aux image of a HEIC without libheif decoding it.

    :returns: an :class:`AppleDepthData` (``aux_orientation`` is always
        ``None``; the EXIF orientation applies, as with ImageIO when the aux
        description carries none), or ``None`` when the file has no depth item.
    :raises AppleDepthDecodeError: the depth item exists but cannot be read,
        decoded, or interpreted.
    """
    import pyheif

    data = Path(path).read_bytes()
    container = pyheif.open_container(data)
    primary = container.primary_image
    depth = primary.depth_image
    if depth is None:
        return None

    xmp_blocks = [
        block["data"]
        for block in (depth.image.metadata or [])
        if block.get("type") == "mime" and b"pixeldatainfo" in block["data"]
    ]
    if not xmp_blocks:
        raise AppleDepthDecodeError(f"{str(path)!r}: depth item has no Apple pixel-data XMP")
    xmp = parse_depth_xmp(xmp_blocks[0])

    blob = read_item_data(data, depth.id)
    if not blob.startswith(b"\xff\xd8"):
        raise AppleDepthDecodeError(
            f"{str(path)!r}: the depth item is not JPEG-coded (HEVC depth is "
            "libheif's job, not this reader's)"
        )
    stored = _decode_lossless_jpeg(blob)
    depth_m, source_type = stored_to_depth_m(stored, xmp)
    xmp_type = xmp.get("apdi:AuxiliaryImageType")
    if xmp_type and xmp_type != source_type:
        raise AppleDepthDecodeError(
            f"{str(path)!r}: apdi:AuxiliaryImageType {xmp_type!r} contradicts the "
            f"native format ({source_type})"
        )

    intrinsics, reference_size, center = _intrinsics(xmp)
    return AppleDepthData(
        depth_m=depth_m,
        accuracy=xmp.get("depthData:Accuracy", "relative"),
        filtered=str(xmp.get("depthData:Filtered", "False")).lower() == "true",
        quality=xmp.get("depthData:Quality", "low"),
        source_type=source_type,
        intrinsics=intrinsics,
        intrinsics_reference_size=reference_size,
        lens_distortion_center=center,
        exif_orientation=_exif_orientation(primary),
        aux_orientation=None,
    )
