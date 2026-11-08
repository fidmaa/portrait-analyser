"""Cross-platform 16-bit depth reader (heif_depth.py).

Real capture files cannot be shipped (they show real people), so the
end-to-end tests build a minimal HEIF container around a genuine 16-bit
lossless JPEG (encoded with imagecodecs) and stand in for pyheif's container
listing. Set ``PORTRAIT_ANALYSER_16BIT_SAMPLES`` to a folder of real
16-bit-depth ``.HEIC`` files to also run the reader on those (the first 20;
``PORTRAIT_ANALYSER_16BIT_LIMIT=0`` for all), and -- on macOS -- compare it
with the ImageIO/AVFoundation reader pixel by pixel.
"""

import os
import struct
import sys
from pathlib import Path

import imagecodecs
import numpy as np
import pyheif
import pytest

from portrait_analyser import heif_depth, ios
from portrait_analyser.apple_depth import AppleDepthData
from portrait_analyser.exceptions import AppleDepthDecodeError, AppleDepthUnavailable
from portrait_analyser.heif_depth import (
    parse_depth_xmp,
    read_heif_depth,
    read_item_data,
    stored_to_depth_m,
)

# Values copied from a real iPhone 17 Pro / iOS 26.3.1 portrait.
XMP_ELEMENTS = b"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 6.0.0">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about=""
            xmlns:apdi="http://ns.apple.com/pixeldatainfo/1.0/"
            xmlns:depthData="http://ns.apple.com/depthData/1.0/">
         <apdi:IntMaxValue>-2</apdi:IntMaxValue>
         <apdi:StoredFormat>1278226742</apdi:StoredFormat>
         <apdi:NativeInvalidValue>32767</apdi:NativeInvalidValue>
         <apdi:IntInvalidValue>-1</apdi:IntInvalidValue>
         <apdi:IntMinValue>0</apdi:IntMinValue>
         <apdi:NativeFormat>1751411059</apdi:NativeFormat>
         <apdi:FloatMaxValue>2.349609</apdi:FloatMaxValue>
         <apdi:FloatMinValue>1.292969</apdi:FloatMinValue>
         <apdi:AuxiliaryImageType>disparity</apdi:AuxiliaryImageType>
         <depthData:IntrinsicMatrixReferenceWidth>3024</depthData:IntrinsicMatrixReferenceWidth>
         <depthData:Quality>high</depthData:Quality>
         <depthData:IntrinsicMatrix>
            <rdf:Seq>
               <rdf:li>2766.2353515625</rdf:li><rdf:li>0.0</rdf:li><rdf:li>0.0</rdf:li>
               <rdf:li>0.0</rdf:li><rdf:li>2766.2353515625</rdf:li><rdf:li>0.0</rdf:li>
               <rdf:li>1498.9735107421875</rdf:li><rdf:li>2019.524169921875</rdf:li>
               <rdf:li>1</rdf:li>
            </rdf:Seq>
         </depthData:IntrinsicMatrix>
         <depthData:IntrinsicMatrixReferenceHeight>4032</depthData:IntrinsicMatrixReferenceHeight>
         <depthData:LensDistortionCenterOffsetX>1498.9735107421875</depthData:LensDistortionCenterOffsetX>
         <depthData:LensDistortionCenterOffsetY>2019.524169921875</depthData:LensDistortionCenterOffsetY>
         <depthData:Accuracy>absolute</depthData:Accuracy>
         <depthData:Filtered>False</depthData:Filtered>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>"""

FLOAT_MIN, FLOAT_MAX = 1.292969, 2.349609
DEPTH_ITEM_ID = 59


def _xmp():
    return parse_depth_xmp(XMP_ELEMENTS)


# --------------------------------------------------------------------------
# XMP
# --------------------------------------------------------------------------


def test_parse_depth_xmp_elements():
    xmp = _xmp()
    assert xmp["apdi:IntMaxValue"] == "-2"
    assert xmp["apdi:NativeFormat"] == "1751411059"
    assert xmp["depthData:Accuracy"] == "absolute"
    assert len(xmp["depthData:IntrinsicMatrix"]) == 9


def test_parse_depth_xmp_attributes():
    xmp = parse_depth_xmp(
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description xmlns:apdi="http://ns.apple.com/pixeldatainfo/1.0/" '
        b'apdi:FloatMinValue="0.5" apdi:IntMinValue="0"/></rdf:RDF></x:xmpmeta>'
    )
    assert xmp == {"apdi:FloatMinValue": "0.5", "apdi:IntMinValue": "0"}


def test_parse_depth_xmp_rejects_garbage():
    with pytest.raises(AppleDepthDecodeError):
        parse_depth_xmp(b"<not xml")


# --------------------------------------------------------------------------
# Integer -> metres
# --------------------------------------------------------------------------


def test_stored_to_depth_m_linear_disparity():
    stored = np.array([[0, 65534, 32767, 65535]], dtype=np.uint16)
    depth_m, source_type = stored_to_depth_m(stored, _xmp())
    assert source_type == "disparity"
    # Ends of the range: 0 -> FloatMin (far), IntMax -> FloatMax (near).
    assert depth_m[0, 0] == pytest.approx(1 / FLOAT_MIN, rel=1e-3)
    assert depth_m[0, 1] == pytest.approx(1 / FLOAT_MAX, rel=1e-3)
    mid = FLOAT_MIN + 32767 / 65534 * (FLOAT_MAX - FLOAT_MIN)
    assert depth_m[0, 2] == pytest.approx(1 / mid, rel=1e-3)
    # IntInvalidValue (-1 -> 65535) is NaN.
    assert np.isnan(depth_m[0, 3])
    assert depth_m.dtype == np.float32


def test_stored_to_depth_m_rounds_through_float16_like_avfoundation():
    stored = np.arange(0, 65534, 7, dtype=np.uint16)[None, :]
    depth_m, _ = stored_to_depth_m(stored, _xmp())
    disparity = 1.0 / depth_m.astype(np.float64)
    # Every disparity is exactly representable in float16 (native 'hdis').
    np.testing.assert_allclose(
        disparity, disparity.astype(np.float16).astype(np.float64), rtol=1e-6
    )


def test_stored_to_depth_m_depth_native_format():
    xmp = dict(_xmp(), **{"apdi:NativeFormat": str(0x66646570)})  # 'fdep'
    depth_m, source_type = stored_to_depth_m(np.array([[0, 65534]], dtype=np.uint16), xmp)
    assert source_type == "depth"
    assert depth_m[0, 0] == pytest.approx(FLOAT_MIN, rel=1e-6)
    assert depth_m[0, 1] == pytest.approx(FLOAT_MAX, rel=1e-6)


def test_stored_to_depth_m_requires_mapping():
    xmp = _xmp()
    del xmp["apdi:FloatMaxValue"]
    with pytest.raises(AppleDepthDecodeError):
        stored_to_depth_m(np.zeros((1, 1), dtype=np.uint16), xmp)


def test_stored_to_depth_m_rejects_unknown_format():
    xmp = dict(_xmp(), **{"apdi:NativeFormat": "1"})
    with pytest.raises(AppleDepthDecodeError):
        stored_to_depth_m(np.zeros((1, 1), dtype=np.uint16), xmp)


# --------------------------------------------------------------------------
# Minimal HEIF container
# --------------------------------------------------------------------------


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), box_type) + payload


def _heif_with_item(item_id: int, payload: bytes, construction_method: int = 0) -> bytes:
    """ftyp + meta(iloc v1[, idat]) + mdat holding ``payload`` as one item."""
    ftyp = _box(b"ftyp", b"heic" + b"\0\0\0\0" + b"mif1heic")

    def iloc(offset):
        body = bytes([1, 0, 0, 0])  # version 1, flags
        body += bytes([0x44, 0x00])  # offset_size=4 length_size=4 base=0 index=0
        body += struct.pack(">H", 1)  # item_count
        body += struct.pack(">HHH", item_id, construction_method, 0)
        body += struct.pack(">H", 1)  # extent_count
        body += struct.pack(">II", offset, len(payload))
        return _box(b"iloc", body)

    if construction_method == 1:
        meta = _box(b"meta", b"\0\0\0\0" + iloc(0) + _box(b"idat", payload))
        return ftyp + meta
    # Two passes: the mdat payload offset depends on the meta box size.
    meta_size = len(_box(b"meta", b"\0\0\0\0" + iloc(0)))
    offset = len(ftyp) + meta_size + 8
    meta = _box(b"meta", b"\0\0\0\0" + iloc(offset))
    return ftyp + meta + _box(b"mdat", payload)


@pytest.mark.parametrize("construction_method", [0, 1])
def test_read_item_data(construction_method):
    payload = bytes(range(256)) * 3
    data = _heif_with_item(7, payload, construction_method)
    assert read_item_data(data, 7) == payload


def test_read_item_data_unknown_item():
    with pytest.raises(AppleDepthDecodeError):
        read_item_data(_heif_with_item(7, b"abc"), 8)


def test_read_item_data_on_real_fixture(heic_image_path):
    """iloc of a real Apple file: the depth item's bytes are an HEVC stream."""
    data = Path(heic_image_path).read_bytes()
    depth_id = pyheif.open_container(data).primary_image.depth_image.id
    blob = read_item_data(data, depth_id)
    assert len(blob) > 1000
    # HEVC items are length-prefixed NAL units: the first prefix fits the item.
    assert 0 < int.from_bytes(blob[:4], "big") < len(blob)


# --------------------------------------------------------------------------
# read_heif_depth end to end
# --------------------------------------------------------------------------


class _FakeImage:
    def __init__(self, metadata):
        self.metadata = metadata


class _FakeDepth:
    def __init__(self, item_id, xmp):
        self.id = item_id
        self.image = _FakeImage([{"type": "mime", "data": xmp}])


class _FakePrimary:
    def __init__(self, depth):
        self.depth_image = depth
        # No Exif block -> orientation unknown; the EXIF path is covered by
        # the real-sample test below.
        self.image = _FakeImage([])


class _FakeContainer:
    def __init__(self, depth):
        self.primary_image = _FakePrimary(depth)


def _synthetic_heic(tmp_path, monkeypatch, stored, xmp=XMP_ELEMENTS):
    blob = imagecodecs.ljpeg_encode(stored)
    path = tmp_path / "synthetic.heic"
    path.write_bytes(_heif_with_item(DEPTH_ITEM_ID, blob))
    monkeypatch.setattr(
        pyheif, "open_container", lambda data: _FakeContainer(_FakeDepth(DEPTH_ITEM_ID, xmp))
    )
    return path


def test_read_heif_depth_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    stored = rng.integers(0, 65535, size=(48, 64), dtype=np.uint16)
    stored[0, 0] = 65535  # invalid
    path = _synthetic_heic(tmp_path, monkeypatch, stored)

    result = read_heif_depth(path)

    assert isinstance(result, AppleDepthData)
    expected, _ = stored_to_depth_m(stored, _xmp())
    np.testing.assert_array_equal(result.depth_m, expected)
    assert np.isnan(result.depth_m[0, 0])
    assert result.source_type == "disparity"
    assert result.accuracy == "absolute"
    assert result.quality == "high"
    assert result.filtered is False
    assert result.intrinsics == pytest.approx(
        (2766.2353515625, 2766.2353515625, 1498.9735107421875, 2019.524169921875)
    )
    assert result.intrinsics_reference_size == (3024.0, 4032.0)
    assert result.lens_distortion_center == pytest.approx((1498.9735107421875, 2019.524169921875))
    assert result.aux_orientation is None


def test_read_heif_depth_without_pixel_data_xmp(tmp_path, monkeypatch):
    path = _synthetic_heic(
        tmp_path, monkeypatch, np.zeros((8, 8), np.uint16), xmp=b"<x:xmpmeta/>"
    )
    with pytest.raises(AppleDepthDecodeError, match="pixel-data XMP"):
        read_heif_depth(path)


def test_read_heif_depth_no_depth(tmp_path, monkeypatch):
    path = tmp_path / "x.heic"
    path.write_bytes(b"")
    monkeypatch.setattr(pyheif, "open_container", lambda data: _FakeContainer(None))
    assert read_heif_depth(path) is None


# --------------------------------------------------------------------------
# load_image uses it when the macOS reader is unavailable
# --------------------------------------------------------------------------


def test_load_image_falls_back_to_heif_reader(monkeypatch, heic_face_image_path):
    def heif_16_bit_error(primary_image):
        raise pyheif.error.HeifError(code=4, subcode=0, message="Unsupported JPEG data precision 16")

    def apple_unavailable(path):
        raise AppleDepthUnavailable("not macOS")

    calls = []
    sensor = np.full((480, 640), 0.5, dtype=np.float32)

    def fake_heif_reader(path):
        calls.append(path)
        return AppleDepthData(
            depth_m=sensor,
            accuracy="absolute",
            filtered=False,
            quality="high",
            source_type="disparity",
            intrinsics=None,
            intrinsics_reference_size=None,
            exif_orientation=6,
        )

    monkeypatch.setattr(ios, "_load_pyheif_depth", heif_16_bit_error)
    monkeypatch.setattr(ios, "read_apple_depth", apple_unavailable)
    monkeypatch.setattr(ios, "read_heif_depth", fake_heif_reader)

    portrait = ios.load_image(str(heic_face_image_path))

    assert calls == [str(heic_face_image_path)]
    assert portrait.depth_accuracy == "absolute"
    assert portrait.depth_m.shape == (640, 480)  # rotated upright (orientation 6)


def test_load_image_reports_heif_reader_failure(monkeypatch, heic_face_image_path):
    def heif_16_bit_error(primary_image):
        raise pyheif.error.HeifError(code=4, subcode=0, message="precision 16")

    def apple_unavailable(path):
        raise AppleDepthUnavailable("not macOS")

    def broken(path):
        raise AppleDepthDecodeError("boom")

    monkeypatch.setattr(ios, "_load_pyheif_depth", heif_16_bit_error)
    monkeypatch.setattr(ios, "read_apple_depth", apple_unavailable)
    monkeypatch.setattr(ios, "read_heif_depth", broken)

    with pytest.raises(ios.NoDepthMapFound, match="imagecodecs reader.*boom"):
        ios.load_image(str(heic_face_image_path))


# --------------------------------------------------------------------------
# Real 16-bit samples (opt-in) and the macOS cross-check
# --------------------------------------------------------------------------

SAMPLES = os.environ.get("PORTRAIT_ANALYSER_16BIT_SAMPLES")


def _sample_files():
    if not SAMPLES:
        return []
    limit = int(os.environ.get("PORTRAIT_ANALYSER_16BIT_LIMIT", "20"))
    files = sorted(p for p in Path(SAMPLES).iterdir() if p.suffix.lower() == ".heic")
    return files[:limit] if limit > 0 else files


@pytest.mark.skipif(not SAMPLES, reason="set PORTRAIT_ANALYSER_16BIT_SAMPLES to a folder")
@pytest.mark.parametrize("path", _sample_files(), ids=lambda p: p.name)
def test_real_sample_decodes(path):
    result = read_heif_depth(path)
    if result is None:
        pytest.skip("no depth item")
    assert np.isfinite(result.depth_m).mean() > 0.5
    assert result.exif_orientation in range(1, 9)


@pytest.mark.skipif(
    not SAMPLES or sys.platform != "darwin",
    reason="needs macOS and PORTRAIT_ANALYSER_16BIT_SAMPLES",
)
@pytest.mark.parametrize("path", _sample_files(), ids=lambda p: p.name)
def test_real_sample_matches_avfoundation(path):
    """Ground truth: the ImageIO/AVFoundation reader on the same file."""
    from portrait_analyser.apple_depth import read_apple_depth

    ours = read_heif_depth(path)
    apple = read_apple_depth(path)
    if ours is None or apple is None:
        assert ours is None and apple is None
        return
    assert ours.depth_m.shape == apple.depth_m.shape
    np.testing.assert_array_equal(np.isnan(ours.depth_m), np.isnan(apple.depth_m))
    valid = np.isfinite(apple.depth_m)
    # Same float16 disparity -> identical up to float32 rounding of 1/x.
    np.testing.assert_allclose(ours.depth_m[valid], apple.depth_m[valid], rtol=1e-4)
    assert ours.accuracy == apple.accuracy
    assert ours.quality == apple.quality
    assert ours.filtered == apple.filtered
    assert ours.source_type == apple.source_type
    assert ours.intrinsics == pytest.approx(apple.intrinsics, rel=1e-5)
    assert ours.intrinsics_reference_size == pytest.approx(apple.intrinsics_reference_size)
    assert ours.depth_orientation == apple.depth_orientation


def test_read_heif_depth_rejects_non_jpeg_item(heic_image_path):
    """Camera-app fixture: HEVC depth is libheif's job; say so clearly."""
    with pytest.raises(AppleDepthDecodeError, match="not JPEG-coded"):
        read_heif_depth(heic_image_path)
