"""load_image() on capture-app (absolute, 16-bit) depth -- runs on any OS.

The capture app's files cannot be shipped (they show a real person), and
building one needs macOS. Instead these tests take an anonymised fixture
HEIC for the photo and mattes, make pyheif's depth decode fail exactly like
it does on 16-bit disparity, and hand ``load_image`` a synthetic
``AppleDepthData`` in place of the macOS reader. Everything downstream --
rotation, 8-bit encoding, intrinsics, plausibility, code-0 handling and the
teeth measurement -- is the real code.
"""

import logging
import sys

import numpy as np
import pyheif
import pytest
from PIL import Image

from portrait_analyser import apple_depth, ios
from portrait_analyser.apple_depth import AppleDepthData, read_apple_depth
from portrait_analyser.camera import CameraModel, rotate_by_exif_orientation
from portrait_analyser.exceptions import (
    AppleDepthDecodeError,
    AppleDepthUnavailable,
    NoDepthMapFound,
)
from portrait_analyser.face import sample_depth_at_point
from portrait_analyser.incisor import (
    compute_incisor_distance_3d,
    depth_raw_to_distance_cm,
    raw_depth_to_distance_cm,
)
from portrait_analyser.tmd import compute_tmd_3d

SENSOR_W, SENSOR_H = 640, 480  # capture-app depth, sensor orientation
FX = 2766.0
INTRINSICS = (FX, FX, 1499.0, 2019.6)
REFERENCE = (3024.0, 4032.0)

# Synthetic incisal points (photo pixels of the 2320x3087 fixture photo).
UPPER = (1100.0, 1700.0)
LOWER = (1100.0, 2100.0)
TEETH_BBOX = (1000, 1650, 200, 500)


def _heif_16_bit_error(primary_image):
    raise pyheif.error.HeifError(
        code=7,
        subcode=0,
        message="Decoder plugin generated an error: Unspecified: "
        "Unsupported JPEG data precision 16",
    )


def _upright_to_sensor(depth_upright, orientation):
    """Inverse of rotate_by_exif_orientation for 1/3/6/8."""
    inverse = {1: 1, 3: 3, 6: 8, 8: 6}[orientation]
    sensor = rotate_by_exif_orientation(depth_upright, inverse)
    np.testing.assert_array_equal(
        rotate_by_exif_orientation(sensor, orientation), depth_upright
    )
    return sensor


@pytest.fixture(scope="module")
def fixture_mattes(request):
    """Skin and hair mattes of the fixture photo, resized to the upright depth grid."""
    from pathlib import Path

    path = Path(request.fspath).parent / "heic_face_data.heic"
    with open(path, "rb") as f:
        primary = pyheif.open_container(f).primary_image
        raw = {getattr(a, "type", ""): a.image for a in primary.auxiliary_images}
    skin = ios._decode_semantic_map(
        raw["urn:com:apple:photo:2019:aux:semanticskinmatte"]
    )
    hair = ios._decode_semantic_map(
        raw["urn:com:apple:photo:2019:aux:semantichairmatte"]
    )
    size = (SENSOR_H, SENSOR_W)  # upright portrait: 480 wide, 640 tall
    return (
        np.asarray(skin.resize(size)) >= 128,
        np.asarray(hair.resize(size)) >= 128,
    )


def _depth_map(fixture_mattes, face_m, background_m):
    """Upright 480x640 depth: ``face_m`` on skin/hair, ``background_m`` elsewhere,
    and ``face_m`` in a patch around the synthetic incisal points."""
    skin, hair = fixture_mattes
    depth = np.where(skin | hair, face_m, background_m).astype(np.float32)
    photo_w, photo_h = 2320, 3087
    x = round(UPPER[0] * (SENSOR_H - 1) / (photo_w - 1))
    y0 = round(UPPER[1] * (SENSOR_W - 1) / (photo_h - 1))
    y1 = round(LOWER[1] * (SENSOR_W - 1) / (photo_h - 1))
    depth[y0 - 6 : y1 + 7, x - 6 : x + 7] = face_m
    return depth


def _patch_capture_app(
    monkeypatch, depth_upright, orientation=6, accuracy="absolute", aux_orientation=None
):
    """Make load_image take the capture-app path with this upright depth."""
    sensor = _upright_to_sensor(depth_upright, orientation or 1)
    data = AppleDepthData(
        depth_m=sensor,
        accuracy=accuracy,
        filtered=False,
        quality="high",
        source_type="disparity",
        intrinsics=INTRINSICS,
        intrinsics_reference_size=REFERENCE,
        lens_distortion_center=INTRINSICS[2:],
        exif_orientation=orientation,
        aux_orientation=aux_orientation,
    )
    monkeypatch.setattr(ios, "_load_pyheif_depth", _heif_16_bit_error)
    monkeypatch.setattr(ios, "read_apple_depth", lambda path: data)
    return data


class _FakeArches:
    """Stands in for detect_teeth_arches' result: a fixed teeth box."""

    bbox = TEETH_BBOX
    weak_side = None

    def mask_image(self):
        return Image.new("L", (2320, 3087), 255)


def _patch_teeth(monkeypatch):
    monkeypatch.setattr(
        ios, "detect_teeth_arches", lambda image, threshold: _FakeArches()
    )
    monkeypatch.setattr(
        ios,
        "find_incisor_distance_teeth",
        lambda mask, bbox, threshold: (UPPER[0], UPPER[1], LOWER[0], LOWER[1]),
    )
    monkeypatch.setattr(
        ios, "find_incisor_centroids", lambda mask, bbox, threshold: (UPPER, LOWER)
    )


def _expected_pinhole_mm(portrait, z_m):
    camera = portrait.camera
    return abs(LOWER[1] - UPPER[1]) * z_m * 1000.0 / camera.fy


# --------------------------------------------------------------------------
# load_image producing a camera and measuring with it
# --------------------------------------------------------------------------


def test_load_image_builds_camera_and_measures_with_it(
    monkeypatch, heic_face_image_path, fixture_mattes
):
    depth = _depth_map(fixture_mattes, face_m=0.35, background_m=1.8)
    _patch_capture_app(monkeypatch, depth)
    _patch_teeth(monkeypatch)

    portrait = ios.load_image(str(heic_face_image_path))

    assert portrait.depth_accuracy == "absolute"
    assert portrait.depth_plausible is True
    assert portrait.depthmap.mode == "L"
    assert portrait.depthmap.size == (SENSOR_H, SENSOR_W)
    assert portrait.depth_code_zero_is_invalid is True
    assert portrait.depth_valid_mask.shape == (SENSOR_W, SENSOR_H)
    np.testing.assert_array_equal(portrait.depth_m, depth)

    camera = portrait.camera
    assert camera is not None
    assert (camera.width, camera.height) == portrait.photo.size
    # Reference 3024x4032 scaled to the 2320x3087 photo.
    assert camera.fy == pytest.approx(FX * 3087 / 4032)
    assert camera.cx == pytest.approx(1499.0 * 2320 / 3024)

    measurement = portrait.incisor_measurement
    assert measurement is not None
    assert measurement.upper_distance_cm == pytest.approx(35.0, abs=0.2)
    assert measurement.distance_3d_mm == pytest.approx(
        _expected_pinhole_mm(portrait, 0.35), rel=0.01
    )
    assert portrait.incisor_distance_3d_mm == pytest.approx(
        _expected_pinhole_mm(portrait, 0.35), rel=0.01
    )


def test_camera_mode_ignores_a_single_pixel_hole(
    monkeypatch, heic_face_image_path, fixture_mattes
):
    depth = _depth_map(fixture_mattes, face_m=0.35, background_m=1.8)
    x = round(UPPER[0] * (SENSOR_H - 1) / 2319)
    y = round(UPPER[1] * (SENSOR_W - 1) / 3086) - 1  # inward_y=-1 for upper
    depth[y, x] = np.nan
    _patch_capture_app(monkeypatch, depth)
    _patch_teeth(monkeypatch)

    portrait = ios.load_image(str(heic_face_image_path))

    assert portrait.depthmap.getpixel((x, y)) == 0
    measurement = portrait.incisor_measurement
    # The median of the 8 valid neighbours, not Z_far.
    assert measurement.upper_distance_cm == pytest.approx(35.0, abs=0.2)
    assert measurement.distance_3d_mm == pytest.approx(
        _expected_pinhole_mm(portrait, 0.35), rel=0.01
    )


def test_camera_mode_hole_gives_none_never_z_far(
    monkeypatch, heic_face_image_path, fixture_mattes
):
    depth = _depth_map(fixture_mattes, face_m=0.35, background_m=1.8)
    x = round(UPPER[0] * (SENSOR_H - 1) / 2319)
    y = round(UPPER[1] * (SENSOR_W - 1) / 3086)
    depth[y - 4 : y + 5, x - 4 : x + 5] = np.nan
    _patch_capture_app(monkeypatch, depth)
    _patch_teeth(monkeypatch)

    portrait = ios.load_image(str(heic_face_image_path))

    measurement = portrait.incisor_measurement
    assert measurement.upper_depth_raw is None
    assert measurement.upper_distance_cm is None
    assert measurement.distance_3d_mm is None
    assert portrait.incisor_distance_3d_mm is None


def test_inverted_depth_is_flagged_and_not_measured(
    monkeypatch, heic_face_image_path, fixture_mattes, caplog
):
    # IMG_2348-like: face decodes at ~2.7 m, background at ~0.55 m.
    depth = _depth_map(fixture_mattes, face_m=2.7, background_m=0.55)
    _patch_capture_app(monkeypatch, depth)
    _patch_teeth(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="portrait_analyser.ios"):
        portrait = ios.load_image(str(heic_face_image_path))

    assert portrait.depth_plausible is False
    assert "implausible depth" in caplog.text
    assert portrait.incisor_distance_3d_mm is None
    assert portrait.incisor_measurement.distance_3d_mm is None
    assert portrait.incisor_measurement.upper_depth_raw is None
    # The file is still loaded (not auto-inverted).
    np.testing.assert_array_equal(portrait.depth_m, depth)


def test_relative_capture_app_depth_stays_on_polynomial(
    monkeypatch, heic_face_image_path, fixture_mattes, caplog
):
    depth = _depth_map(fixture_mattes, face_m=0.35, background_m=1.8)
    _patch_capture_app(monkeypatch, depth, accuracy="relative")

    with caplog.at_level(logging.WARNING, logger="portrait_analyser.ios"):
        portrait = ios.load_image(str(heic_face_image_path))

    assert portrait.camera is None
    assert portrait.depth_accuracy == "relative"
    assert "not absolute" in caplog.text


@pytest.mark.parametrize("orientation", [None, 1])
def test_misaligned_depth_is_not_measured(
    monkeypatch, heic_face_image_path, fixture_mattes, orientation
):
    # Orientation missing, or identity leaving a landscape map on a portrait photo.
    depth = _depth_map(fixture_mattes, face_m=0.35, background_m=1.8)
    sensor_landscape = _upright_to_sensor(depth, 6)
    _patch_capture_app(monkeypatch, sensor_landscape, orientation=orientation)
    _patch_teeth(monkeypatch)

    portrait = ios.load_image(str(heic_face_image_path))

    assert portrait.depth_plausible is False
    assert portrait.camera is None
    assert portrait.focal_length_px is None
    assert portrait.incisor_measurement.distance_3d_mm is None


def test_aux_orientation_is_preferred(
    monkeypatch, heic_face_image_path, fixture_mattes
):
    depth = _depth_map(fixture_mattes, face_m=0.35, background_m=1.8)
    data = _patch_capture_app(monkeypatch, depth, orientation=6, aux_orientation=6)
    data.exif_orientation = None  # only the aux data knows

    portrait = ios.load_image(str(heic_face_image_path))

    assert portrait.depth_plausible is True
    np.testing.assert_array_equal(portrait.depth_m, depth)


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


def test_16_bit_depth_without_macos_raises_no_depth_map_found(
    monkeypatch, heic_face_image_path
):
    """What Linux CI sees for a capture-app file: a chained, explained error."""
    monkeypatch.setattr(ios, "_load_pyheif_depth", _heif_16_bit_error)

    def unavailable(path):
        raise AppleDepthUnavailable("running on sys.platform='linux'")

    monkeypatch.setattr(ios, "read_apple_depth", unavailable)

    with pytest.raises(NoDepthMapFound, match="16-bit disparity") as excinfo:
        ios.load_image(str(heic_face_image_path))
    assert isinstance(excinfo.value.__cause__, AppleDepthUnavailable)


def test_16_bit_depth_decode_failure_is_chained(monkeypatch, heic_face_image_path):
    monkeypatch.setattr(ios, "_load_pyheif_depth", _heif_16_bit_error)

    def broken(path):
        raise AppleDepthDecodeError("boom")

    monkeypatch.setattr(ios, "read_apple_depth", broken)

    with pytest.raises(NoDepthMapFound) as excinfo:
        ios.load_image(str(heic_face_image_path))
    assert isinstance(excinfo.value.__cause__, AppleDepthDecodeError)


def test_legacy_load_survives_metadata_failure(
    monkeypatch, heic_face_image_path, caplog
):
    before = ios.load_image(str(heic_face_image_path))

    def broken(path):
        raise AppleDepthDecodeError("unexpected pyobjc shape")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(ios, "read_apple_depth", broken)
    with caplog.at_level(logging.WARNING, logger="portrait_analyser.ios"):
        after = ios.load_image(str(heic_face_image_path))

    assert after.depth_accuracy is None
    assert "depth accuracy unknown" in caplog.text
    assert after.depthmap.tobytes() == before.depthmap.tobytes()
    assert after.floatValueMin == before.floatValueMin
    assert after.depth_code_zero_is_invalid is False
    assert after.depth_valid_mask is None
    assert after.camera is None


def test_read_apple_depth_wraps_unexpected_pyobjc_shapes(monkeypatch, tmp_path):
    monkeypatch.setattr(apple_depth, "_import_backend", lambda: (None, None, None))

    def bad_shape(*args):
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(apple_depth, "_read_apple_depth", bad_shape)
    with pytest.raises(AppleDepthDecodeError, match="IndexError") as excinfo:
        read_apple_depth(tmp_path / "x.heic")
    assert isinstance(excinfo.value.__cause__, IndexError)


def test_calibration_with_unexpected_shape_raises_index_error():
    class Calibration:
        def intrinsicMatrix(self):
            return ()

    with pytest.raises(IndexError):
        apple_depth._calibration_to_intrinsics(Calibration())


def test_orientation_disagreement_is_refused():
    assert (
        apple_depth._aux_orientation(
            {"kCGImageAuxiliaryDataInfoDataDescription": {"Orientation": 6}}
        )
        == 6
    )
    assert apple_depth._aux_orientation({}) is None
    apple_depth._check_orientations_agree(6, 6, "f")
    apple_depth._check_orientations_agree(None, 6, "f")
    with pytest.raises(AppleDepthDecodeError, match="disagrees"):
        apple_depth._check_orientations_agree(3, 6, "f")


# --------------------------------------------------------------------------
# Code 0 semantics in camera mode (unit level)
# --------------------------------------------------------------------------


def test_code_zero_is_invalid_only_in_camera_mode():
    camera = CameraModel(2766.0, 2766.0, 1512.0, 2016.0)
    # Legacy: code 0 is the farthest depth (100 / float_min cm).
    assert depth_raw_to_distance_cm(0, 0.34, 3.4) == pytest.approx(100 / 0.34)
    assert raw_depth_to_distance_cm(0, 0.34, 3.4) == pytest.approx(100 / 0.34)
    assert raw_depth_to_distance_cm(0, 0.34, 3.4, camera) is None
    assert (
        compute_incisor_distance_3d(
            (1500, 2200), (1500, 2500), 0, 200, 0.34, 3.4, 3024, 4032, camera=camera
        )
        is None
    )
    assert (
        compute_tmd_3d(
            (1500, 2600), (1500, 3000), 200, 0, 0.34, 3.4, 3024, 4032, camera=camera
        )
        is None
    )


def test_sample_depth_at_point_can_exclude_invalid_code():
    depthmap = Image.new("L", (3, 3), 0)
    depthmap.putpixel((0, 0), 200)
    depthmap.putpixel((1, 0), 202)
    depthmap.putpixel((2, 0), 204)
    # Legacy: zeros dominate the median.
    assert sample_depth_at_point(depthmap, 1, 1, 3, 3) == 0
    assert sample_depth_at_point(depthmap, 1, 1, 3, 3, invalid_value=0) == 202
    empty = Image.new("L", (3, 3), 0)
    assert sample_depth_at_point(empty, 1, 1, 3, 3, invalid_value=0) is None


def test_plausibility_check_units():
    skin = Image.new("L", (4, 2), 0)
    skin.putpixel((0, 0), 255)
    skin.putpixel((1, 0), 255)
    good = np.array([[0.35, 0.36, 1.8, 1.9], [1.8, 1.8, 1.8, np.nan]], dtype=np.float32)
    plausible, skin_cm, background_cm = ios.check_depth_plausibility(good, skin)
    assert plausible is True
    assert skin_cm == pytest.approx(35.5)
    assert background_cm == pytest.approx(180.0)

    inverted = np.array(
        [[2.7, 2.7, 0.55, 0.55], [0.55, 0.55, 0.6, 0.6]], dtype=np.float32
    )
    assert ios.check_depth_plausibility(inverted, skin)[0] is False
    assert ios.check_depth_plausibility(good, None) == (None, None, None)


# --------------------------------------------------------------------------
# Semantic matte stride fallback
# --------------------------------------------------------------------------


class _FakeLoaded:
    def __init__(self, mode, size, stride, data):
        self.mode, self.size, self.stride, self.data = mode, size, stride, data


class _FakeRaw:
    def __init__(self, loaded):
        self._loaded = loaded

    def load(self):
        return self._loaded


def test_semantic_matte_stride_fallback():
    # RGB rows with 8 bytes of padding (as in capture-app mattes). Tall
    # enough that the Camera-app layout (stride 3w+14, h-1 rows) needs more
    # bytes than there are, exactly as on the real 1512x2016 mattes.
    width, height, stride = 4, 5, 4 * 3 + 8
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            value = 10 * (y * width + x)
            row += bytes((value, value, value))
        row += b"\xee" * 8
        rows.append(bytes(row))
    raw = _FakeRaw(_FakeLoaded("RGB", (width, height), stride, b"".join(rows)))

    matte = ios._decode_semantic_map(raw)

    assert matte.mode == "L"
    assert matte.size == (width, height)
    assert np.asarray(matte).ravel().tolist() == [10 * i for i in range(width * height)]


def test_semantic_matte_camera_app_layout_unchanged():
    width, height = 4, 2
    stride = width * 3 + 14
    data = bytes(range(stride * height))
    matte = ios._decode_semantic_map(
        _FakeRaw(_FakeLoaded("RGB", (width, height), stride, data))
    )
    expected = Image.frombytes("L", (stride, height - 1), data)
    assert matte.tobytes() == expected.tobytes()
