"""Tests for file-intrinsics (pinhole) measurement and absolute-depth loading.

Everything here is synthetic. The pure-maths tests run on any OS; the
``load_image`` end-to-end test builds a HEIC through macOS ImageIO and is
skipped elsewhere.
"""

import logging
import math
import sys

import numpy as np
import pytest
from PIL import Image, ImageOps

from portrait_analyser.apple_depth import (
    DISPARITY_FAR_CAP_M,
    encode_depth_as_disparity_8bit,
)
from portrait_analyser.camera import (
    CameraModel,
    intrinsics_in_photo_space,
    map_point_by_exif_orientation,
    rotate_by_exif_orientation,
)
from portrait_analyser.depth_sampling import (
    measure_filtered_surface_length,
    median_filter_depthmap,
)
from portrait_analyser.incisor import (
    compute_incisor_distance_3d,
    depth_raw_to_distance_cm,
    pixel_to_mm,
    pixels_per_mm_at_distance,
    point_to_mm,
)
from portrait_analyser.tmd import compute_tmd_3d

requires_macos = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="building/reading Apple depth HEICs needs macOS ImageIO/AVFoundation",
)


# --------------------------------------------------------------------------
# Pinhole pixel -> mm
# --------------------------------------------------------------------------


class TestPinholePixelToMm:
    def test_offset_from_principal_point_scales_with_distance(self):
        # 100 px right of the principal point, f = 2000 px, Z = 40 cm
        # -> 100 * 400 mm / 2000 = 20 mm.
        assert pixel_to_mm(
            1600.0, 40.0, 3000, focal_px=2000.0, principal_px=1500.0
        ) == (pytest.approx(20.0))
        assert pixel_to_mm(
            1400.0, 40.0, 3000, focal_px=2000.0, principal_px=1500.0
        ) == (pytest.approx(-20.0))

    def test_principal_point_defaults_to_image_centre(self):
        assert pixel_to_mm(1600.0, 40.0, 3000, focal_px=2000.0) == pytest.approx(20.0)

    def test_no_calibrated_range_limit(self):
        # The polynomial refuses 150 cm; the pinhole model does not.
        assert pixel_to_mm(1600.0, 150.0, 3000) is None
        assert pixel_to_mm(1600.0, 150.0, 3000, focal_px=2000.0) == pytest.approx(75.0)
        assert pixel_to_mm(1600.0, 5.0, 3000, focal_px=2000.0) == pytest.approx(2.5)

    @pytest.mark.parametrize("distance_cm", [None, 0.0, -10.0])
    def test_non_positive_distance_is_rejected(self, distance_cm):
        assert pixel_to_mm(1600.0, distance_cm, 3000, focal_px=2000.0) is None
        assert pixels_per_mm_at_distance(distance_cm, focal_px=2000.0) is None

    def test_pixels_per_mm_pinhole(self):
        assert pixels_per_mm_at_distance(40.0, focal_px=2000.0) == pytest.approx(5.0)

    def test_point_to_mm_uses_per_axis_intrinsics(self):
        camera = CameraModel(fx=2000.0, fy=4000.0, cx=1000.0, cy=500.0)
        x_mm, y_mm = point_to_mm(1100.0, 600.0, 40.0, 3000, 4000, camera)
        assert x_mm == pytest.approx(100 * 400 / 2000)
        assert y_mm == pytest.approx(100 * 400 / 4000)

    def test_incisor_distance_with_camera(self):
        camera = CameraModel(fx=2500.0, fy=2500.0, cx=1512.0, cy=2016.0)
        float_min, float_max = 0.5, 4.0
        upper, lower = (1500.0, 2200.0), (1520.0, 2600.0)
        result = compute_incisor_distance_3d(
            upper, lower, 200, 190, float_min, float_max, 3024, 4032, camera=camera
        )
        assert result is not None
        distance, upper_cm, lower_cm = result
        expected_points = [
            (
                (x - camera.cx) * z_cm * 10 / camera.fx,
                (y - camera.cy) * z_cm * 10 / camera.fy,
                z_cm * 10,
            )
            for (x, y), z_cm in ((upper, upper_cm), (lower, lower_cm))
        ]
        assert distance == pytest.approx(math.dist(*expected_points))


class TestCameraModel:
    def test_rejects_non_positive_focal(self):
        with pytest.raises(ValueError):
            CameraModel(fx=0.0, fy=1.0, cx=0.0, cy=0.0)

    def test_from_portrait_without_intrinsics_is_none(self):
        class Legacy:
            focal_length_px = None
            principal_point_px = None

        assert CameraModel.from_portrait(Legacy()) is None

    def test_from_portrait(self):
        class New:
            focal_length_px = (2766.0, 2765.0)
            principal_point_px = (1499.0, 2019.5)

        assert CameraModel.from_portrait(New()) == CameraModel(
            2766.0, 2765.0, 1499.0, 2019.5
        )


# --------------------------------------------------------------------------
# Legacy path unchanged when camera=None
# --------------------------------------------------------------------------


def _legacy_pixel_to_mm(pixel, distance_cm, dimension):
    """The pre-CameraModel implementation, copied verbatim as an oracle."""
    ppmm = pixels_per_mm_at_distance(distance_cm)
    if ppmm is None or ppmm <= 0:
        return None
    return (pixel - dimension / 2.0) / ppmm


class TestLegacyUnchanged:
    def test_pixel_to_mm_matches_polynomial(self):
        for pixel in (0.0, 870.0, 1450.0, 2319.0):
            for distance in (14.0, 15.0, 30.0, 55.5, 80.0, 81.0):
                assert pixel_to_mm(pixel, distance, 2320) == _legacy_pixel_to_mm(
                    pixel, distance, 2320
                )

    def test_incisor_distance_camera_none_is_polynomial(self):
        upper, lower = (1100.0, 1800.0), (1130.0, 2100.0)
        default = compute_incisor_distance_3d(
            upper, lower, 200, 180, 0.5, 2.0, 2320, 3087
        )
        explicit = compute_incisor_distance_3d(
            upper, lower, 200, 180, 0.5, 2.0, 2320, 3087, camera=None
        )
        assert default == explicit
        upper_cm, lower_cm = default[1], default[2]
        expected = math.dist(
            (
                _legacy_pixel_to_mm(upper[0], upper_cm, 2320),
                _legacy_pixel_to_mm(upper[1], upper_cm, 3087),
                upper_cm * 10,
            ),
            (
                _legacy_pixel_to_mm(lower[0], lower_cm, 2320),
                _legacy_pixel_to_mm(lower[1], lower_cm, 3087),
                lower_cm * 10,
            ),
        )
        assert default[0] == expected

    def test_tmd_camera_none_is_polynomial(self):
        args = ((1100.0, 2400.0), (1120.0, 2800.0), 200, 190, 0.5, 2.0, 2320, 3087)
        assert compute_tmd_3d(*args) == compute_tmd_3d(*args, camera=None)

    def test_surface_length_camera_none_vs_camera(self):
        depthmap = Image.new("L", (48, 64), 200)
        filtered = median_filter_depthmap(depthmap)
        points = [(100.0, 300.0), (200.0, 300.0), (300.0, 300.0)]
        legacy = measure_filtered_surface_length(filtered, points, 480, 640, 0.5, 4.0)
        z_cm = depth_raw_to_distance_cm(200, 0.5, 4.0)
        assert legacy == pytest.approx(200.0 / pixels_per_mm_at_distance(z_cm))

        camera = CameraModel(fx=500.0, fy=500.0, cx=240.0, cy=320.0)
        pinhole = measure_filtered_surface_length(
            filtered, points, 480, 640, 0.5, 4.0, camera=camera
        )
        assert pinhole == pytest.approx(200.0 * z_cm * 10 / 500.0)


# --------------------------------------------------------------------------
# Metric depth -> 8-bit disparity encoding
# --------------------------------------------------------------------------


class TestDisparityEncoding:
    def _synthetic_depth(self):
        rng = np.random.default_rng(1)
        depth = rng.uniform(0.25, 1.2, size=(60, 80)).astype(np.float32)
        depth[0, :5] = np.nan
        depth[1, :5] = 7.5  # beyond the far cap -> "no depth"
        depth[2, 0] = 0.25  # exact near end
        return depth

    def test_round_trip_within_documented_quantisation(self):
        depth = self._synthetic_depth()
        image, float_min, float_max = encode_depth_as_disparity_8bit(depth)
        assert image.mode == "L"
        assert image.size == (80, 60)
        valid = np.isfinite(depth) & (depth <= DISPARITY_FAR_CAP_M)
        assert float_max == pytest.approx(1.0 / np.nanmin(depth[valid]))
        assert float_min == pytest.approx(1.0 / np.nanmax(depth[valid]))

        codes = np.asarray(image)
        assert np.all(codes[~valid] == 0)
        assert np.all(codes[valid] >= 1)

        step_disparity = (float_max - float_min) / 255.0
        for (y, x), z_m in np.ndenumerate(depth):
            if not valid[y, x]:
                continue
            decoded_cm = depth_raw_to_distance_cm(
                int(codes[y, x]), float_min, float_max
            )
            # Half a code step in disparity (one step for the clamped far
            # end), converted to depth: dZ ~= Z**2 * d(disparity).
            tolerance_cm = 100.0 * (z_m**2) * step_disparity * 1.01
            assert abs(decoded_cm - z_m * 100.0) <= tolerance_cm, (
                y,
                x,
                z_m,
                decoded_cm,
            )

    def test_documented_step_at_30_cm_is_about_a_millimetre(self):
        depth = np.array([[0.25, 0.30], [0.50, DISPARITY_FAR_CAP_M]], dtype=np.float32)
        _, float_min, float_max = encode_depth_as_disparity_8bit(depth)
        step_mm_at_30_cm = 1000.0 * 0.30**2 * (float_max - float_min) / 255.0
        assert 1.0 < step_mm_at_30_cm < 1.5

    def test_far_cap_limits_range(self):
        depth = np.array([[0.3, 0.5], [2.0, 12.0]], dtype=np.float32)
        image, float_min, _ = encode_depth_as_disparity_8bit(depth, far_cap_m=3.0)
        assert float_min == pytest.approx(1.0 / 2.0)
        assert np.asarray(image)[1, 1] == 0

    def test_flat_map(self):
        depth = np.full((4, 4), 0.4, dtype=np.float32)
        image, float_min, float_max = encode_depth_as_disparity_8bit(depth)
        assert np.all(np.asarray(image) == 255)
        assert depth_raw_to_distance_cm(255, float_min, float_max) == pytest.approx(
            40.0
        )

    def test_no_valid_pixels_raises(self):
        with pytest.raises(ValueError):
            encode_depth_as_disparity_8bit(np.full((3, 3), np.nan, dtype=np.float32))


# --------------------------------------------------------------------------
# EXIF orientation of the depth map + principal point
# --------------------------------------------------------------------------


ASYMMETRIC = np.arange(12, dtype=np.uint8).reshape(3, 4) * 10  # 4 wide, 3 tall


def _pil_exif_transpose(array, orientation):
    image = Image.fromarray(array)
    exif = image.getexif()
    exif[0x0112] = orientation
    image.info["exif"] = exif.tobytes()
    return np.asarray(ImageOps.exif_transpose(image))


class TestOrientation:
    @pytest.mark.parametrize("orientation", range(1, 9))
    def test_rotation_matches_pil_exif_transpose(self, orientation):
        expected = _pil_exif_transpose(ASYMMETRIC, orientation)
        actual = rotate_by_exif_orientation(ASYMMETRIC, orientation)
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("orientation", [1, 3, 6, 8])
    def test_named_rotations(self, orientation):
        actual = rotate_by_exif_orientation(ASYMMETRIC, orientation)
        if orientation == 1:
            expected = ASYMMETRIC
        elif orientation == 3:
            expected = ASYMMETRIC[::-1, ::-1]
        elif orientation == 6:  # 90 degrees clockwise: bottom-left -> top-left
            expected = np.array(
                [[80, 40, 0], [90, 50, 10], [100, 60, 20], [110, 70, 30]]
            )
        else:  # 8: 90 degrees counter-clockwise: top-right -> top-left
            expected = np.array(
                [[30, 70, 110], [20, 60, 100], [10, 50, 90], [0, 40, 80]]
            )
        np.testing.assert_array_equal(actual, expected)

    def test_none_is_identity(self):
        assert rotate_by_exif_orientation(ASYMMETRIC, None) is ASYMMETRIC

    def test_invalid_orientation(self):
        with pytest.raises(ValueError):
            rotate_by_exif_orientation(ASYMMETRIC, 9)

    @pytest.mark.parametrize("orientation", range(1, 9))
    def test_point_mapping_follows_the_pixels(self, orientation):
        rotated = rotate_by_exif_orientation(ASYMMETRIC, orientation)
        height, width = ASYMMETRIC.shape
        for (y, x), value in np.ndenumerate(ASYMMETRIC):
            new_x, new_y, new_w, new_h = map_point_by_exif_orientation(
                x + 0.5, y + 0.5, width, height, orientation
            )
            assert (new_w, new_h) == (rotated.shape[1], rotated.shape[0])
            assert rotated[int(new_y - 0.5), int(new_x - 0.5)] == value


class TestPrincipalPointMapping:
    def test_legacy_same_frame_orientation_1_scales(self):
        focal, principal = intrinsics_in_photo_space(
            (2086.0, 2718.0, 1158.0, 1544.0),
            (2316.0, 3088.0),
            (480, 640),
            1,
            (2316, 3088),
        )
        assert focal == pytest.approx((2086.0, 2718.0))
        assert principal == pytest.approx((1158.0, 1544.0))

    def test_sensor_frame_reference_rotates_with_depth(self):
        # Landscape reference == landscape sensor frame; EXIF 6 -> portrait
        # photo at half resolution. (cx, cy) -> (H - cy, cx), fx <-> fy.
        focal, principal = intrinsics_in_photo_space(
            (3000.0, 2900.0, 2000.0, 1400.0),
            (4032.0, 3024.0),
            (640, 480),
            6,
            (1512, 2016),
        )
        assert focal == pytest.approx((2900.0 / 2, 3000.0 / 2))
        assert principal == pytest.approx(((3024.0 - 1400.0) / 2, 2000.0 / 2))

    @pytest.mark.parametrize("orientation", [3, 8])
    def test_sensor_frame_reference_other_rotations(self, orientation):
        cx, cy = 2000.0, 1400.0
        _, principal = intrinsics_in_photo_space(
            (3000.0, 3000.0, cx, cy),
            (4032.0, 3024.0),
            (640, 480),
            orientation,
            (4032, 3024) if orientation == 3 else (3024, 4032),
        )
        if orientation == 3:
            assert principal == pytest.approx((4032.0 - cx, 3024.0 - cy))
        else:
            assert principal == pytest.approx((cy, 4032.0 - cx))

    def test_capture_app_portrait_reference_orientation_6_is_identity(self):
        # Capture-app files: depth 640x480 (landscape sensor), reference
        # 3024x4032 portrait, EXIF 6, photo stored upright 3024x4032.
        focal, principal = intrinsics_in_photo_space(
            (2766.0, 2766.0, 1499.0, 2019.6),
            (3024.0, 4032.0),
            (640, 480),
            6,
            (3024, 4032),
        )
        assert focal == pytest.approx((2766.0, 2766.0))
        assert principal == pytest.approx((1499.0, 2019.6))

    def test_capture_app_portrait_reference_orientation_3(self):
        # Same reference, EXIF 3, photo stored 4032x3024: the reference is
        # the EXIF-6 frame, so relative to the photo it is rotated 90 deg
        # clockwise: (cx, cy) -> (4032 - cy, cx).
        focal, principal = intrinsics_in_photo_space(
            (2766.0, 2760.0, 1499.0, 2019.6),
            (3024.0, 4032.0),
            (640, 480),
            3,
            (4032, 3024),
        )
        assert focal == pytest.approx((2760.0, 2766.0))
        assert principal == pytest.approx((4032.0 - 2019.6, 1499.0))

    def test_mismatched_aspect_falls_back_to_centre(self, caplog):
        with caplog.at_level(logging.WARNING, logger="portrait_analyser.camera"):
            focal, principal = intrinsics_in_photo_space(
                (2000.0, 2000.0, 900.0, 600.0),
                (1920.0, 1080.0),
                (640, 480),
                1,
                (1600, 1200),
            )
        assert principal == pytest.approx((800.0, 600.0))
        assert focal == pytest.approx((2000.0 * 1600 / 1920, 2000.0 * 1600 / 1920))
        assert "photo centre" in caplog.text


# --------------------------------------------------------------------------
# load_image end to end on a synthetic 16-bit-depth HEIC (macOS only)
# --------------------------------------------------------------------------


def _build_heic_with_disparity(path, photo_size, disparities, exif_orientation):
    """Write a HEIC whose disparity aux image ImageIO encodes itself.

    The photo is a flat grey image of ``photo_size``; ``disparities`` is the
    sensor-orientation disparity map (1/m). EXIF carries the TrueDepth lens
    model and ``exif_orientation``.
    """
    import AVFoundation
    import Quartz
    from Foundation import NSURL

    height, width = disparities.shape
    err, disparity_buf = Quartz.CVPixelBufferCreate(
        None, width, height, Quartz.kCVPixelFormatType_DisparityFloat32, None, None
    )
    assert err == 0
    Quartz.CVPixelBufferLockBaseAddress(disparity_buf, 0)
    try:
        bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(disparity_buf)
        raw = Quartz.CVPixelBufferGetBaseAddress(disparity_buf).as_buffer(
            bytes_per_row * height
        )
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row)
        rows[:, : width * 4] = (
            np.asarray(disparities, dtype=np.float32)
            .view(np.uint8)
            .reshape(height, width * 4)
        )
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(disparity_buf, 0)
    depth_data = (
        AVFoundation.AVDepthData.alloc().initWithPixelBuffer_depthMetadataDictionary_(
            disparity_buf, {}
        )
    )
    aux_dict, aux_type = depth_data.dictionaryRepresentationForAuxiliaryDataType_(None)

    photo_w, photo_h = photo_size
    err, rgb_buf = Quartz.CVPixelBufferCreate(
        None, photo_w, photo_h, Quartz.kCVPixelFormatType_32ARGB, None, None
    )
    assert err == 0
    Quartz.CVPixelBufferLockBaseAddress(rgb_buf, 0)
    try:
        bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(rgb_buf)
        raw = Quartz.CVPixelBufferGetBaseAddress(rgb_buf).as_buffer(
            bytes_per_row * photo_h
        )
        np.frombuffer(raw, dtype=np.uint8)[:] = 128
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(rgb_buf, 0)
    ci_image = Quartz.CIImage.imageWithCVPixelBuffer_(rgb_buf)
    cg_image = Quartz.CIContext.context().createCGImage_fromRect_(
        ci_image, ((0, 0), (photo_w, photo_h))
    )

    properties = {
        Quartz.kCGImagePropertyOrientation: exif_orientation,
        Quartz.kCGImagePropertyExifDictionary: {
            Quartz.kCGImagePropertyExifLensModel: "synthetic front TrueDepth camera",
        },
    }
    url = NSURL.fileURLWithPath_(str(path))
    destination = Quartz.CGImageDestinationCreateWithURL(url, "public.heic", 1, None)
    assert destination is not None
    Quartz.CGImageDestinationAddImage(destination, cg_image, properties)
    Quartz.CGImageDestinationAddAuxiliaryDataInfo(destination, aux_type, aux_dict)
    assert Quartz.CGImageDestinationFinalize(destination)


def _simulate_16_bit_depth(monkeypatch):
    """Make pyheif's depth decode fail the way it does on 16-bit disparity.

    ImageIO writes the synthetic aux image in a form pyheif can decode, so
    the real capture-app failure is reproduced by raising pyheif's own error.
    """
    import pyheif

    from portrait_analyser import ios

    def fail(primary_image):
        raise pyheif.error.HeifError(
            code=7,
            subcode=0,
            message="Decoder plugin generated an error: Unspecified: "
            "Unsupported JPEG data precision 16",
        )

    monkeypatch.setattr(ios, "_load_pyheif_depth", fail)


@requires_macos
def test_load_image_legacy_path_reports_accuracy_but_no_intrinsics(tmp_path):
    from portrait_analyser.ios import load_image

    depth_sensor = np.full((6, 8), 0.4, dtype=np.float32)
    path = tmp_path / "synthetic.heic"
    _build_heic_with_disparity(path, (80, 60), 1.0 / depth_sensor, exif_orientation=1)

    portrait = load_image(str(path))

    assert portrait.depth_accuracy in ("absolute", "relative")
    assert portrait.depth_m is None
    assert portrait.focal_length_px is None
    assert portrait.camera is None


@requires_macos
def test_load_image_reads_depth_via_macos_and_rotates_it(tmp_path, monkeypatch):
    from portrait_analyser.ios import load_image

    # Sensor frame 8 wide x 6 tall; depth increases left -> right so the
    # rotation is observable. EXIF 6: upright photo is portrait.
    depth_sensor = np.tile(np.linspace(0.3, 0.6, 8, dtype=np.float32), (6, 1))
    path = tmp_path / "synthetic.heic"
    _build_heic_with_disparity(path, (60, 80), 1.0 / depth_sensor, exif_orientation=6)
    _simulate_16_bit_depth(monkeypatch)

    portrait = load_image(str(path))

    assert portrait.depth_accuracy in ("absolute", "relative")
    assert portrait.depth_m is not None
    assert portrait.depth_m.shape == (8, 6)  # rotated to portrait
    # After a 90-degree clockwise rotation the sensor's left->right ramp runs
    # top->bottom.
    column = portrait.depth_m[:, 3]
    assert np.all(np.diff(column) > 0)
    assert portrait.depthmap.mode == "L"
    assert portrait.depthmap.size == (6, 8)
    decoded_top = depth_raw_to_distance_cm(
        portrait.depthmap.getpixel((3, 0)),
        portrait.floatValueMin,
        portrait.floatValueMax,
    )
    assert decoded_top == pytest.approx(portrait.depth_m[0, 3] * 100, abs=0.5)
