"""DepthMap: legacy (8-bit) vs float (full-precision) depth sampling."""

import math

import numpy as np
import pytest
from PIL import Image

from portrait_analyser import ios
from portrait_analyser.apple_depth import encode_depth_as_disparity_8bit
from portrait_analyser.camera import CameraModel
from portrait_analyser.depth_map import FloatDepthMap, LegacyDepthMap
from portrait_analyser.depth_sampling import (
    measure_filtered_surface_length,
    median_filter_depthmap,
    sample_points_along_line,
)
from portrait_analyser.extended_neck import compute_neck_width_3d
from portrait_analyser.face import sample_depth_at_point
from portrait_analyser.incisor import compute_incisor_distance_3d, depth_raw_to_distance_cm
from portrait_analyser.mouth import compute_mouth_measurement_from_facemesh
from portrait_analyser.neck import compute_neck_circumference
from portrait_analyser.tmd import compute_tmd_3d

PHOTO = (400, 600)  # photo width, height
ROWS, COLS = 60, 40  # native depth map
CAMERA = CameraModel(500.0, 500.0, 200.0, 300.0, width=400, height=600)


def _ramp_depth():
    """A smooth face-like surface: 30 cm centre, bulging back to 40 cm."""
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    r2 = ((xx - COLS / 2) / COLS) ** 2 + ((yy - ROWS / 2) / ROWS) ** 2
    return (0.30 + 0.4 * r2).astype(np.float32)


def _both(depth_m):
    image, fmin, fmax = encode_depth_as_disparity_8bit(depth_m)
    legacy = LegacyDepthMap(image, fmin, fmax, PHOTO, zero_is_invalid=True)
    return legacy, FloatDepthMap(depth_m, PHOTO)


# --------------------------------------------------------------------------
# Legacy DepthMap == the historical functions (bit for bit)
# --------------------------------------------------------------------------


class TestLegacyMatchesHistoricalFunctions:
    def setup_method(self):
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 256, size=(ROWS, COLS, 3), dtype=np.uint8)
        codes[..., 1] = 7  # channel 0 must be the one used
        self.image = Image.fromarray(codes, "RGB")
        self.fmin, self.fmax = 0.51, 3.77
        self.depth = LegacyDepthMap(self.image, self.fmin, self.fmax, PHOTO)

    def test_sample_is_sample_depth_at_point(self):
        for x, y in [(0, 0), (123.4, 456.7), (399, 599), (200, 10)]:
            raw = sample_depth_at_point(self.image, x, y, *PHOTO)
            sample = self.depth.sample(x, y)
            assert sample.raw == raw
            assert sample.distance_cm == depth_raw_to_distance_cm(raw, self.fmin, self.fmax)

    def test_filtered_bilinear_is_the_old_pipeline(self):
        filtered = median_filter_depthmap(self.image)
        points = list(sample_points_along_line(50, 100, 350, 500, 7))
        old = measure_filtered_surface_length(filtered, points, *PHOTO, self.fmin, self.fmax)
        new = measure_filtered_surface_length(
            None, points, *PHOTO, None, None, depth=self.depth.median_filtered()
        )
        assert new == old  # exact

    def test_code_array_is_channel_zero(self):
        np.testing.assert_array_equal(self.depth.code_array(), np.asarray(self.image)[..., 0])
        assert self.depth.valid_mask.all()  # Camera-app: code 0 is a real depth

    def test_depth_keyword_equals_positional(self):
        args = ((100.0, 200.0), (110.0, 260.0))
        raw_a = sample_depth_at_point(self.image, *args[0], *PHOTO)
        raw_b = sample_depth_at_point(self.image, *args[1], *PHOTO)
        assert compute_incisor_distance_3d(
            *args, raw_a, raw_b, self.fmin, self.fmax, *PHOTO
        ) == compute_incisor_distance_3d(*args, None, None, None, None, *PHOTO, depth=self.depth)
        assert compute_tmd_3d(
            *args, raw_a, raw_b, self.fmin, self.fmax, *PHOTO
        ) == compute_tmd_3d(*args, None, None, None, None, *PHOTO, depth=self.depth)
        assert compute_neck_width_3d(
            self.image, 300, 100.0, 300.0, *PHOTO, self.fmin, self.fmax
        ) == compute_neck_width_3d(None, 300, 100.0, 300.0, *PHOTO, None, None, depth=self.depth)
        landmarks = [(200.0, 250.0)] * 18
        landmarks[17] = (200.0, 320.0)
        assert compute_mouth_measurement_from_facemesh(
            landmarks, self.image, *PHOTO, self.fmin, self.fmax
        ) == compute_mouth_measurement_from_facemesh(
            landmarks, None, *PHOTO, None, None, depth=self.depth
        )


# --------------------------------------------------------------------------
# Legacy vs float agreement where both apply
# --------------------------------------------------------------------------


class TestLegacyFloatAgreement:
    def test_point_distances_agree_within_quantisation(self):
        legacy, flt = _both(_ramp_depth())
        for x, y in [(200, 300), (50, 80), (350, 520), (120.5, 400.2)]:
            # One 8-bit step at <= 40 cm is < 2.5 mm.
            assert flt.distance_cm(x, y) == pytest.approx(legacy.distance_cm(x, y), abs=0.25)
            assert flt.bilinear_cm(x, y) == pytest.approx(legacy.bilinear_cm(x, y), abs=0.25)

    def test_float_is_exact(self):
        depth = _ramp_depth()
        flt = FloatDepthMap(depth, PHOTO)
        # Photo pixel mapping to native pixel (10, 20) exactly.
        x = 10 * (PHOTO[0] - 1) / (COLS - 1)
        y = 20 * (PHOTO[1] - 1) / (ROWS - 1)
        assert flt.distance_cm(x, y, radius=0) == pytest.approx(depth[20, 10] * 100, rel=1e-7)
        assert flt.bilinear_cm(x, y) == pytest.approx(depth[20, 10] * 100, rel=1e-6)

    def test_surface_length_and_3d_distance_agree(self):
        legacy, flt = _both(_ramp_depth())
        points = list(sample_points_along_line(60, 300, 340, 300, 10))
        lengths = [
            d.median_filtered().surface_length_mm(points, camera=CAMERA) for d in (legacy, flt)
        ]
        assert lengths[1] == pytest.approx(lengths[0], rel=0.02)
        a, b = (200.0, 200.0), (200.0, 400.0)
        assert flt.distance_3d_mm(a, b, camera=CAMERA)[0] == pytest.approx(
            legacy.distance_3d_mm(a, b, camera=CAMERA)[0], abs=0.3
        )

    def test_profile_and_detector_scale(self):
        legacy, flt = _both(_ramp_depth())
        points = [(x, 300.0) for x in range(20, 380, 40)]
        np.testing.assert_allclose(flt.profile(points), legacy.profile(points), atol=0.25)
        # Detector scale: unquantised version of the 8-bit codes.
        np.testing.assert_allclose(flt.code_array(), legacy.code_array(), atol=0.5 + 1e-6)
        assert flt.bilinear_code(200, 300) == pytest.approx(legacy.bilinear_code(200, 300), abs=0.5)

    def test_neck_circumference_agrees(self):
        depth = np.full((ROWS, COLS), 1.5, dtype=np.float32)
        yy, xx = np.mgrid[0:ROWS, 0:COLS]
        neck = (xx >= 12) & (xx <= 28)
        depth[neck] = 0.33 + 0.002 * (xx[neck] - 20) ** 2
        skin = Image.fromarray(np.where(np.repeat(np.repeat(neck, 10, 0), 10, 1), 255, 0)
                               .astype(np.uint8)).resize(PHOTO)
        legacy, flt = _both(depth)
        kwargs = dict(face_location=(100, 60, 200, 150), scan_start_y=250, scan_end_y=450)
        old = compute_neck_circumference(skin, None, *PHOTO, None, None, depth=legacy, **kwargs)
        new = compute_neck_circumference(skin, None, *PHOTO, None, None, depth=flt, **kwargs)
        assert old is not None and new is not None
        assert new.neck_y == old.neck_y
        assert new.front_arc_length_mm == pytest.approx(old.front_arc_length_mm, rel=0.03)


# --------------------------------------------------------------------------
# Float sampling: NaN holes, no far cap, sentinel filtering
# --------------------------------------------------------------------------


class TestFloatSampling:
    def test_median_ignores_nan_hole(self):
        depth = np.full((ROWS, COLS), 0.35, dtype=np.float32)
        depth[20, 10] = np.nan
        depth[20, 11] = 0.36
        flt = FloatDepthMap(depth, PHOTO)
        x = 10 * (PHOTO[0] - 1) / (COLS - 1)
        y = 20 * (PHOTO[1] - 1) / (ROWS - 1)
        assert flt.distance_cm(x, y) == pytest.approx(35.0)
        assert flt.distance_cm(x, y, radius=0) is None
        assert flt.sample(x, y).raw is None  # float maps have no codes

    def test_all_nan_window_is_none(self):
        depth = np.full((ROWS, COLS), np.nan, dtype=np.float32)
        depth[0, 0] = 0.3
        flt = FloatDepthMap(depth, PHOTO)
        assert flt.distance_cm(200, 300) is None

    def test_bilinear_never_interpolates_across_nan(self):
        depth = np.full((ROWS, COLS), 0.35, dtype=np.float32)
        depth[20, 11] = np.nan
        flt = FloatDepthMap(depth, PHOTO)
        x = 10.5 * (PHOTO[0] - 1) / (COLS - 1)
        y = 20 * (PHOTO[1] - 1) / (ROWS - 1)
        assert flt.bilinear_cm(x, y) is None
        # Pixel-exact position of a valid neighbour does not touch the hole.
        assert flt.bilinear_cm(10 * (PHOTO[0] - 1) / (COLS - 1), y) == pytest.approx(35.0)

    def test_median_filter_fills_isolated_holes_only(self):
        depth = np.full((ROWS, COLS), 0.35, dtype=np.float32)
        depth[20, 10] = np.nan  # isolated hole: filled
        depth[40:50, 5:15] = np.nan  # large hole: stays a hole
        depth[5, 30] = 0.9  # spike: removed
        filtered = FloatDepthMap(depth, PHOTO).median_filtered()
        assert filtered.depth_m[20, 10] == pytest.approx(0.35)
        assert np.isnan(filtered.depth_m[45, 10])
        assert filtered.depth_m[5, 30] == pytest.approx(0.35)
        assert filtered.shape == (ROWS, COLS)

    def test_no_far_cap(self):
        depth = np.full((ROWS, COLS), 5.0, dtype=np.float32)  # wall at 5 m
        depth[:, :20] = 0.4
        flt = FloatDepthMap(depth, PHOTO)
        assert flt.distance_cm(350, 300) == pytest.approx(500.0)
        # The 3 m-capped display encoding calls the wall "no depth" ...
        assert flt.to_display_image().getpixel((35, 30)) == 0
        # ... and so does the detector scale, but not the measurement.
        assert flt.sample(350, 300).code is None
        assert flt.valid_mask.all()

    def test_sentinels_are_invalid(self):
        depth = np.full((ROWS, COLS), 0.35, dtype=np.float32)
        depth[0, 0] = 9999.0
        depth[0, 1] = -1.0
        depth[0, 2] = np.inf
        flt = FloatDepthMap(depth, PHOTO)
        assert not flt.valid_mask[0, :3].any()
        assert np.isnan(flt.to_cm_array()[0, :3]).all()

    def test_shape_mapping_and_display(self):
        flt = FloatDepthMap(_ramp_depth(), PHOTO)
        assert flt.shape == (ROWS, COLS)
        assert flt.photo_to_depth(PHOTO[0] - 1, PHOTO[1] - 1) == (COLS - 1, ROWS - 1)
        image = flt.to_display_image()
        assert image.mode == "L" and image.size == (COLS, ROWS)


# --------------------------------------------------------------------------
# Float measurements never read the 8-bit display image
# --------------------------------------------------------------------------


def test_float_measurements_ignore_the_display_image(monkeypatch):
    depth = _ramp_depth()
    flt = FloatDepthMap(depth, PHOTO)
    skin = Image.new("L", PHOTO, 0)
    skin.paste(255, (120, 0, 280, 600))

    def run():
        filtered = flt.median_filtered()
        landmarks = [(200.0, 250.0)] * 18
        landmarks[17] = (200.0, 320.0)
        neck = compute_neck_circumference(
            skin, None, *PHOTO, None, None, face_location=(100, 20, 200, 150),
            scan_start_y=250, scan_end_y=450, camera=CAMERA, depth=flt,
        )
        return (
            flt.distance_cm(200, 300),
            flt.distance_3d_mm((200, 200), (210, 400), camera=CAMERA),
            filtered.profile([(150, 300), (250, 310)]),
            filtered.surface_length_mm([(150, 300), (250, 310)], camera=CAMERA),
            compute_incisor_distance_3d(
                (200, 200), (210, 400), None, None, None, None, *PHOTO, camera=CAMERA, depth=flt
            ),
            compute_tmd_3d(
                (200, 200), (210, 400), None, None, None, None, *PHOTO, camera=CAMERA, depth=flt
            ),
            compute_mouth_measurement_from_facemesh(
                landmarks, None, *PHOTO, None, None, camera=CAMERA, depth=flt
            ),
            compute_neck_width_3d(
                None, 300, 150.0, 250.0, *PHOTO, None, None, camera=CAMERA, depth=flt
            ),
            None if neck is None else (neck.front_arc_length_mm, neck.arc_points_photo),
        )

    before = run()
    assert before[0] is not None and before[-1] is not None
    # Corrupt the display image and make any further encoding explode.
    display = flt.to_display_image()
    display.paste(0, (0, 0) + display.size)

    def forbidden(*args, **kwargs):
        raise AssertionError("a float-depth measurement touched the 8-bit encoding")

    monkeypatch.setattr(FloatDepthMap, "display_encoding", forbidden)
    monkeypatch.setattr(
        "portrait_analyser.depth_map.encode_depth_as_disparity_8bit", forbidden
    )
    assert run() == before


# --------------------------------------------------------------------------
# Inverted-depth repair (unit level)
# --------------------------------------------------------------------------


def _mattes():
    skin = Image.new("L", (COLS, ROWS), 0)
    skin.paste(255, (10, 10, 30, 50))
    return skin


def _face_on(face_m, background_m):
    depth = np.full((ROWS, COLS), background_m, dtype=np.float32)
    depth[10:50, 10:30] = face_m
    return depth


class TestRepairInvertedDepth:
    def test_reciprocal_plausible_is_repaired(self):
        repaired, (plausible, skin_cm, background_cm) = ios.repair_inverted_depth(
            _face_on(2.7, 0.57), _mattes()
        )
        assert repaired is not None
        assert plausible is True
        assert skin_cm == pytest.approx(100 / 2.7, rel=1e-5)
        assert background_cm == pytest.approx(100 / 0.57, rel=1e-5)

    def test_plausible_depth_is_never_repaired(self):
        repaired, _ = ios.repair_inverted_depth(_face_on(0.37, 1.8), _mattes())
        assert repaired is None

    def test_both_interpretations_fail(self):
        # As read: 1.5 m face. Reciprocal: face behind the background.
        repaired, (plausible, _, _) = ios.repair_inverted_depth(_face_on(1.5, 3.0), _mattes())
        assert repaired is None
        assert plausible is False

    def test_small_margin_is_ambiguous(self):
        # Reciprocal face 40 cm, background 45 cm: plausible but not clear.
        repaired, (plausible, _, _) = ios.repair_inverted_depth(
            _face_on(1 / 0.40, 1 / 0.45), _mattes()
        )
        assert plausible is True
        assert repaired is None

    def test_no_background_no_repair(self):
        skin = Image.new("L", (COLS, ROWS), 255)
        repaired, _ = ios.repair_inverted_depth(_face_on(2.7, 2.7), skin)
        assert repaired is None

    def test_nan_stays_invalid(self):
        depth = _face_on(2.7, 0.57)
        depth[0, 0] = np.nan
        depth[0, 1] = 0.01  # reciprocal 100 m: beyond the sentinel cap
        repaired, _ = ios.repair_inverted_depth(depth, _mattes())
        assert np.isnan(repaired[0, 0]) and np.isnan(repaired[0, 1])
        assert repaired[20, 20] == pytest.approx(1 / 2.7)
        assert not math.isnan(repaired[5, 5])
