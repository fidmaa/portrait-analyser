"""DepthMap: legacy (8-bit) vs float (full-precision) depth sampling."""

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from portrait_analyser import apple_depth, ios
from portrait_analyser.apple_depth import encode_depth_as_disparity_8bit
from portrait_analyser.camera import CameraModel
from portrait_analyser.depth_map import FLOAT_DETECTOR_CODE_RANGE, FloatDepthMap, LegacyDepthMap
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
        raw_profile = [flt.bilinear_cm(x, y) for x, y in points]
        np.testing.assert_allclose(raw_profile, legacy.profile(points), atol=0.25)
        # The float profile integrates over the smoothed map: close, not equal.
        np.testing.assert_allclose(flt.profile(points), raw_profile, atol=1.0)
        # Detector scale: the fixed reference range, not the file's own.
        fmin, fmax = FLOAT_DETECTOR_CODE_RANGE
        expected = np.clip(255 * (1 / _ramp_depth() - fmin) / (fmax - fmin), 1, 255)
        np.testing.assert_allclose(flt.code_array(), expected, rtol=1e-5)

    def test_neck_circumference_agrees(self):
        depth, skin = _neck_scene()
        legacy, flt = _both(depth)
        # With the camera the float arc is smoothed at a physical 6 mm scale.
        old = compute_neck_circumference(
            skin, None, *PHOTO, None, None, camera=CAMERA, depth=legacy, **NECK_KW
        )
        new = compute_neck_circumference(
            skin, None, *PHOTO, None, None, camera=CAMERA, depth=flt, **NECK_KW
        )
        assert old is not None and new is not None
        assert new.neck_y == old.neck_y
        assert new.front_arc_length_mm == pytest.approx(old.front_arc_length_mm, rel=0.03)


NECK_KW = {"face_location": (100, 60, 200, 150), "scan_start_y": 250, "scan_end_y": 450}


def _neck_scene():
    """A 17-pixel-wide rounded 'neck' at ~33 cm before a 1.5 m wall, and its skin matte."""
    depth = np.full((ROWS, COLS), 1.5, dtype=np.float32)
    _, xx = np.mgrid[0:ROWS, 0:COLS]
    neck = (xx >= 12) & (xx <= 28)
    depth[neck] = 0.33 + 0.002 * (xx[neck] - 20) ** 2
    skin = Image.fromarray(
        np.where(np.repeat(np.repeat(neck, 10, 0), 10, 1), 255, 0).astype(np.uint8)
    )
    return depth, skin


class TestFloatDetectorScaleIsFixed:
    def _neck(self, depth, skin):
        m = compute_neck_circumference(
            skin, None, *PHOTO, None, None, depth=FloatDepthMap(depth, PHOTO), **NECK_KW
        )
        assert m is not None
        return (m.neck_y, m.left_x, m.right_x, m.front_arc_length_mm, m.arc_points_photo)

    def test_display_constants_do_not_matter(self, monkeypatch):
        depth, skin = _neck_scene()
        reference = self._neck(depth, skin)

        def forbidden(*args, **kwargs):
            raise AssertionError("float detectors used the display encoding")

        monkeypatch.setattr(apple_depth, "DISPARITY_FAR_CAP_M", 1.0)
        monkeypatch.setattr(apple_depth, "NEAR_END_MEDIAN_SIZE", 7)
        monkeypatch.setattr(apple_depth, "disparity_encoding_range", forbidden)
        monkeypatch.setattr(apple_depth, "encode_depth_as_disparity_8bit", forbidden)
        monkeypatch.setattr(
            "portrait_analyser.depth_map.encode_depth_as_disparity_8bit", forbidden
        )
        assert self._neck(depth, skin) == reference

    def test_file_near_and_far_ends_do_not_matter(self):
        depth, skin = _neck_scene()
        reference = self._neck(depth, skin)
        # A stray near blob and a far wall away from the neck would have moved
        # the old per-file code range (robust near end / far end).
        changed = depth.copy()
        changed[0:3, 0:3] = 0.12
        changed[55:60, 35:40] = 9.0
        assert self._neck(changed, skin) == reference

    def test_codes_use_the_reference_range(self):
        flt = FloatDepthMap(np.full((ROWS, COLS), 0.5, dtype=np.float32), PHOTO)
        fmin, fmax = FLOAT_DETECTOR_CODE_RANGE
        assert (fmin, fmax) == pytest.approx((1 / 3.0, 1 / 0.25))
        assert flt.sample(200, 300).code == pytest.approx(255 * (2 - fmin) / (fmax - fmin))


class TestIntegrationSmoothing:
    """Float depth is smoothed (median + 2 mm bilateral) before integration."""

    SIZE = (3024, 4032)
    CAM = CameraModel(2766.0, 2766.0, 1512.0, 2016.0, width=3024, height=4032)
    DEPTH_SHAPE = (640, 480)

    def _pixel_rays(self):
        cols = self.DEPTH_SHAPE[1]
        u = (np.arange(cols) + 0.0) * (self.SIZE[0] - 1) / (cols - 1)
        return (u - self.CAM.cx) / self.CAM.fx  # X/Z per depth column

    def _walk(self, x_from_mm, x_to_mm, z_mm):
        """Photo points along a row whose ends project from X at depth z."""
        u0 = self.CAM.cx + x_from_mm * self.CAM.fx / z_mm
        u1 = self.CAM.cx + x_to_mm * self.CAM.fx / z_mm
        return list(sample_points_along_line(u0, 2016.0, u1, 2016.0, 6.3))

    def test_noisy_plane_surface_equals_linear(self):
        rng = np.random.default_rng(3)
        rows = self.DEPTH_SHAPE[0]
        # Tilted plane (10 % slope in X) at 40 cm with 1 mm white noise,
        # the per-pixel noise measured on real capture-app skin.
        slope = 0.10
        ray = np.tile(self._pixel_rays(), (rows, 1))
        z = 0.40 / (1 - slope * ray)
        depth = (z + rng.normal(0, 0.001, z.shape)).astype(np.float32)
        flt = FloatDepthMap(depth, self.SIZE)
        points = self._walk(-50, 50, 400)
        linear = flt.distance_3d_mm(points[0], points[-1], camera=self.CAM, radius=3)[0]
        smoothed = flt.surface_length_mm(points, camera=self.CAM)
        median_only = FloatDepthMap(flt.median_filtered().depth_m, self.SIZE)
        from portrait_analyser.depth_map import surface_length_mm

        unsmoothed = surface_length_mm(median_only, points, *self.SIZE, camera=self.CAM)
        assert smoothed == pytest.approx(linear, rel=0.02)
        assert unsmoothed > smoothed * 1.02  # the jitter it removes (~3 %)

    def test_noisy_cylinder_keeps_its_arc(self):
        rng = np.random.default_rng(4)
        rows = self.DEPTH_SHAPE[0]
        radius, z_axis = 60.0, 460.0  # mm: a neck-sized cylinder, front at 40 cm
        ray = self._pixel_rays()
        # Ray x = t * Z meets the cylinder x^2 + (Z - z_axis)^2 = R^2 (front).
        a = 1 + ray**2
        disc = z_axis**2 - a * (z_axis**2 - radius**2)
        z_row = np.where(disc >= 0, (z_axis - np.sqrt(np.maximum(disc, 0))) / a, 1500.0)
        depth = np.tile(z_row / 1000.0, (rows, 1))
        depth = (depth + self._correlated_noise(rng, depth.shape, corr_mm=1.5)).astype(np.float32)
        flt = FloatDepthMap(depth, self.SIZE)
        # Walk the front arc between X = -0.8 R and +0.8 R.
        angle = math.asin(0.8)
        z_edge = z_axis - radius * math.cos(angle)
        u0 = self.CAM.cx - 0.8 * radius * self.CAM.fx / z_edge
        u1 = self.CAM.cx + 0.8 * radius * self.CAM.fx / z_edge
        points = list(sample_points_along_line(u0, 2016.0, u1, 2016.0, 6.3))
        true_arc = 2 * radius * angle
        assert flt.surface_length_mm(points, camera=self.CAM) == pytest.approx(true_arc, rel=0.03)

    def _correlated_noise(self, rng, shape, white_mm=1.0, corr_mm=2.0, length_mm=7.0):
        """TrueDepth-like noise: 1 mm white + mid-frequency relief correlated
        over ~7 mm (the kind a flat board shows), in metres."""
        import cv2

        focal_depth = self.CAM.fx * shape[1] / self.SIZE[0]
        relief = cv2.GaussianBlur(rng.normal(0, 1, shape), (0, 0), length_mm * focal_depth / 400)
        relief *= corr_mm / relief.std()
        return (rng.normal(0, white_mm, shape) + relief) / 1000.0

    def _rays(self):
        rows, cols = self.DEPTH_SHAPE
        u = np.arange(cols) * (self.SIZE[0] - 1) / (cols - 1)
        v = np.arange(rows) * (self.SIZE[1] - 1) / (rows - 1)
        return np.meshgrid((u - self.CAM.cx) / self.CAM.fx, (v - self.CAM.cy) / self.CAM.fy)

    def _front_arc_points(self, radius, z_centre):
        angle = math.asin(0.8)
        z_edge = z_centre - radius * math.cos(angle)
        half = 0.8 * radius * self.CAM.fx / z_edge
        points = list(
            sample_points_along_line(
                self.CAM.cx - half, self.CAM.cy, self.CAM.cx + half, self.CAM.cy, 6.3
            )
        )
        return points, 2 * radius * angle

    def test_flat_board_with_correlated_noise(self):
        rng = np.random.default_rng(7)
        tx, _ = self._rays()
        z = 0.40 / (1 - 0.3 * tx)  # a board tilted ~17 degrees
        noisy = (z + self._correlated_noise(rng, z.shape)).astype(np.float32)
        flt = FloatDepthMap(noisy, self.SIZE)
        flt.subject_depth_m = 0.40
        for x_from, x_to in ((-50, 50), (-40, 60), (-60, 20)):
            points = self._walk(x_from, x_to, 400)
            smooth = flt.integration_map(self.CAM)
            linear = smooth.distance_3d_mm(points[0], points[-1], camera=self.CAM, radius=0)[0]
            assert flt.surface_length_mm(points, camera=self.CAM) / linear < 1.03

    @pytest.mark.parametrize("noisy", [False, True])
    def test_sphere_keeps_its_curvature(self, noisy):
        # r = 40 mm, like a nose tip / forehead; apex at 40 cm.
        radius, z_centre = 40.0, 440.0
        tx, ty = self._rays()
        a = 1 + tx**2 + ty**2
        disc = z_centre**2 - a * (z_centre**2 - radius**2)
        z = np.where(disc >= 0, (z_centre - np.sqrt(np.maximum(disc, 0))) / a, 1500.0) / 1000
        if noisy:
            z = z + self._correlated_noise(np.random.default_rng(8), z.shape, corr_mm=1.5)
        flt = FloatDepthMap(z.astype(np.float32), self.SIZE)
        flt.subject_depth_m = 0.40
        points, true_arc = self._front_arc_points(radius, z_centre)
        # Noise-free: the 6 mm smoothing rounds the sphere by ~2 %.
        assert flt.surface_length_mm(points, camera=self.CAM) == pytest.approx(true_arc, rel=0.03)

    def test_same_map_whichever_variant_is_used(self):
        flt = FloatDepthMap(_ramp_depth(), PHOTO)
        assert flt.integration_map() is flt.median_filtered().integration_map()
        points = [(100.0, 300.0), (200.0, 310.0), (300.0, 300.0)]
        assert flt.surface_length_mm(points) == flt.median_filtered().surface_length_mm(points)
        assert flt.profile(points) == flt.integration_map().profile(points)
        assert measure_filtered_surface_length(
            None, points, *PHOTO, None, None, depth=flt.median_filtered()
        ) == flt.surface_length_mm(points)

    def test_silhouette_is_not_blended_with_the_background(self):
        depth = np.full((ROWS, COLS), 1.5, dtype=np.float32)
        depth[:, 12:29] = 0.35  # a flat subject before a wall 1.15 m behind
        smooth = FloatDepthMap(depth, PHOTO).integration_map()
        assert smooth.depth_m[30, 12] == pytest.approx(0.35, abs=1e-4)
        assert smooth.depth_m[30, 11] == pytest.approx(1.5, abs=1e-4)

    def test_legacy_integration_map_is_the_median_filter(self):
        legacy, _ = _both(_ramp_depth())
        assert np.array_equal(
            np.asarray(legacy.integration_map().image),
            np.asarray(median_filter_depthmap(legacy.image)),
        )


def test_float_neck_refuses_a_mostly_missing_arc():
    depth, skin = _neck_scene()
    flt = FloatDepthMap(depth, PHOTO)
    assert compute_neck_circumference(skin, None, *PHOTO, None, None, depth=flt, **NECK_KW)
    holed = depth.copy()
    holed[:, 13:20] = np.nan  # ~40 % of the arc has no depth
    flt_holed = FloatDepthMap(holed, PHOTO)
    assert (
        compute_neck_circumference(skin, None, *PHOTO, None, None, depth=flt_holed, **NECK_KW)
        is None
    )


def test_lazy_portrait_depth():
    photo = Image.new("RGB", PHOTO)
    legacy_image = Image.new("RGB", (COLS, ROWS), (200, 200, 200))
    legacy = ios.IOSPortrait(photo, legacy_image, floatValueMin=0.5, floatValueMax=3.5)
    assert legacy.depth.kind == "legacy"
    assert legacy.depth is legacy.depth  # built once
    assert legacy.depth.distance_cm(10, 10) == depth_raw_to_distance_cm(200, 0.5, 3.5)
    flt = ios.IOSPortrait(photo, depth_m=_ramp_depth())
    assert flt.depth.kind == "float"
    assert ios.IOSPortrait(photo).depth is None
    replacement = FloatDepthMap(_ramp_depth(), PHOTO)
    legacy.depth = replacement
    assert legacy.depth is replacement


def test_weak_arch_samples_untouched_without_calibration():
    from portrait_analyser.depth_map import DepthSample

    upper, lower = DepthSample(None, raw=200, code=200), DepthSample(None, raw=150, code=150)
    assert ios._reconcile_weak_arch_samples(upper, lower, "upper", calibrated=False) == (
        upper,
        lower,
        None,
    )
    # Calibrated but a distance is missing: the strong arch's sample is used.
    assert ios._reconcile_weak_arch_samples(upper, lower, "upper") == (lower, lower, "upper")


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
    display.paste(0, (0, 0, *display.size))

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

# Matte 40x60 for a 400x600 "photo"; a 20x40-pixel face blob has a minor axis
# of 4 * 20 / sqrt(12) = 23.1 matte px = 231 photo px, i.e. ~15 cm at 37 cm
# with this focal length -- and ~109 cm at 2.7 m.
FOCAL = 570.0
MATTE_PHOTO = (400, 600)


def _mattes(box=(10, 10, 30, 50)):
    skin = Image.new("L", (COLS, ROWS), 0)
    skin.paste(255, box)
    return skin


def _face_on(face_m, background_m, box=(10, 10, 30, 50)):
    depth = np.full((ROWS, COLS), background_m, dtype=np.float32)
    depth[box[1] : box[3], box[0] : box[2]] = face_m
    return depth


def _repair(depth, skin, focal=FOCAL):
    return ios.repair_inverted_depth(depth, skin, focal_px=focal, photo_size=MATTE_PHOTO)


class TestRepairInvertedDepth:
    def test_reciprocal_plausible_is_repaired(self):
        repaired, check = _repair(_face_on(2.7, 0.57), _mattes())
        assert repaired is not None
        assert check.repaired is True
        assert check.reciprocal_skin_cm == pytest.approx(100 / 2.7, rel=1e-5)
        assert check.reciprocal_background_cm == pytest.approx(100 / 0.57, rel=1e-5)
        assert check.face_width_px == pytest.approx(4 * 200 / math.sqrt(12), rel=0.05)
        assert 10 <= check.reciprocal_face_width_cm <= 25
        assert check.as_read_face_width_cm > 25

    def test_background_does_not_veto(self):
        # IMG_2376-like: a board held in front makes the non-skin median
        # nearer than the face even after the repair.
        repaired, check = _repair(_face_on(1 / 0.41, 1 / 0.39), _mattes())
        assert repaired is not None
        assert check.reciprocal_plausible is False  # background nearer than face

    def test_plausible_depth_is_never_repaired(self):
        repaired, check = _repair(_face_on(0.37, 1.8), _mattes())
        assert repaired is None
        assert check.reason == "depth as read is not implausible"

    @pytest.mark.parametrize("z_m", [1.0, 1.1, 1.26])
    def test_correct_far_face_is_never_repaired(self, z_m):
        # A correct capture, face 15 cm wide at 1.0-1.26 m before a 2.5 m
        # wall: as read it fails the 15-100 cm window (or sits on its edge),
        # its reciprocal (79-100 cm) would pass it -- but the width as read
        # is a face width, so nothing may be "repaired".
        photo = (3024, 4032)
        focal = 2766.0
        width_px = 15.0 / (z_m * 100) * focal
        height_px = 21.0 / (z_m * 100) * focal
        skin = Image.new("L", (photo[0] // 4, photo[1] // 4), 0)
        cx, cy = photo[0] / 8, photo[1] / 8
        ImageDraw.Draw(skin).ellipse(
            (cx - width_px / 8, cy - height_px / 8, cx + width_px / 8, cy + height_px / 8),
            fill=255,
        )
        depth = np.full((640, 480), 2.5, dtype=np.float32)
        depth[np.asarray(skin.resize((480, 640))) >= 128] = z_m
        depth[450:, :] = 0.6  # plus a near foreground
        repaired, check = ios.repair_inverted_depth(
            depth, skin, focal_px=focal, photo_size=photo
        )
        assert repaired is None, check
        if check.as_read_face_width_cm is not None:
            assert check.as_read_face_width_cm == pytest.approx(15.0, rel=0.05)

    def test_both_interpretations_fail(self):
        # As read: 1.5 m face (implausible). Reciprocal 67 cm: plausible
        # distance, but a 231 px face there is 27 cm wide -- not a face.
        repaired, check = _repair(_face_on(1.5, 3.0), _mattes())
        assert repaired is None
        assert not check.repaired

    def test_far_subject_with_near_foreground_is_not_repaired(self):
        # A real person 2 m away (face 4x6 matte px -> ~46 photo px, i.e.
        # 16 cm at 2 m) behind a near foreground at 0.5 m. The reciprocal
        # (face 50 cm, background 2 m) passes the distance tests, but the face
        # would be 4 cm wide there.
        box = (18, 27, 22, 33)
        repaired, check = _repair(_face_on(2.0, 0.5, box), _mattes(box))
        assert repaired is None
        assert 10 <= check.as_read_face_width_cm <= 25
        assert check.reciprocal_face_width_cm < 10
        assert "plausible as read" in check.reason

    def test_correct_depth_with_near_foreground_is_not_repaired(self):
        # Correct map, but a board held in front (non-skin nearer than the
        # face) makes the plausibility check fail. As read, the face is at a
        # plausible distance and width, so it must not be "repaired".
        repaired, check = _repair(_face_on(0.37, 0.30), _mattes())
        assert repaired is None
        assert "plausible as read" in check.reason

    def test_both_ways_plausible_is_ambiguous(self):
        # Only Z = 1 m is its own reciprocal within 15-100 cm: a 1 m face in
        # front of a 0.5 m foreground, 15 cm wide either way.
        repaired, check = _repair(_face_on(1.0, 0.5), _mattes(), focal=1500.0)
        assert repaired is None
        assert "ambiguous" in check.reason

    @pytest.mark.parametrize(
        ("focal", "repaired_expected"),
        [
            # width_cm = 230.9 px * 37.04 cm / focal: 25 cm at focal ~342
            # (as read, 2.7 m: > 180 cm either way).
            (350.0, True),  # 24.4 cm -- inside
            (335.0, False),  # 25.5 cm -- just outside
        ],
    )
    def test_face_width_boundary(self, focal, repaired_expected):
        repaired, check = _repair(_face_on(2.7, 0.57), _mattes(), focal=focal)
        assert (repaired is not None) is repaired_expected, check

    def test_without_focal_nothing_is_repaired(self):
        repaired, check = _repair(_face_on(2.7, 0.57), _mattes(), focal=None)
        assert repaired is None
        assert "focal" in check.reason

    def test_no_skin_no_repair(self):
        repaired, _ = _repair(_face_on(2.7, 0.57), Image.new("L", (COLS, ROWS), 0))
        assert repaired is None

    def test_nan_stays_invalid(self):
        depth = _face_on(2.7, 0.57)
        depth[0, 0] = np.nan
        depth[0, 1] = 0.01  # reciprocal 100 m: beyond the sentinel cap
        repaired, _ = _repair(depth, _mattes())
        assert np.isnan(repaired[0, 0]) and np.isnan(repaired[0, 1])
        assert repaired[20, 20] == pytest.approx(1 / 2.7)
        assert not math.isnan(repaired[5, 5])

    def test_face_width_is_orientation_free(self):
        upright = ios.face_width_px(_mattes())
        sideways = ios.face_width_px(_mattes().transpose(Image.Transpose.ROTATE_90))
        assert sideways == pytest.approx(upright, rel=1e-6)
