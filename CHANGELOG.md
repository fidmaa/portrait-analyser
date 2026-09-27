# Changelog

All notable changes to portrait-analyser are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `apple_depth` module: `read_apple_depth()` reads Apple's embedded
  depth/disparity data via macOS ImageIO + AVFoundation (pyobjc), for HEIC
  files whose depth aux image is 16-bit JPEG-compressed disparity -- a
  format `pyheif`/`pyheif-iplweb` and current `pillow-heif` cannot decode
  ("Unsupported JPEG data precision 16" / "JPEG decoder plugin not built
  in"). Returns an `AppleDepthData` (metres, accuracy, filtered, quality,
  source type, intrinsics, EXIF orientation) or `None` when a file has no
  depth/disparity aux image at all. macOS-only: raises
  `AppleDepthUnavailable` elsewhere or when pyobjc isn't installed;
  `AppleDepthDecodeError` on a genuine decode failure. Adds
  `pyobjc-framework-Quartz` / `pyobjc-framework-AVFoundation` as
  `sys_platform == 'darwin'` dependencies, matching the existing
  `pyheif-iplweb` convention. The map is returned in sensor orientation;
  `AppleDepthData.exif_orientation` is the rotation that lines it up with
  the (already upright) photo.
- `load_image()` reads photos from the TrueDepth capture app (absolute
  depth stored as 16-bit disparity, which pyheif rejects with "Unsupported
  JPEG data precision 16"): on any pyheif error decoding the depth image it
  falls back to `read_apple_depth()` (macOS), rotates the depth map by the
  EXIF orientation (never the photo/mattes, which are stored upright) and
  re-encodes it as the familiar 8-bit `L` disparity `depthmap` with
  `floatValueMax = 1/Z_near` and `floatValueMin = 1/Z_far`, `Z_far =
  min(farthest valid depth, DISPARITY_FAR_CAP_M = 3 m)`; farther/invalid
  pixels encode as 0. Quantisation: one code step is ~1.1-1.3 mm of depth at
  30 cm and ~3-3.6 mm at 50 cm (round-trip error at most half a step).
  `depthmap` mode is `"L"` for these files and `"RGB"` for Camera-app files. New
  `encode_depth_as_disparity_8bit()` does the encoding. Without macOS the
  load fails with `NoDepthMapFound` chained to the reason.
- New `IOSPortrait` attributes (all `None` when unknown): `depth_m`
  (full-precision upright depth in metres, NaN = invalid; capture-app files
  only), `depth_accuracy` (`"absolute"`/`"relative"`, also filled for
  Camera-app files on macOS -- iPhone 17 Pro Camera-app depth is
  `"relative"` and runs 7-28 % short), `depth_filtered`, `focal_length_px`
  `(fx, fy)` and `principal_point_px` `(cx, cy)` in upright-photo pixels
  (capture-app files only), and the `camera` property.
- `camera` module: `CameraModel(fx, fy, cx, cy)` (frozen dataclass,
  `CameraModel.from_portrait()`), `rotate_by_exif_orientation()`,
  `map_point_by_exif_orientation()` and `intrinsics_in_photo_space()`
  (maps AVCameraCalibrationData intrinsics through the depth rotation into
  photo pixels).
- Pinhole metric conversion from file intrinsics: `pixel_to_mm(...,
  *, focal_px=None, principal_px=None)` and
  `pixels_per_mm_at_distance(..., *, focal_px=None)` use
  `(pixel - principal) * Z_mm / focal_px` when `focal_px` is given (no
  15-80 cm range limit), and `point_to_mm()` converts both axes from a
  `CameraModel`. `compute_incisor_distance_3d`, `compute_tmd_3d`,
  `measure_filtered_surface_length`, `compute_neck_circumference`,
  `compute_neck_width_3d`, `detect_neck_midpoint_from_dual_mask` and
  `compute_mouth_measurement_from_facemesh` take a keyword-only
  `camera=None`. `load_image()` passes the file's camera for its own incisor
  measurements on capture-app files. `pixels_per_mm_at_distance` is now
  exported from the package root.

- Safety for capture-app depth (review fixes):
  - Depth code 0 means "no depth" in capture-app `depthmap`s (NaN or beyond
    the 3 m cap) but a real farthest depth in Camera-app ones;
    `depth_raw_to_distance_cm(0)` returns `Z_far` either way. New
    `raw_depth_to_distance_cm(v, fmin, fmax, camera=None)` returns None for 0
    when a camera is given; `compute_incisor_distance_3d` / `compute_tmd_3d`
    use it; `sample_depth_at_point(..., invalid_value=None)` can exclude a
    code from its median (load_image and the mouth measurement pass 0 for
    capture-app files). `IOSPortrait.depth_code_zero_is_invalid` and
    `IOSPortrait.depth_valid_mask` expose it.
  - Pinhole conversion has a working range, 10-150 cm
    (`MIN/MAX_PINHOLE_DISTANCE_CM`); None outside.
  - `IOSPortrait.depth_plausible` / `check_depth_plausibility()`: skin-matte
    median depth must be 15-100 cm and nearer than the background, else
    False (e.g. inverted depth) and load_image skips its 3-D teeth
    measurements. Never auto-inverted.
  - Depth is rotated by the aux data's own `Orientation`
    (`AppleDepthData.aux_orientation` / `.depth_orientation`), EXIF as
    fallback; a disagreement raises `AppleDepthDecodeError`. Unknown
    orientation or an aspect mismatch after rotation -> no intrinsics,
    `depth_plausible = False`, nothing measured.
  - The camera (pinhole) model is only used for `"absolute"` depth; relative
    capture-app depth stays on the polynomial with a warning.
  - The 8-bit encoding's near end is the 0.1st percentile of valid depths
    (`NEAR_PERCENTILE`); nearer pixels saturate at 255, so one stray pixel
    no longer coarsens the quantisation.
  - `read_apple_depth` turns unexpected pyobjc shapes (Attribute/Index/Key/
    Type/ValueError) into `AppleDepthDecodeError`; a Camera-app load never
    fails because of it (logged, `depth_accuracy = None`).
  - `CameraModel` gains optional `width`/`height` (the image its intrinsics
    refer to, i.e. the full-resolution photo).
  - pyobjc lower bound raised to 12.2 (the tested version).

### Changed

- Semantic mattes whose row padding does not fit the historical Camera-app
  layout (capture-app files) are now decoded by row stride instead of being
  dropped as `None`. Camera-app files decode exactly as before.
- Everything is unchanged for Camera-app photos: with `camera=None` (the
  default) the iPhone 14 calibration polynomial is used bit-for-bit, and
  `load_image()` never switches them to file intrinsics.

## [0.6.2] - 2026-09-27

### Added

- `ulbt` module measuring the upper lip bite test (ULBT) by colour rather than
  by landmark position. `compute_ulbt_from_facemesh()` classifies the band
  FaceMesh proposes as upper lip against reference colours sampled from skin,
  the lower lip and (optionally) the iOS teeth matte, and reports the share of
  that band which still reads as vermilion.
- `measure-ulbt` console script: point it at a HEIC and it reports how far the
  lower incisors have covered the upper lip vermilion. `--json` for batching.
- `detect_face_mesh()` in `pose`, exposing the 478 Face Mesh landmarks without
  also running Pose estimation (which needs shoulders in frame).
- `teeth_threshold()`: adaptive per-matte confidence threshold for Apple's
  semantic teeth matte (half of the matte's robust peak, clipped to
  `[64, 200]`), and `IOSPortrait.teeth_threshold` recording the value used.
- `detect_teeth_arches()` / `TeethArches`: per-arch teeth detection that also
  finds a weak opposite arch (Apple often gives one arch only ~10-35
  confidence) across the mouth gap with its own threshold.
  `IOSPortrait.teeth_arches`, and `IncisorMeasurement.weak_arch` /
  `depth_assumed` flags.

### Changed

- Teeth detection now works on weak teeth mattes. `load_image()` uses one
  adaptive threshold for the teeth bounding box, the legacy incisor distance,
  the incisal-edge points and the depth support mask instead of a fixed
  `> 200`, which missed teeth on many real captures whose whole matte peaks
  at ~100-220. `find_bounding_box_teeth`, `find_incisor_distance_teeth`,
  `find_incisor_centroids` and `sample_depth_at_point` default their
  threshold to `None` (adaptive); explicit numeric thresholds behave as before.
- `find_bounding_box_teeth` is vectorised, ignores speckle components and uses
  margins / minimum height / minimum area relative to the matte size instead
  of fixed 100 px margins and a 200 px minimum height, so a single visible arch
  now yields a bounding box. `load_image()` only runs the incisor measurements
  when the box is at least 6.5 % of the matte height (the old 200 px), so a
  single arch is never measured as a mouth opening.
- `load_image()` measures incisors on the per-arch teeth mask, so photos with
  one faint arch are measured too. When the faint arch's edge depth lies
  more than 2 cm from the other arch's (the depth map resolves the mouth
  cavity, not the barely visible teeth), the other arch's depth is used for
  both edges and `depth_assumed` says so.
- `pose` diagnostic chatter ("Mouth open ratio") now goes to stderr instead of
  stdout, so it cannot corrupt a caller's machine-readable output.

### Notes

- ULBT class I/II/III thresholds are **not** established. The measurement is
  deliberately continuous and returns no class label; it has been checked
  against a single capture and needs calibrating against a graded series.
  Neither FaceMesh nor semantic face parsing can be trusted to locate the
  vermilion in this pose — both place "upper lip" on the philtrum skin once
  the vermilion is rolled under by the bite, which is why the measurement
  keys off pixel colour instead of model labels.

## [0.6.1] - 2026-08-11

### Changed

- Increased the default neck-circumference multiplier for the measured 3D
  front arc from `2.7` to `3.0`.

## [0.6.0] - 2026-08-11

### Added

- `local_surface` module with robust local-plane detrending and pose-invariant
  peak/valley scoring for click-centred anatomical landmark patches.
- `find_stable_depth_x_from_edge()` for walking inward from a skin-matte edge
  until the depth profile reaches its first stable run.
- Direct Euclidean `front_chord_length_mm` diagnostics alongside the existing
  3D surface-polyline neck arc.

### Changed

- `compute_neck_circumference()` replaces its fixed 5% silhouette inset with
  adaptive left/right depth stabilization. `NeckMeasurement` now also exposes
  the original `mask_left_x` and `mask_right_x` coordinates for diagnostics.
- Neck skin selection now rejects weak matte values, median-cleans isolated
  noise, subtracts an optional semantic hair matte, re-reads the skin boundary
  at the actual shifted arc Y, and prevents stable-depth samples from leaving
  the cleaned skin mask. Maximum inward search is reduced to 12% of neck width.
- Added FaceMesh-scaled anatomical neck-search bounds and changed bounded
  skin-width selection from a global minimum to the first median-smoothed,
  prominent local basin, preventing collar/shoulder dropouts from winning.

### Fixed

- `find_incisor_centroids()` now measures robust, paired points on the facing
  incisal edges instead of whole-tooth centroids. It requires a genuine dark
  gap, validates support after choosing a side, rejects boundary outliers, and
  always returns actual teeth-mask pixels in one shared column.
- Incisor depth sampling can be restricted to the teeth matte and shifted one
  native depth pixel into each tooth, avoiding values from the mouth cavity.
- Photo-to-depth coordinate scaling now maps both image endpoints exactly.

## [0.5.0] - 2026-08-11

### Added

- New `depth_sampling` module: `median_filter_depthmap()`, `bilinear_sample()`,
  `sample_points_along_line()`, `sample_filtered_depth()`, and
  `measure_filtered_surface_length()`. Moved here from fidmaa-gui (which had
  independently built the same primitives for its `surface_vector_filtered`
  measurement) so both the GUI and this library share one, tested
  implementation.
- `.github/workflows/release.yml`: publishes to PyPI via Trusted Publishing
  (OIDC) on GitHub Release publish. No API token stored or required. Needs a
  one-time Trusted Publisher entry added on PyPI's project settings
  (owner=fidmaa, repository=portrait-analyser, workflow=release.yml,
  environment=pypi).

### Changed

- `compute_neck_circumference()`'s arc-length loop now reads depth via a
  once-per-call median-filtered, bilinearly-sampled depth map instead of
  `face.sample_depth_at_point()`'s nearest-neighbour + integer-kernel median.
  This smooths TrueDepth sensor noise before it can accumulate across the
  many points walked along the neck arc — the same class of fix as the
  principal-point correction below, applied to sampling instead of
  positioning. The sag auto-detection step (`_find_best_sag`) is unchanged.

## [0.4.0] - 2026-08-11

### Fixed

- Changed `pixel_to_mm()` to measure pixel coordinates from the image centre
  (the principal-point approximation) instead of the top-left corner. Points
  at different depths no longer pick up a phantom lateral displacement
  proportional to their distance from the image centre and the depth
  difference between them. This corrects `compute_incisor_distance_3d`,
  `compute_tmd_3d`, `compute_neck_circumference`, and
  `extended_neck.compute_neck_width_3d` — the neck-circumference arc is the
  most affected, since it samples points across varying depths along a
  curved surface.

### Changed

- **Breaking:** `pixel_to_mm()` now requires an `image_dimension` argument
  (image width for an x coordinate, image height for a y coordinate).
- **Breaking:** `compute_incisor_distance_3d()` and `compute_tmd_3d()` now
  require `image_width` and `image_height` arguments.

### Tests

- Added `tests/test_principal_point.py`: image centre maps to 0mm at every
  depth, same-depth measurements keep the calibrated pixel scale, motion
  along the optical axis carries no phantom XY component, and
  `compute_incisor_distance_3d` introduces no lateral offset from a large
  absolute pixel position alone.
