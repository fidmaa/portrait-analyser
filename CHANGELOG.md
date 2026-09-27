# Changelog

All notable changes to portrait-analyser are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `neck_width` module: width-based neck measurement for TrueDepth
  capture-app photos (float depth with absolute accuracy + file
  intrinsics). `measure_neck_width(portrait, *, face_mesh=None,
  body_pose=None, camera=None, use_vision=True)` returns a
  `NeckWidthResult` (`status`, `quality` + `quality_reasons`, `row_y`,
  `left_x`, `right_x`, `width_mm`, `rows_used`, `support_mm`,
  `height_below_chin_mm`, `roll_deg`, `neck_roll_deg`, `band`,
  `band_source`, `circumference_circle_mm`, `circumference_ellipse_mm`,
  `warnings`, `message`, per-row `rows` with `reject_codes`,
  `reject_counts`), or None for Camera-app files. An `edges-occluded`
  message gives the advice for the dominant rejection reason.
  Edges come from the skin matte; the depth only validates them (outside
  nearer than inside by 0.5 cm = collar, rejected; 0.2-0.5 cm = "collar
  close to the neck edge", low quality) and gives each edge's depth 3 mm
  inside it. Rows with oblique edges (open-collar V), the depth stepping
  nearer towards an edge or the neck centre more than 12 mm from the
  FaceMesh jaw centre, both back-projected to camera space (a hand beside
  the neck, a turned head; framing off the optical axis does not matter), a span of at
  least 1.5x the jaw width, or left/right depths more than 4 cm apart are
  rejected; a neck wider than 1.2x the jaw (thick neck) or 5-12 mm
  off-centre is measured with low quality (a narrow skin strip flush with
  the neck side is a documented residual risk); the width is the median over the topmost run of clean, stable
  rows spanning at least 2 mm (low quality below 5 mm), else
  `status="edges-occluded"`. Head or neck roll above 8 degrees warns. The
  band runs from the chin (FaceMesh 152) to Apple Vision's neck joint, or
  6 cm below the chin without Vision (`band_source="chin-offset"`).
  Circumference: pi*W (circle model) and a Ramanujan ellipse range for
  b/a 0.85-0.90 (population assumption). `width_mm` reads 0-5 % low (depth
  3 mm inside the silhouette); the b/a prior and this bias are
  co-calibrated on one person, and repeatability between photos is about
  4 %. No front-arc or sagitta model: on frontal portraits they swing
  15-30 % with the row. Float files without intrinsics get
  `status="no-camera"` (pass `camera=` to measure with assumed
  intrinsics); non-absolute depth gets `status="relative-depth"` even with
  `camera=`; a portrait without depth gets `"no-depth"`.
- `neck_width_from_edges(portrait, left_xy, right_xy)`: the same width
  from two clicked edges, with the collar warnings.
- `IOSPortrait.neck_width`: `measure_neck_width` computed lazily on first
  access (load time unchanged; first access ~0.8 s cold for the FaceMesh
  model and the first Vision request, ~0.1 s after). An exception is
  logged and cached as a `status="error"` result.
- `apple_vision` module: `detect_body_pose(photo) -> BodyPose | None`,
  `VNDetectHumanBodyPoseRequest` on a padded canvas (photo at 1/3 scale;
  the full frame of a close portrait yields no person). macOS only, lazy
  pyobjc import; `AppleVisionUnavailable` (also on macOS < 11) /
  `AppleVisionError`.
- `neck.ellipse_circumference()` is public (the old private name remains).
- Dependency: `pyobjc-framework-vision>=12.2` on macOS.

## [0.8.0] - 2026-09-28

### Added

- `depth_map` module: one depth abstraction for every measurement.
  `portrait.depth` is a `DepthMap` answering "camera distance in cm at photo
  pixel (x, y)": `distance_cm(x, y, radius=1)` (neighbourhood median, None =
  invalid), `sample()` (`DepthSample`), `bilinear_cm()`, `profile(points)`,
  `median_filtered(size=3)`, `surface_length_mm(points, camera=)`,
  `distance_3d_mm(p1, p2, camera=)`, `point_3d_mm()`, `valid_mask`, `shape`,
  `photo_to_depth()`, `to_cm_array()` and `to_display_image()` (8-bit,
  display only).
  - `LegacyDepthMap` wraps the Camera-app 8-bit disparity map +
    `FloatMinValue`/`FloatMaxValue` and samples it exactly as before
    (`sample_depth_at_point`, `median_filter_depthmap`,
    `sample_filtered_depth`, `depth_raw_to_distance_cm`): every Camera-app
    number is byte-identical.
  - `FloatDepthMap` wraps the capture app's full-precision `depth_m`
    (metres; NaN, `<= 0` and `> MAX_PLAUSIBLE_DEPTH_M` invalid): NaN-aware
    medians, bilinear interpolation that never crosses a hole, a NaN-aware
    median filter (a pixel is kept when at least half its window is valid),
    no 8-bit quantisation and no 3 m far cap.
- Keyword-only `depth=` (a `DepthMap`, e.g. `portrait.depth`) on
  `compute_incisor_distance_3d`, `compute_tmd_3d`,
  `compute_mouth_measurement_from_facemesh`, `compute_neck_circumference`,
  `compute_neck_width_3d`, `detect_neck_midpoint_from_dual_mask` and
  `measure_filtered_surface_length` (pass an already filtered map there).
  With it the positional `depthmap`/`float_min`/`float_max` are ignored (may
  be None); without it they work exactly as before. `find_stable_depth_x_from_edge`
  also accepts a `DepthMap`.
- `incisor.distance_3d_from_cm()`: the shared two-point 3-D distance from
  known camera distances.
- `repair_inverted_depth()`, `IOSPortrait.depth_repaired` and
  `IOSPortrait.depth_repair_check` (`DepthRepairCheck`). iOS 26 on iPhone
  17's front TrueDepth camera writes HEIC depth labelled "disparity" whose
  values are metres, so it reads as `1/depth` (IMG_2348/2363/2376: face at
  2.4-2.7 m). When absolute capture-app depth fails the plausibility check,
  both interpretations are judged on the face alone with the file's focal
  length. The reciprocal must put the face (skin median) within 15-100 cm
  *and* give a face width (`face_width_px()`: minor axis of the skin
  matte's largest blob) within `PLAUSIBLE_FACE_WIDTH_CM` (10-25 cm); the map
  as read is judged by the face width alone, so a correct capture with the
  face 1.0-1.3 m away (outside the distance window, reciprocal 77-100 cm) is
  never "repaired". Only if the reciprocal passes and the map as read does
  not, `load_image` uses the reciprocal, sets
  `depth_repaired = "metres stored under a disparity label (iOS 26)"`
  (`DEPTH_REPAIRED_RECIPROCAL`) and `depth_plausible = True`, logs the
  evidence, and derives display image, camera and measurements from the
  repaired map (the posterised face in the display image is gone). The
  background comparison is recorded but never vetoes (a board held in front
  makes it fail on IMG_2376). Real files: face width 16.0-16.6 cm repaired vs
  97-117 cm as read; correct files 15.8-16.2 cm. Never for Camera-app or
  relative depth, without intrinsics or a face blob, or when both/neither
  interpretation passes.
- `apple_depth.disparity_encoding_range()`; `analyse-portrait` reports the
  measurement depth kind, plausibility and repair.

### Changed

- Capture-app files are measured on the float depth, not the 8-bit
  re-encoding: `load_image`'s incisor measurements (centroid and legacy
  edge points, weak-arch reconciliation) sample `portrait.depth`. Their
  `IncisorMeasurement.upper_depth_raw`/`lower_depth_raw` (and the mouth
  measurement's) are now None -- float depth has no codes; use the
  `*_distance_cm` fields. On the real captures the incisor distance moves by
  +0.25 mm (IMG_2346: 42.92 -> 43.17 mm) and +0.42 mm (IMG_2347: 47.40 ->
  47.82 mm) and no longer depends on the encoding range (8-bit results
  varied by up to 0.4 mm with the far cap).
- For capture-app files `depthmap`, `floatValueMin` and `floatValueMax` are
  documented as a display-only encoding (still produced exactly as in
  0.7.0); `depth_valid_mask` still describes that image, use
  `portrait.depth.valid_mask` for the measurement map.
- The chin/body/stable-edge/arc-sag detectors read a "detector scale" from
  the `DepthMap` (`code_array`, `bilinear_code`, `DepthSample.code`): the
  stored 8-bit codes for Camera-app maps; for float maps the same 0-255
  disparity scale, unquantised, on the fixed `FLOAT_DETECTOR_CODE_RANGE`
  (1/3 m .. 1/0.25 m) -- independent of the file's own range and of the
  display encoding. Automatic neck circumference on capture-app files is
  unvalidated (experimental).
- Float depth is smoothed before anything integrates along it:
  `DepthMap.integration_map(camera)` = 3x3 NaN-aware median + NaN-aware
  bilateral filter (spatial sigma 6 mm via the file's focal length at the
  subject's distance -- `FloatDepthMap.subject_depth_m`, set by `load_image`
  to the skin-matte median, so a near foreground cannot shrink it --
  `INTEGRATION_SIGMA_MM`; range sigma 20 mm, `INTEGRATION_RANGE_SIGMA_MM`, so
  silhouettes are never blended with the background; pixels beyond 3 m are
  left unsmoothed). `FloatDepthMap.profile()` / `surface_length_mm()`,
  `measure_filtered_surface_length(depth=...)` and the float neck arc always
  use it; legacy maps keep the plain median filter. Unfiltered TrueDepth
  depth shows ~1 mm per-pixel jitter plus +-2-4 mm relief correlated over
  5-10 mm. Calibrated on a flat, PnP-verified ChArUco board (IMG_2376, six
  100-150 mm lines): surface/linear 1.21-1.40 with the median only,
  1.010-1.022 now. FaceMesh eye-corner line (33 -> 263) surface/linear:
  IMG_2346 1.70 -> 1.31, IMG_2347 1.63 -> 1.37, IMG_2348 1.45 -> 1.25
  (Apple-filtered Camera-app depth ~1.29); forehead lines 1.26-1.28 ->
  1.07-1.10; short cheek lines 1.00-1.02.
  - Limitations: the smoothing rounds tight curvature -- a noise-free
    r = 40 mm sphere's arc over +-0.8 r comes out 2.0 % short (apex 0.7 mm
    back), a r = 60 mm cylinder's 0.4 % short; features narrower than
    ~6 mm (lips, eyelids, the nasal ridge) are flattened. Surfaces seen at
    grazing angles and depth steps below ~40 mm are partly averaged across.
    The board is the only real ground truth so far.
- Float neck arcs return None when more than `MAX_DROPPED_ARC_FRACTION`
  (20 %) of the arc points have no depth (legacy unchanged).

## [0.7.0] - 2026-09-27

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
  - The 8-bit encoding's near end is the minimum of the 3x3 NaN-aware median
    of valid depths (`NEAR_END_MEDIAN_SIZE`); nearer pixels (< 1 mm on real
    captures) saturate at 255, so one stray pixel no longer coarsens the
    quantisation while the nose tip is kept (IMG_2346: 29.49 vs raw min
    29.39 cm; a 0.1st percentile would clip it to 30.37 cm).
  - `read_apple_depth` turns unexpected pyobjc shapes (Attribute/Index/Key/
    Type/ValueError) into `AppleDepthDecodeError`; a Camera-app load never
    fails because of it (logged, `depth_accuracy = None`).
  - `portrait.camera` / `CameraModel.from_portrait()` return None when
    `depth_plausible is False` (second line of defence).
  - Whether code 0 is invalid follows the file format, not camera presence:
    `compute_incisor_distance_3d`, `compute_tmd_3d`,
    `compute_mouth_measurement_from_facemesh` and `raw_depth_to_distance_cm`
    take keyword-only `zero_is_invalid=None` (pass
    `portrait.depth_code_zero_is_invalid`; None infers it from `camera`), so
    relative capture-app files are covered too.
  - pyobjc's own `objc.error` is also reported as `AppleDepthDecodeError`.
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
