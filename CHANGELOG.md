# Changelog

All notable changes to portrait-analyser are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
