# Changelog

All notable changes to portrait-analyser are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- New `depth_sampling` module: `median_filter_depthmap()`, `bilinear_sample()`,
  `sample_points_along_line()`, `sample_filtered_depth()`, and
  `measure_filtered_surface_length()`. Moved here from fidmaa-gui (which had
  independently built the same primitives for its `surface_vector_filtered`
  measurement) so both the GUI and this library share one, tested
  implementation.

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
