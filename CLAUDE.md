# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

This project uses **uv** with Python 3.12 (pinned, `>=3.12,<3.13`).

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file / class / test
uv run pytest tests/test_neck.py
uv run pytest tests/test_neck.py::TestComputeNeckCircumference
uv run pytest tests/test_neck.py::TestComputeNeckCircumference::test_returns_none_without_face

# Build package (sdist + wheel via hatchling)
uv build

# CLI diagnostic tool (inspects a HEIC's raw container, EXIF, depth, segmentation mattes)
uv run analyse-portrait path/to/photo.heic
```

There is no configured linter (no `[tool.ruff]` / pre-commit) — don't invent lint commands.

### Platform-specific HEIF dependency
- **macOS/Darwin**: uses `pyheif-iplweb`
- **Linux**: uses `pyheif` (requires system packages `libheif-dev`, `libde265-dev` — see `.github/workflows/build.yml`)

### Test fixtures
`tests/*.heic` / `*.jpg` are real (anonymized) iPhone Portrait Mode captures used across the suite via fixtures in `tests/conftest.py` (`heic_image_path`, `heic_face_image_path`, `jpeg_depth_data_path`, `jpeg_no_depth_data_path`). Don't add new photo fixtures without stripping GPS/location EXIF first — prior fixtures leaked precise GPS coordinates (see CHANGELOG/git history around the v0.4.x releases).

## Architecture

This library extracts quantitative facial, dental, and neck measurements from iOS Portrait Mode photos (.heic/.heif) captured with Apple's TrueDepth camera. The core idea running through every module: HEIC Portrait Mode files carry more than the visible photo — a **depth map** and Apple's own **semantic segmentation mattes** (teeth, skin) are embedded in the same container — and this library turns those hidden layers into physical (mm/cm) measurements instead of raw pixels.

### Data flow

1. **`ios.py`** `load_image(path)` — parses the HEIC container, validates TrueDepth EXIF (`const.TRUEDEPTH_EXIF_ID`), and returns an `IOSPortrait` with `.photo`, `.depthmap`, `.teethmap`, `.skinmap` (PIL Images) plus `.floatValueMin`/`.floatValueMax` (Apple's depth calibration range). Handles device-specific dimension differences (iPhone 12 vs 14).
2. **`face.py`** `get_face_parameters(image)` — OpenCV Haar-cascade face/eye detection, returns a `Face` (auto-detects `Eye`s on init). This is the geometric anchor most measurement functions key off of.
3. One or more **measurement modules** consume the photo + depth map + semantic maps + face geometry to produce a physical measurement (see below). Most return a dataclass alongside an optional `*Debug` dataclass carrying intermediate state for overlay/debug visualization — always populate/check the debug object when diagnosing a detector rather than re-deriving intermediate values.

### Why three independent neck/chin detectors

The library ships **three unrelated strategies** for locating the neck and chin, because no single signal is reliably present across all captures (skin matte and depth map are sometimes missing/noisy; MediaPipe Pose needs shoulders in frame; segmentation needs a clean silhouette):

- **`face.py`** (`find_neck_measurement_point`, `find_neck_narrowest_row`, `estimate_neck_search_zone`) — Haar-cascade-anchored, walks the skin matte below the detected face for the narrowest horizontal line.
- **`pose.py`** (`detect_neck_midpoint`) — MediaPipe PoseLandmarker + FaceMesh; interpolates between the shoulder midpoint (~C7/T1) and the nose to approximate the mid-cervical level, and separately flags `neck_extended` via a face-flattening ratio. FaceMesh runs independently of Pose, so its debug output can be populated even when pose detection fails.
- **`extended_neck.py`** (`detect_neck_midpoint_from_segmentation`, `detect_neck_midpoint_from_dual_mask`) — MediaPipe Selfie Segmentation silhouette width-profile analysis, or a dual-mask approach combining the iOS skin matte + depth map + optional hair mask.

`neck.py`'s `compute_neck_circumference` is downstream of whichever detector supplied a neck row/x-range: it densely samples the 3D front arc of the neck (skin matte + depth map together, via `depth_sampling.py`) and extrapolates to a full circumference using `circumference_multiplier`.

### The depth → physical-distance pipeline

Several modules share one calibration path for turning raw depth pixel values + pixel coordinates into real-world millimeters — read `incisor.py` before touching any of them:
- `incisor.depth_raw_to_distance_cm` — raw depth byte (0–255) → camera distance in cm, using Apple's disparity encoding and `float_min`/`float_max` from `IOSPortrait`.
- `incisor.pixel_to_mm` — pixel coordinate → mm, via a calibration polynomial fitted to TrueDepth camera data, centered on the image's principal point (`image_dimension` must be the *full* image width/height, not a cropped region's).
- `depth_sampling.py` — robustness layer on top: `median_filter_depthmap` + `bilinear_sample`/`sample_filtered_depth` avoid sampling raw noisy/invalid (zero-disparity) depth pixels directly, and `measure_filtered_surface_length` sums 3D distance across a walked path (used by neck arc integration and could be reused for any multi-point surface measurement).

This pipeline feeds `incisor.compute_incisor_distance_3d` (teeth), `mouth.compute_mouth_measurement_from_facemesh` (lips, fallback when teeth aren't visible), `neck.compute_neck_circumference` / `extended_neck.compute_neck_width_3d` (neck), and `tmd.compute_tmd_3d` (thyromental/chin-to-neck distance, an airway-assessment metric) — all return `(distance_3d_mm, ...per-point_distance_cm)` tuples following the same shape.

### Key conventions

- **Coordinate systems**: face coordinates are image-relative; eye coordinates are face-relative and must go through `translate_coordinates()` to become absolute; teeth/incisor coordinates live in teethmap space and are separately translated via `IOSPortrait.teeth_bbox_translated()`.
- **Semantic maps** (teeth, skin) are mirrored and resized to match photo dimensions before analysis — do this before comparing them pixel-for-pixel against the photo or depth map.
- **PIL Image ↔ OpenCV** conversion goes through `Image_to_cv2()` in `face.py`.
- **Dataclass + Debug pairing**: functions that do multi-step geometric detection (pose, extended_neck, neck) return `(Measurement | None, Debug | None)` rather than raising on a partial detection — check for `None` rather than assuming success.
- `__init__.py` re-exports the full public API flat (no submodule imports needed by consumers); when adding a new public function/class, export it there too and add it to `__all__`.
