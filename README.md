# portrait-analyser

[![Build](https://github.com/fidmaa/portrait-analyser/actions/workflows/build.yml/badge.svg)](https://github.com/fidmaa/portrait-analyser/actions/workflows/build.yml)
[![PyPI Version](https://img.shields.io/pypi/v/portrait-analyser.svg)](https://pypi.org/project/portrait-analyser/)
[![Python Version](https://img.shields.io/pypi/pyversions/portrait-analyser.svg)](https://pypi.org/project/portrait-analyser/)
[![License](https://img.shields.io/pypi/l/portrait-analyser.svg)](LICENSE)

Extract quantitative facial and dental measurements from iOS Portrait Mode photos (.heic/.heif) captured with Apple's TrueDepth camera.

## Why?

iPhone Portrait Mode photos (TrueDepth front camera) carry more than a picture: the HEIC container also stores a depth map and Apple's own semantic segmentation mattes (teeth, skin) used for the bokeh effect. That data is normally locked away — consumer apps only ever show you the blurred photo. `portrait-analyser` unpacks the container and turns those extra layers into physical, reproducible measurements (incisor distance, mouth opening, neck circumference, jaw/chin position) instead of just pixels, which is useful for orthodontic/clinical tracking, research, or any workflow that needs quantitative facial metrics from a phone photo instead of specialized 3D-scanning hardware.

## Features

- **HEIC/HEIF parsing** — decode the primary photo, depth map, and Apple semantic segmentation mattes (teeth, skin) out of the TrueDepth container, with EXIF-based validation that the source is actually a TrueDepth capture
- **Face & eye detection** — OpenCV Haar-cascade based, with coordinate translation between image regions
- **Teeth & incisor analysis** — teeth bounding box, incisor centroid detection, and 3D incisor distance computed from depth data (not just pixel distance)
- **Mouth opening measurement** — MediaPipe FaceMesh-based fallback for patients without visible upper teeth
- **Neck & chin detection** — three independent strategies depending on what's available: MediaPipe Pose (shoulder/nose interpolation), MediaPipe Selfie Segmentation (silhouette width profile), or a dual-mask approach combining the skin matte, depth map, and hair mask
- **3D neck circumference** — dense arc integration over the depth map to estimate physical neck circumference, not just a 2D collar-line width
- **Pose-invariant local landmarks** — robust local-plane removal finds anatomical peaks and valleys without letting mild patient rotation choose the camera-nearest side of a patch
- **Thyromental distance** — physical chin-to-neck-midpoint measurement, a standard airway/intubation-difficulty screening metric
- **CLI diagnostic tool** (`analyse-portrait`) — inspect a HEIC file's raw container, EXIF, depth metadata, and segmentation mattes from the command line

## Requirements

- Python 3.12 (see [Supported versions](#supported-versions))
- iOS Portrait Mode photos in HEIC/HEIF format, taken with a TrueDepth camera (e.g. iPhone 12, 14)

## Installation

### Using uv (recommended)

```bash
uv add portrait-analyser
```

### Using pip

```bash
pip install portrait-analyser
```

### Platform-specific HEIF support

- **macOS** -- works out of the box (uses `pyheif-iplweb`)
- **Linux** -- requires system packages before installing:
  ```bash
  sudo apt install libheif-dev libde265-dev
  ```

## Supported versions

### Python

| 3.12 |
|------|
| ✓    |

## Quick start

```python
from portrait_analyser import load_image, get_face_parameters, find_neck_measurement_point

# Load an iOS Portrait Mode photo
portrait = load_image("photo.heic")

# portrait.photo       -- PIL Image of the photo
# portrait.depth       -- DepthMap: measure depth through this (see below)
# portrait.depthmap    -- PIL Image of the 8-bit depth map (display only for
#                         TrueDepth capture-app files)
# portrait.teethmap    -- PIL Image of the teeth segmentation mask (or None)
# portrait.skinmap     -- PIL Image of the skin segmentation mask (or None)

# Detect face and eyes
face = get_face_parameters(portrait.photo)
print(f"Face at ({face.x}, {face.y}), size {face.width}x{face.height}")
print(f"Eyes detected: {len(face.eyes)}")

# Measure neck width using the skin map
if portrait.skinmap is not None:
    neck = find_neck_measurement_point(portrait.skinmap, face)
    # Returns (x1, y1, x2, y2) of the narrowest horizontal line below the face
```

## API reference

### `load_image(fileName, use_exif=True) -> IOSPortrait`

Parses a HEIC/HEIF file and returns an `IOSPortrait` containing the photo, depth map, and Apple semantic segmentation masks. Validates TrueDepth EXIF data by default.

### `IOSPortrait`

Attributes:
- `photo` -- primary PIL Image
- `depth` -- `DepthMap` to measure with (see "Depth map" below): wraps the 8-bit map for Camera-app files and the full-precision float map for TrueDepth capture-app files
- `depthmap` -- depth map as PIL Image (for capture-app files a display-only 8-bit re-encoding; never measure with it)
- `depth_m` -- full-precision depth in metres (capture-app files only, else `None`)
- `depth_plausible` / `depth_repaired` -- sanity check of capture-app depth, and how it was repaired on load (`"metres stored under a disparity label (iOS 26)"` or `None`), with the evidence in `depth_repair_check`
- `teethmap` -- teeth segmentation mask (PIL Image or `None`)
- `skinmap` -- skin segmentation mask (PIL Image or `None`)
- `teeth_bbox` -- bounding box `(x, y, width, height)` of detected teeth, or `None`
- `teeth_threshold` -- adaptive teeth-matte confidence cut used for this photo (pixels `>=` it count as teeth), or `None` without a teeth matte
- `teeth_arches` -- `TeethArches` (binary teeth `mask`, `bbox`, `threshold`, and `weak_side`/`weak_threshold` when one arch was found only as a faint blob), or `None`
- `incisor_distance` -- incisor measurement as `(x, y1, x, y2)`, or `None`
- `floatValueMin`, `floatValueMax` -- depth map float range from Apple metadata

Methods:
- `teeth_bbox_translated(max_wi, max_he)` -- scale teeth bounding box to a target resolution

### `get_face_parameters(image, raise_opencv_exceptions=False) -> Face`

Detects a single face in a PIL Image using OpenCV Haar cascades. Raises `NoFacesDetected` or `MultipleFacesDetected` if not exactly one face is found.

### `Face` and `Eye`

Both extend `Rectangle` (attributes: `x`, `y`, `width`, `height`, `center_x`, `center_y`).

`Face`:
- `image` -- reference to the source PIL Image
- `eyes` -- list of `Eye` instances (detected automatically)
- `translate_coordinates(new_max_width, new_max_height)` -- scale face coordinates to a target resolution
- `calculate_percentage_of_image()` -- returns `(percent_width, percent_height)`

`Eye`:
- `face` -- reference to the parent `Face`
- `translate_coordinates(max_wi, max_he)` -- absolute coordinates in a target resolution

### Teeth & incisor utility functions

- `find_neck_measurement_point(skinmap, face_location, threshold=200)` -- finds the narrowest horizontal line below the face in the skin map. Returns `(x1, y1, x2, y2)`.
- `teeth_threshold(teethmap, floor=None, cap=None, peak_fraction=None, peak_area_fraction=None) -> int` -- per-matte teeth confidence threshold: half of the matte's robust peak (the K-th brightest pixel, K ≈ 0.014 % of the image area), clipped to `[64, 200]`. Apple's teeth matte gain varies a lot between captures (some peak at ~100–140 with clearly visible teeth), so a fixed cut misses weak mattes. All `threshold`/`min_value`/`support_threshold` arguments below default to `None`, meaning this adaptive value; passing a number keeps the historical fixed-cut behaviour.
- `detect_teeth_arches(teethmap, threshold=None, margin_x=None, margin_y=None, min_height=None, min_area=None, find_weak_arch=True) -> TeethArches | None` -- per-arch teeth detection. Strong teeth use `teeth_threshold`; when they form a single arch, the opposite arch is searched for across the mouth gap (upper side first) in the strong arch's central columns, with a threshold of half that side's own robust peak (at least 8). It is accepted only if it is a separate blob (not the strong arch's soft edge), large and wide enough, and separated by a real empty gap. A faint blob alone never produces a detection. `TeethArches.mask_image()` gives the combined mask for `find_incisor_centroids` / `sample_depth_at_point` (use `threshold=128`). `IncisorMeasurement.weak_arch` / `depth_assumed` report when an arch was weak and when its edge depth fell into the mouth cavity (> 2 cm from the other arch) and the other arch's depth was used instead.
- `find_bounding_box_teeth(teethmap, margin_x=None, margin_y=None, min_value=None, min_height=None, min_area=None)` -- finds the bounding box of teeth in the teeth map. Margins, minimum height and minimum teeth area default to fractions of the matte size; connected components smaller than 10 % of the largest one (speckle) are ignored. With the default `min_value=None` it is the box of `detect_teeth_arches` (including a weak arch). An explicit `min_value` keeps the historical strict `> min_value` comparison and no weak-arch search. Returns `(x, y, width, height)` or `None`.
- `find_incisor_distance_teeth(teethmap, bounding_box_teeth, threshold=None, margin_x=0.5)` -- measures the vertical pixel distance between upper and lower incisors. Returns `(x, y1, x, y2)` or `None`.
- `find_incisor_centroids(teethmap, bounding_box_teeth, threshold=None, margin_x=0.5, min_pixels=50, centroid_margin_x=0.5, ...)` -- finds robust representative points on the facing upper and lower incisal edges. The historical function/field names still use “centroid”, but returned points are snapped to real paired teeth-mask pixels so they measure the inter-incisal gap and provide valid locations for depth sampling. Returns `((upper_x, upper_y), (lower_x, lower_y))` in teethmap coordinates, or `None`.
- `sample_depth_at_point(depthmap, point_x, point_y, photo_width, photo_height, kernel_size=3, support_mask=None, support_threshold=None, inward_y=0) -> int | None` -- samples the depth map at a photo-space coordinate using median filtering over a `kernel_size x kernel_size` region. An optional foreground mask restricts sampling to the intended surface; `inward_y` moves an edge sample inward in native depth-map pixels.

### 3D depth conversion (`incisor` module)

- `depth_raw_to_distance_cm(value, float_min, float_max) -> float | None` -- converts a raw depth pixel value (0-255) to physical distance in centimeters, using Apple's disparity-based depth encoding.
- `pixel_to_mm(pixel_coord, distance_cm, image_dimension) -> float | None` -- converts a pixel coordinate (original, full-resolution image space) to physical millimeters at a given camera distance, via a calibration polynomial fitted to TrueDepth camera data. `image_dimension` is the full image width (for an x coordinate) or height (for a y coordinate), used to centre the conversion on the principal point.
- `vector_length_3d(x1, y1, z1, x2, y2, z2) -> float` -- Euclidean distance between two 3D points.
- `compute_incisor_distance_3d(upper_centroid, lower_centroid, upper_depth_raw, lower_depth_raw, float_min, float_max, image_width, image_height) -> tuple[float, float, float] | None` -- converts two incisor centroids + their raw depth values into physical mm/cm and returns `(distance_3d_mm, upper_distance_cm, lower_distance_cm)`.

### Mouth opening (`mouth` module)

- `compute_mouth_measurement_from_facemesh(landmarks, depthmap, photo_w, photo_h, float_min, float_max) -> MouthMeasurement | None` -- fallback mouth-opening measurement using MediaPipe FaceMesh outer lip landmarks (indices 0 and 17) and the depth map, for cases where teethmap-based incisor detection fails (e.g. no visible upper teeth).
- `MouthMeasurement` -- dataclass with `upper_point`, `lower_point` (photo-space pixels), `upper_depth_raw`, `lower_depth_raw`, `upper_distance_cm`, `lower_distance_cm`, `distance_3d_mm`.

### Neck & chin detection (`pose` module — MediaPipe Pose)

- `detect_neck_midpoint(image, interpolation_ratio=0.35, min_detection_confidence=0.5, min_visibility=0.5) -> tuple[NeckMidpoint | None, MediaPipeDebug | None, FaceMeshDebug | None]` -- locates shoulders and nose via MediaPipe PoseLandmarker, then interpolates between the shoulder midpoint (neck base, ~C7/T1) and the nose to approximate the mid-cervical level (~C3-C4). FaceMesh detection runs independently, so `FaceMeshDebug` may be populated even when pose detection fails.
- `NeckMidpoint` -- dataclass with `nose`, `mouth_left`, `mouth_right`, `chin`, `neck_extended` (True when the neck appears maximally extended, detected via face-flattening ratio), `face_flatness_ratio`, `pose`, `mouth_open_ratio`, plus shoulder-dependent fields (`x`, `y`, `left_shoulder`, `right_shoulder`, visibilities, `interpolation_ratio`) that are `None` when only FaceMesh (not Pose) detected the face.
- `PortraitPose`, `MediaPipeDebug`, `FaceMeshDebug` -- raw MediaPipe landmark containers, useful for debug visualization.

### Neck & chin detection (`extended_neck` module — segmentation-based)

- `detect_neck_midpoint_from_segmentation(image, threshold=0.5, jaw_flare_fraction=0.15, smoothing_window=15) -> tuple[NeckMidpoint | None, SegmentationDebug | None]` -- uses MediaPipe Selfie Segmentation to build a person silhouette, then analyzes the width profile to find the narrowest point (neck) and where the jaw flares out above it (chin).
- `detect_neck_midpoint_from_dual_mask(image, skinmap, depthmap, hairmap=None, threshold=0.5, skin_threshold=30, float_min=None, float_max=None) -> tuple[NeckMidpoint | None, SegmentationDebug | None]` -- combines the iOS skin matte, depth map, and (optional) hair mask: the chin is found as the closest-to-camera skin pixel, neck/shoulders from the depth width profile with hair removed.
- `compute_neck_width_3d(depthmap, neck_y, neck_left_x, neck_right_x, photo_width, photo_height, float_min, float_max, n_samples=25) -> tuple[float | None, float | None]` -- samples N evenly-spaced points across the neck row and converts them to 3D coordinates, returning front-arc length and straight-line width.
- `SegmentationDebug` -- dataclass exposing the binary mask, width profile, and detected neck/chin/shoulder/ear rows for debug visualization.

### Neck circumference (`neck` module — 3D arc integration)

- `compute_neck_circumference(skinmap, depthmap, photo_width, photo_height, float_min, float_max, face_location=None, n_samples=25, skin_threshold=30, circumference_multiplier=3.0, arc_sag=None, face=None, eyes=None, image_width=None, scan_start_y=None, scan_end_y=None, neck_midpoint_y=None, hairmap=None, hair_threshold=30) -> NeckMeasurement | None` -- computes neck circumference by densely sampling the front arc of the neck (using the skin matte and depth map together) and extrapolating to a full circumference. It denoises the skin matte, removes semantic hair, re-reads the contiguous skin boundary at the actual arc-edge Y, and walks inward only across allowed skin until the depth profile stabilizes.
- `find_stable_depth_x_from_edge(depthmap, edge_x, y, direction, photo_width, photo_height, max_distance, stability_run=4, valid_mask=None) -> int | None` -- walks from a left (`direction=1`) or right (`direction=-1`) skin edge in native-depth-pixel steps and returns the centre of the first locally stable depth run. An optional mask prevents stabilization on background or hair.
- `neck_search_bounds_from_face_landmarks(chin=..., nose=..., image_height=..., face_mesh_landmarks=None, pose_neck_y=None) -> tuple[int, int]` -- starts below the lowest FaceMesh row and caps the search using visible face height. A Pose neck estimate may shorten this band but cannot extend it toward the shoulders.
- `estimate_face_from_skinmap(skinmap, threshold=1) -> tuple[int, int, int, int] | None` -- estimates a synthetic face bounding box from the skin segmentation map alone, for when no OpenCV face detection is available.
- `NeckMeasurement` -- dataclass with stable `left_x`, `right_x` sampling coordinates, original `mask_left_x`, `mask_right_x` silhouette coordinates, `neck_y`, `arc_points_3d` (physical mm coordinates), `arc_points_photo` (pixel coordinates, for overlay painting), the surface-polyline `front_arc_length_mm`, and its direct Euclidean `front_chord_length_mm`.
- Within an explicit MediaPipe search band, neck-row selection median-smooths the skin-width profile and chooses the first prominent local minimum rather than a later global minimum caused by a collar or shoulder matte dropout.

### Neck width (`neck_width` module — TrueDepth capture-app photos)

For capture-app files (float depth with absolute accuracy + file intrinsics) only; Camera-app files return `None` and keep the detectors above.

- `measure_neck_width(portrait, *, face_mesh=None, body_pose=None, camera=None, use_vision=True) -> NeckWidthResult | None` -- neck width just below the jaw: per row between the chin (FaceMesh 152) and Apple Vision's neck joint (macOS; otherwise 6 cm below the chin), the outer skin-matte edges around the facial midline, rejected when the depth just outside an edge is nearer than inside (collar), the edge is oblique (collar V), the edges are off-centre or asymmetric about the midline or wider than the jaw (hand next to the neck), or the two edges' depths disagree. Width = median 3-D distance over the topmost run of clean rows spanning at least 2 mm, each edge's depth read 3 mm inside it. `portrait.neck_width` computes it lazily on first access (errors are logged and cached as `status="error"`).
- `neck_width_from_edges(portrait, left_xy, right_xy, *, camera=None) -> NeckWidthResult | None` -- the same width from two clicked edge points (photo px), with the collar warnings.
- `NeckWidthResult` -- `status` (`"ok"`, `"edges-occluded"`, `"no-face"`, `"no-depth"`, `"no-camera"`, `"relative-depth"`, `"error"`), `quality` (`"good"`/`"low"`) with `quality_reasons`, `row_y` (the used row nearest the median), `left_x`, `right_x`, `width_mm`, `rows_used`, `support_mm`, `height_below_chin_mm`, `roll_deg`, `neck_roll_deg`, `band`, `band_source` (`"vision-neck"`, `"chin-offset"`, `"manual"`), `circumference_circle_mm` (π·W, circle model -- not a bound), `circumference_ellipse_mm` (Ramanujan range for b/a 0.85-0.90 -- a population assumption), `warnings`, `message`, `rows` (every evaluated row, for overlays).
- Caveats: `width_mm` reads 0-5 % low (depth read 3 mm inside the silhouette; ~4.5 % on an ideal cylinder). The b/a prior and that bias are co-calibrated on one person only. Repeatability between photos of one person is about 4 % (~±1.6 cm circumference); treat the circumference as indicative.
- `detect_body_pose(photo) -> BodyPose | None` -- Apple Vision body pose on a padded canvas (the photo at 1/3 scale; on the full frame Vision finds no one in a close portrait). Raises `AppleVisionUnavailable` off macOS or without `VNDetectHumanBodyPoseRequest` (macOS < 11).
- `ellipse_circumference(a, b)` -- Ramanujan's ellipse perimeter (was the private `neck._ellipse_circumference`).

### Pose-invariant local surface landmarks (`local_surface` module)

- `score_local_surface_feature(x, y, z, valid, feature, radial_fraction=None, smoothing_size=5, center_bias=0.15) -> LocalSurfaceScores` -- robustly fits and removes the dominant local 3D plane, median-smooths the residual, then ranks a `SurfaceFeature.PEAK` or `SurfaceFeature.VALLEY`. Lower scores are always better. An optional normalized radial distance weakly favours the user's clicked area without overriding a strong off-centre feature.
- `LocalSurfaceScores` -- contains the ranking `score`, detrended `residual`, fitted `baseline`, and final `valid` mask as NumPy arrays.

### Thyromental distance (`tmd` module)

- `compute_tmd_3d(chin_coord, neck_coord, chin_depth_raw, neck_depth_raw, float_min, float_max, image_width, image_height) -> tuple[float, float, float] | None` -- computes the 3D physical distance between chin (mentum) and neck midpoint (a standard airway/intubation-difficulty screening measure), returning `(distance_3d_mm, chin_z_cm, neck_z_cm)`.

### Depth map (`depth_map` module)

`portrait.depth` answers "how far is photo pixel (x, y)?" for every file type.
Coordinates are pixels of the full-resolution upright photo.

```python
depth = portrait.depth                   # LegacyDepthMap or FloatDepthMap
if portrait.depth_plausible is not False:
    z_cm = depth.distance_cm(x, y, radius=1)          # median of 3x3, None = invalid
    smooth = depth.median_filtered()                  # invalid-aware 3x3 median
    profile_cm = smooth.profile(points)               # bilinear cm per point
    length_mm = smooth.surface_length_mm(points, camera=portrait.camera)
    mm, z1, z2 = depth.distance_3d_mm(p1, p2, camera=portrait.camera)
depth.shape, depth.valid_mask, depth.photo_to_depth(x, y)
depth.to_cm_array()        # float cm, NaN = invalid
depth.to_display_image()   # 8-bit, display only
```

- `LegacyDepthMap(image, float_min, float_max, photo_size, *, zero_is_invalid=False)` -- the Camera-app 8-bit disparity map; samples exactly like `sample_depth_at_point` / `median_filter_depthmap` / `sample_filtered_depth`, so results are unchanged.
- `FloatDepthMap(depth_m, photo_size)` -- full-precision metres (NaN, `<= 0` and `> 20 m` invalid), NaN-aware medians and bilinear sampling, no quantisation, no far cap. `profile()` and `surface_length_mm()` integrate over `integration_map(camera)` (3x3 median + edge-preserving bilateral, 6 mm spatial / 20 mm range sigma) because unfiltered TrueDepth depth jitters by ~1 mm per pixel plus a few mm of correlated relief; it rounds features narrower than ~6 mm.
- Measurement functions (`compute_incisor_distance_3d`, `compute_tmd_3d`, `compute_mouth_measurement_from_facemesh`, `compute_neck_circumference`, `compute_neck_width_3d`, `detect_neck_midpoint_from_dual_mask`, `measure_filtered_surface_length`) take a keyword-only `depth=` DepthMap; the old `depthmap`/`float_min`/`float_max` arguments keep working unchanged.
- `repair_inverted_depth(depth_m, skinmap, hairmap=None, *, focal_px=None, photo_size=None)` -- returns `(1 / depth_m, check)` when a capture-app map is implausible but the reciprocal puts the face at a plausible distance *and* width (with the file's focal length) while the map as read does not -- iOS 26 writes metres under a disparity label; `load_image` applies it automatically to absolute capture-app depth.
- Automatic neck circumference on capture-app photos is unvalidated (experimental).

### Robust surface-distance measurement (`depth_sampling` module)

- `median_filter_depthmap(depthmap, size=3) -> Image` -- returns a same-size, single-channel median-filtered copy of a depth map, to be sampled once and reused across many points.
- `bilinear_sample(image, x, y, invalid_value=None) -> float | None` -- samples an image at fractional coordinates using bilinear interpolation; returns `None` if a contributing pixel equals `invalid_value`, instead of interpolating across holes.
- `sample_points_along_line(x1, y1, x2, y2, step) -> Iterator[tuple[float, float]]` -- evenly spaced points along a 2D line, always including both endpoints, independent of point order.
- `sample_filtered_depth(filtered_depthmap, photo_x, photo_y, photo_width, photo_height) -> int | None` -- bilinearly samples a pre-filtered depth map at a photo-space point; `None` over invalid (zero) disparity.
- `measure_filtered_surface_length(filtered_depthmap, points_photo, photo_width, photo_height, float_min, float_max) -> float | None` -- sums 3D Euclidean distance across consecutive photo-space points, sampling depth via `sample_filtered_depth`. Prefiltering + bilinear sampling smooths TrueDepth sensor noise before it can accumulate across many points walked along a surface, which matters for curved or long paths (e.g. `compute_neck_circumference`'s neck arc, or a straight line drawn across a cheek). Returns `None` if fewer than 2 points were given or any point falls on invalid depth.

## Exceptions

- `UnknownExtension` -- file is not .heic or .heif
- `ExifValidationFailed` -- EXIF data does not indicate a TrueDepth camera
- `NoDepthMapFound` -- HEIF container has no depth data
- `NoFacesDetected` -- no face found in image
- `MultipleFacesDetected` -- more than one face found

## Development

```bash
# Clone and set up
git clone https://github.com/fidmaa/portrait-analyser.git
cd portrait-analyser
uv sync

# Run tests
uv run pytest

# Build package
uv build
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT
