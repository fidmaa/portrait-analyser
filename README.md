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
# portrait.depthmap    -- PIL Image of the depth map
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
- `depthmap` -- depth map as PIL Image
- `teethmap` -- teeth segmentation mask (PIL Image or `None`)
- `skinmap` -- skin segmentation mask (PIL Image or `None`)
- `teeth_bbox` -- bounding box `(x, y, width, height)` of detected teeth, or `None`
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
- `find_bounding_box_teeth(teethmap, margin_x=100, margin_y=100, min_value=200)` -- finds the bounding box of teeth in the teeth map. Returns `(x, y, width, height)` or `None`.
- `find_incisor_distance_teeth(teethmap, bounding_box_teeth, threshold=200, margin_x=0.5)` -- measures the vertical pixel distance between upper and lower incisors. Returns `(x, y1, x, y2)` or `None`.
- `find_incisor_centroids(teethmap, bounding_box_teeth, threshold=200, margin_x=0.5, min_pixels=50, centroid_margin_x=0.5)` -- finds the centroids of the upper and lower incisor surfaces. Returns `((upper_cx, upper_cy), (lower_cx, lower_cy))` in teethmap coordinates, or `None`.
- `sample_depth_at_point(depthmap, point_x, point_y, photo_width, photo_height, kernel_size=3) -> int | None` -- samples the depth map at a photo-space coordinate using median filtering over a `kernel_size x kernel_size` region.

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

- `compute_neck_circumference(skinmap, depthmap, photo_width, photo_height, float_min, float_max, face_location=None, n_samples=25, skin_threshold=1, circumference_multiplier=2.7, arc_sag=None, face=None, eyes=None, image_width=None, scan_start_y=None, scan_end_y=None, neck_midpoint_y=None) -> NeckMeasurement | None` -- computes neck circumference by densely sampling the front arc of the neck (using the skin matte and depth map together) and extrapolating to a full circumference. It walks inward from each matte edge until the depth profile stabilizes, avoiding the TrueDepth silhouette wall.
- `find_stable_depth_x_from_edge(depthmap, edge_x, y, direction, photo_width, photo_height, max_distance, stability_run=4) -> int | None` -- walks from a left (`direction=1`) or right (`direction=-1`) skin edge in native-depth-pixel steps and returns the centre of the first locally stable depth run.
- `estimate_face_from_skinmap(skinmap, threshold=1) -> tuple[int, int, int, int] | None` -- estimates a synthetic face bounding box from the skin segmentation map alone, for when no OpenCV face detection is available.
- `NeckMeasurement` -- dataclass with stable `left_x`, `right_x` sampling coordinates, original `mask_left_x`, `mask_right_x` silhouette coordinates, `neck_y`, `arc_points_3d` (physical mm coordinates), and `arc_points_photo` (pixel coordinates, for overlay painting).

### Pose-invariant local surface landmarks (`local_surface` module)

- `score_local_surface_feature(x, y, z, valid, feature, radial_fraction=None, smoothing_size=5, center_bias=0.15) -> LocalSurfaceScores` -- robustly fits and removes the dominant local 3D plane, median-smooths the residual, then ranks a `SurfaceFeature.PEAK` or `SurfaceFeature.VALLEY`. Lower scores are always better. An optional normalized radial distance weakly favours the user's clicked area without overriding a strong off-centre feature.
- `LocalSurfaceScores` -- contains the ranking `score`, detrended `residual`, fitted `baseline`, and final `valid` mask as NumPy arrays.

### Thyromental distance (`tmd` module)

- `compute_tmd_3d(chin_coord, neck_coord, chin_depth_raw, neck_depth_raw, float_min, float_max, image_width, image_height) -> tuple[float, float, float] | None` -- computes the 3D physical distance between chin (mentum) and neck midpoint (a standard airway/intubation-difficulty screening measure), returning `(distance_3d_mm, chin_z_cm, neck_z_cm)`.

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
