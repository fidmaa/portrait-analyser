# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

This project uses **uv** with Python 3.12+.

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Build package
uv build
```

### Platform-specific HEIF dependency
- **macOS/Darwin**: uses `pyheif-iplweb`
- **Linux**: uses `pyheif` (requires system packages `libheif-dev`, `libde265-dev`)

## Architecture

This library extracts quantitative facial and dental measurements from iOS Portrait Mode photos (.heic/.heif) captured with Apple's TrueDepth camera.

### Modules (`src/portrait_analyser/`)

- **ios.py** — `IOSPortrait` dataclass and `load_image()`: parses HEIF containers to extract the primary photo, depth map, and Apple semantic segmentation masks (teeth matte, skin matte). Handles device-specific dimension differences (iPhone 12 vs 14).
- **face.py** — `Face` and `Eye` classes (both extend `Rectangle`): OpenCV Haar cascade-based face/eye detection, coordinate translation between image regions, plus utility functions for neck measurement (`find_neck_measurement_point`) and teeth analysis (`find_bounding_box_teeth`, `find_incisor_distance_teeth`).
- **exceptions.py** — Domain-specific exceptions: `NoFacesDetected`, `MultipleFacesDetected`, `NoDepthMapFound`, `ExifValidationFailed`, `UnknownExtension`.
- **const.py** — `TRUEDEPTH_EXIF_ID` constant for EXIF validation.

### Data flow
1. `load_image(path)` → parses HEIC, validates EXIF (TrueDepth), returns `IOSPortrait` with photo + depth map + semantic maps
2. `get_face_parameters(image)` → detects face via Haar cascade, returns `Face` (auto-detects eyes on init)
3. Utility functions operate on semantic maps for teeth bounding box, incisor distance, and neck width

### Key conventions
- PIL Image ↔ OpenCV conversion via `Image_to_cv2()` in `face.py`
- Coordinate systems: face coordinates are image-relative; eye coordinates are face-relative, translated via `translate_coordinates()`
- Semantic maps (teeth, skin) are mirrored and resized to match photo dimensions before analysis
