# portrait-analyser

Extract quantitative facial and dental measurements from iOS Portrait Mode photos (.heic/.heif) captured with Apple's TrueDepth camera.

## Requirements

- Python 3.12+
- iOS Portrait Mode photos in HEIC/HEIF format, taken with a TrueDepth camera (e.g. iPhone 12, 14)

## Installation

```bash
pip install portrait-analyser
```

### Platform-specific HEIF support

- **macOS** -- works out of the box (uses `pyheif-iplweb`)
- **Linux** -- requires system packages before installing:
  ```bash
  sudo apt install libheif-dev libde265-dev
  ```

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

### Utility functions

- `find_neck_measurement_point(skinmap, face_location, threshold=200)` -- finds the narrowest horizontal line below the face in the skin map. Returns `(x1, y1, x2, y2)`.
- `find_bounding_box_teeth(teethmap, margin_x=100, margin_y=100, min_value=200)` -- finds the bounding box of teeth in the teeth map. Returns `(x, y, width, height)` or `None`.
- `find_incisor_distance_teeth(teethmap, bounding_box_teeth, threshold=200, margin_x=0.25)` -- measures the vertical distance between upper and lower incisors. Returns `(x, y1, x, y2)` or `None`.

## Exceptions

- `UnknownExtension` -- file is not .heic or .heif
- `ExifValidationFailed` -- EXIF data does not indicate a TrueDepth camera
- `NoDepthMapFound` -- HEIF container has no depth data
- `NoFacesDetected` -- no face found in image
- `MultipleFacesDetected` -- more than one face found

## Development

```bash
# Clone and set up
git clone https://github.com/mpasternak/portrait-analyser.git
cd portrait-analyser
uv sync

# Run tests
uv run pytest

# Build package
uv build
```

## License

MIT
