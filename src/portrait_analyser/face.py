import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy
from PIL import Image

from .exceptions import MultipleFacesDetected, NoFacesDetected


@dataclass
class IncisorMeasurement:
    upper_centroid: tuple[float, float]  # (x, y) in photo/teethmap coordinates
    lower_centroid: tuple[float, float]  # (x, y) in photo/teethmap coordinates
    upper_depth_raw: int | None = None  # raw pixel value from depth map
    lower_depth_raw: int | None = None
    upper_distance_cm: float | None = None  # physical distance from camera (cm)
    lower_distance_cm: float | None = None
    distance_3d_mm: float | None = None  # 3D Euclidean distance between centroids
    pixel_distance_y: float = 0.0  # legacy-style vertical pixel gap


_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
_CACHE_DIR = Path.home() / ".cache" / "portrait-analyser"
_FACE_MODEL_FILENAME = "blaze_face_short_range.tflite"


def _get_face_model_path() -> str:
    """Return path to the FaceDetector .tflite model, downloading if needed."""
    model_path = _CACHE_DIR / _FACE_MODEL_FILENAME
    if not model_path.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = model_path.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(_FACE_MODEL_URL, tmp_path)
            os.replace(tmp_path, model_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    return str(model_path)


def translate_coordinates(
    rect, new_max_width, new_max_height, image_width, image_height
):
    mul_x = new_max_width / image_width
    mul_y = new_max_height / image_height

    new_x1 = rect.x * mul_x
    new_y1 = rect.y * mul_y
    new_x2 = (rect.x + rect.width) * mul_x
    new_y2 = (rect.y + rect.height) * mul_y

    new_width = new_x2 - new_x1
    new_height = new_y2 - new_y1

    return (new_x1, new_y1, new_width, new_height)


class Rectangle:
    def __init__(self, x, y, wi, he):
        self.x = x
        self.y = y
        self.width = wi
        self.height = he

        self.center_x = x + wi / 2
        self.center_y = y + he / 2

    def __str__(self):
        return f"{id(self)} at {self.x}:{self.y}, {self.width}x{self.height}"

    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.width
        elif i == 3:
            return self.height
        else:
            raise IndexError


class Eye(Rectangle):
    def __init__(self, face, *args, **kw):
        super().__init__(*args, **kw)
        self.face = face

    def translate_coordinates(self, max_wi, max_he):
        rect = self.face.translate_coordinates(max_wi, max_he)
        self_rect = translate_coordinates(
            self, max_wi, max_he, self.face.image.size[0], self.face.image.size[1]
        )
        return (
            rect[0] + self_rect[0],
            rect[1] + self_rect[1],
            self_rect[2],
            self_rect[3],
        )

    def get_image_for_analysis(self, mode="gray"):
        if mode == "gray":
            arr = numpy.array(self.face.image.convert("L"))
        else:
            arr = numpy.array(self.face.image.convert("RGB"))
        return arr[
            self.face.y + self.y : self.y + self.face.y + self.height,
            self.x + self.face.x : self.x + self.face.x + self.width,
        ].copy()


class Face(Rectangle):
    def __init__(self, image, *args, eyes=None, **kw):
        super().__init__(*args, **kw)
        self.image = image
        if eyes is not None:
            self.eyes = eyes
        else:
            self.find_eyes()

    def translate_coordinates(self, new_max_width, new_max_height):
        return translate_coordinates(
            self,
            new_max_width,
            new_max_height,
            self.image.size[0],
            self.image.size[1],
        )

    def calculate_percentage_of_image(self):
        """How much of the image is the face?"""

        img_wi, img_he = self.image.size  # [:2]
        percent_width = float(self.width) / float(img_wi)
        percent_height = float(self.height) / float(img_he)

        return percent_width, percent_height

    def get_image_for_analysis(self, mode="gray"):
        if mode == "gray":
            arr = numpy.array(self.image.convert("L"))
        else:
            arr = numpy.array(self.image.convert("RGB"))
        return arr[
            self.y : self.y + self.height - 1,
            self.x : self.x + self.width - 1,
        ]

    def find_eyes(self):
        self.eyes = []


def detect_eyes(image):
    """Detect eyes in the full image without face detection.

    Returns list of Rectangle objects in image-absolute coordinates.
    Useful as a fallback when face detection fails (NoFacesDetected).
    """
    import mediapipe as mp

    model_path = _get_face_model_path()
    image_array = numpy.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        min_detection_confidence=0.5,
    )

    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        detection_result = detector.detect(mp_image)

    img_w, img_h = image.size
    eyes = []
    for detection in detection_result.detections:
        keypoints = detection.keypoints
        if keypoints and len(keypoints) >= 2:
            face_w = detection.bounding_box.width
            face_h = detection.bounding_box.height
            for i in [0, 1]:  # left eye, right eye keypoints
                kp = keypoints[i]
                eye_x = kp.x * img_w
                eye_y = kp.y * img_h
                eye_w = face_w * 0.15
                eye_h = face_h * 0.08
                eyes.append(Rectangle(
                    int(eye_x - eye_w / 2),
                    int(eye_y - eye_h / 2),
                    int(eye_w),
                    int(eye_h),
                ))
    return eyes


def get_face_parameters(
    input_image: Image.Image, raise_opencv_exceptions=False
):
    """Get face position and size or return an exception in
    case there's none."""
    import mediapipe as mp

    model_path = _get_face_model_path()

    image_array = numpy.array(input_image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        min_detection_confidence=0.5,
    )

    try:
        with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
            result = detector.detect(mp_image)
        detections = result.detections
    except Exception:
        if raise_opencv_exceptions:
            raise
        detections = []

    if len(detections) == 0:
        raise NoFacesDetected()

    if len(detections) > 1:
        raise MultipleFacesDetected()

    detection = detections[0]
    img_w, img_h = input_image.size
    bbox = detection.bounding_box
    x = int(bbox.origin_x)
    y = int(bbox.origin_y)
    w = int(bbox.width)
    h = int(bbox.height)

    # Extract eye keypoints from MediaPipe face detection
    # Keypoints: 0=left eye, 1=right eye, 2=nose tip, 3=mouth center,
    # 4=left ear tragion, 5=right ear tragion
    eyes = []
    keypoints = detection.keypoints
    if keypoints and len(keypoints) >= 2:
        for i in [0, 1]:
            kp = keypoints[i]
            eye_x_abs = kp.x * img_w
            eye_y_abs = kp.y * img_h
            eye_x_rel = eye_x_abs - x
            eye_y_rel = eye_y_abs - y
            eye_w = w * 0.15
            eye_h = h * 0.08
            eyes.append(Eye(
                None,  # face reference set below
                int(eye_x_rel - eye_w / 2),
                int(eye_y_rel - eye_h / 2),
                int(eye_w),
                int(eye_h),
            ))

    face = Face(input_image, x, y, w, h, eyes=eyes)
    for eye in face.eyes:
        eye.face = face

    return face


def _gaussian_kernel(sigma, truncate=4.0):
    """Create a 1D Gaussian kernel for smoothing (no scipy dependency)."""
    radius = int(truncate * sigma + 0.5)
    x = numpy.arange(-radius, radius + 1, dtype=float)
    kernel = numpy.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def find_neck_narrowest_row(
    skinmap,
    search_zone=None,
    face_location=None,
    threshold=1,
    smooth_sigma=3.0,
    max_rows=200,
):
    """Find the collar line where skin disappears in the center strip.

    Scans downward from the chin and detects where the central neck skin
    ends (clothing begins). Uses a center strip to avoid being fooled by
    wide shoulders or V-neck gaps. Requires 3 consecutive rows without
    center skin to confirm the collar.

    Parameters
    ----------
    skinmap : PIL Image "L"
        Skin segmentation map.
    search_zone : tuple (center_x, scan_start_y, scan_width) or None
        Eye-anchored search zone from estimate_neck_search_zone().
    face_location : tuple or Rectangle (x, y, w, h) or None
        Fallback face bounding box. Used if search_zone is None.
    threshold : int
        Minimum pixel value to count as skin.
    smooth_sigma : float
        Unused. Kept for API compatibility.
    max_rows : int
        Maximum number of rows to scan below the start point.

    Returns
    -------
    (x_left, neck_y, x_right, neck_y) or None if no neck found.
    """
    img_width, img_height = skinmap.size
    arr = numpy.array(skinmap)

    # Determine scan parameters
    if search_zone is not None:
        center_x, scan_start_y, scan_width = search_zone
        x_scan_left = max(0, center_x - scan_width // 2)
        x_scan_right = min(img_width, center_x + scan_width // 2)
    elif face_location is not None:
        scan_start_y = face_location[1] + face_location[3]
        x_scan_left = 0
        x_scan_right = img_width
    else:
        return None

    scan_end_y = min(img_height, scan_start_y + max_rows)
    if scan_start_y >= scan_end_y:
        return None

    # For each row, find leftmost and rightmost skin pixels (full width scan)
    widths = []
    lefts = []
    rights = []
    y_coords = []

    for y in range(scan_start_y, scan_end_y):
        row = arr[y, x_scan_left:x_scan_right]
        skin_cols = numpy.where(row >= threshold)[0]
        if len(skin_cols) == 0:
            widths.append(0)
            lefts.append(0)
            rights.append(0)
            y_coords.append(y)
            continue

        left = int(skin_cols[0]) + x_scan_left
        right = int(skin_cols[-1]) + x_scan_left
        widths.append(right - left)
        lefts.append(left)
        rights.append(right)
        y_coords.append(y)

    widths_arr = numpy.array(widths, dtype=float)

    # Filter to only valid (non-zero) rows to avoid boundary artifacts
    # where zero-width rows would pull down smoothed values of neighbors
    valid_mask = widths_arr > 0
    if not numpy.any(valid_mask):
        return None

    valid_indices = numpy.where(valid_mask)[0]
    # Find collar line: where skin disappears in the center strip.
    if search_zone is not None:
        strip_center = center_x
    else:
        first_valid = valid_indices[0]
        strip_center = (lefts[first_valid] + rights[first_valid]) // 2

    strip_half = max(10, int((x_scan_right - x_scan_left) * 0.15))
    strip_left = max(x_scan_left, strip_center - strip_half)
    strip_right = min(x_scan_right, strip_center + strip_half)
    min_center_pixels = 3

    last_skin_idx = None
    gap_count = 0
    collar_gap_rows = 3

    for i in range(len(y_coords)):
        y = y_coords[i]
        center_row = arr[y, strip_left:strip_right]
        center_skin = numpy.count_nonzero(center_row >= threshold)

        if center_skin >= min_center_pixels:
            last_skin_idx = i
            gap_count = 0
        else:
            gap_count += 1
            if gap_count >= collar_gap_rows and last_skin_idx is not None:
                break

    if last_skin_idx is None:
        return None

    neck_y = y_coords[last_skin_idx]
    x_left = lefts[last_skin_idx]
    x_right = rights[last_skin_idx]

    if x_right <= x_left:
        return None

    return (x_left, neck_y, x_right, neck_y)


def estimate_neck_search_zone(face=None, *, eyes=None, image_width=None):
    """Estimate where to search for the neck based on detected eyes.

    Uses interpupillary distance (IPD) and facial proportions to estimate
    chin position and neck search zone, bypassing the unreliable face box.

    Can be called in two ways:
    - face provided: reads face.eyes, uses face.x/face.y as offset
    - eyes + image_width provided: eyes are image-absolute (standalone detection)

    Returns (center_x, scan_start_y, scan_width) or None if <2 usable eyes.
    """
    if face is not None:
        eye_list = face.eyes
        offset_x = face.x
        offset_y = face.y
        ref_width = face.width
    elif eyes is not None and image_width is not None:
        eye_list = eyes
        offset_x = 0
        offset_y = 0
        ref_width = image_width
    else:
        return None

    if len(eye_list) < 2:
        return None

    # Pick the best eye pair: closest Y with reasonable X separation
    sorted_eyes = sorted(eye_list, key=lambda e: e.center_y)
    best_pair = None
    best_y_diff = float("inf")

    for i in range(len(sorted_eyes)):
        for j in range(i + 1, len(sorted_eyes)):
            x_sep = abs(sorted_eyes[i].center_x - sorted_eyes[j].center_x)
            y_diff = abs(sorted_eyes[i].center_y - sorted_eyes[j].center_y)
            # Require reasonable horizontal separation (at least 20% of ref width)
            if x_sep < ref_width * 0.2:
                continue
            if y_diff < best_y_diff:
                best_y_diff = y_diff
                best_pair = (sorted_eyes[i], sorted_eyes[j])

    if best_pair is None:
        return None

    e1, e2 = best_pair
    # Convert to image-absolute coordinates
    mid_x = offset_x + (e1.center_x + e2.center_x) / 2
    mid_y = offset_y + (e1.center_y + e2.center_y) / 2
    ipd = abs(e1.center_x - e2.center_x)

    # Chin is ~1.2 IPD below eye midpoint (standard facial proportions)
    chin_y = mid_y + 1.2 * ipd
    # Start scanning at chin estimate (narrowest point is typically 0.1-0.3 IPD below)
    scan_start_y = int(chin_y)
    # Search width: 3.0x IPD centered on eye midpoint (wide enough for shoulders)
    scan_width = int(3.0 * ipd)

    return (int(mid_x), scan_start_y, scan_width)


def find_narrowest_skin_row(
    skinmap, scan_start_y, scan_end_y, threshold=1,
):
    """Find the narrowest skin row between scan_start_y and scan_end_y.

    Scans every row in the given range and finds the one with the minimum
    horizontal skin extent (leftmost to rightmost skin pixel). No center-strip
    logic, no collar detection — just the minimum-width row that still has skin.

    Parameters
    ----------
    skinmap : PIL Image "L"
        Skin segmentation map.
    scan_start_y : int
        First row to scan (inclusive).
    scan_end_y : int
        Last row to scan (exclusive).
    threshold : int
        Minimum pixel value to count as skin.

    Returns
    -------
    (x_left, neck_y, x_right, neck_y) at the narrowest row, or None if no
    skin rows are found in the range.
    """
    arr = numpy.array(skinmap)
    img_height, img_width = arr.shape[:2]

    scan_start_y = max(0, scan_start_y)
    scan_end_y = min(img_height, scan_end_y)

    if scan_start_y >= scan_end_y:
        return None

    best_width = None
    best_left = 0
    best_right = 0
    best_y = 0

    for y in range(scan_start_y, scan_end_y):
        row = arr[y, :]
        skin_cols = numpy.where(row >= threshold)[0]
        if len(skin_cols) == 0:
            continue

        left = int(skin_cols[0])
        right = int(skin_cols[-1])
        width = right - left

        if width <= 0:
            continue

        if best_width is None or width < best_width:
            best_width = width
            best_left = left
            best_right = right
            best_y = y

    if best_width is None:
        return None

    return (best_left, best_y, best_right, best_y)


def find_neck_measurement_point(
    skinmap, face_location=None, threshold=1, face=None, smooth_sigma=3.0,
    eyes=None, image_width=None,
    scan_start_y=None, scan_end_y=None,
):
    """Find the neck measurement row below the face.

    When ``scan_start_y`` and ``scan_end_y`` are both provided (MediaPipe
    bounds), uses :func:`find_narrowest_skin_row` to locate the actual
    narrowest skin row within that range.  Falls through to the existing
    collar-detection logic if that returns None.

    When a Face object is available (via `face` kwarg, or when `face_location`
    is itself a Face instance with eyes), uses eye-anchored search zone for
    robust detection. Falls back to face-box-based search otherwise.

    When `eyes` and `image_width` are provided (standalone eye detection,
    no face available), uses those for eye-anchored search.

    Returns (x_left, neck_y, x_right, neck_y).
    Raises IndexError if no valid neck row is found.
    """
    # When MediaPipe bounds are provided, try narrowest-row search first
    if scan_start_y is not None and scan_end_y is not None:
        result = find_narrowest_skin_row(
            skinmap, scan_start_y, scan_end_y, threshold=threshold,
        )
        if result is not None:
            return result

    # Auto-detect Face object: callers often pass Face as face_location
    if face is None and hasattr(face_location, "eyes"):
        face = face_location

    search_zone = None
    if face is not None:
        search_zone = estimate_neck_search_zone(face)
    if search_zone is None and eyes is not None and image_width is not None:
        search_zone = estimate_neck_search_zone(eyes=eyes, image_width=image_width)

    result = find_neck_narrowest_row(
        skinmap,
        search_zone=search_zone,
        face_location=face_location,
        threshold=threshold,
        smooth_sigma=smooth_sigma,
    )

    if result is not None:
        return result

    # If numpy scan found nothing, raise IndexError for backward compatibility
    raise IndexError("No valid neck row found")


def find_bounding_box_teeth(teethmap, margin_x=100, margin_y=100, min_value=200):
    min_teeth_x = None
    min_teeth_y = None
    max_teeth_x = None
    max_teeth_y = None

    for y in range(margin_y, teethmap.size[1] - margin_y):
        for x in range(margin_x, teethmap.size[0] - margin_x):
            if teethmap.getpixel((x, y)) > min_value:
                if min_teeth_x is None or min_teeth_x > x:
                    min_teeth_x = x
                if max_teeth_x is None or max_teeth_x < x:
                    max_teeth_x = x

                if min_teeth_y is None or min_teeth_y > y:
                    min_teeth_y = y
                if max_teeth_y is None or max_teeth_y < y:
                    max_teeth_y = y

    if max_teeth_y == teethmap.size[1] - margin_y - 1:
        # bottom not found!
        return

    if min_teeth_x is None:
        return

    if max_teeth_y - min_teeth_y < 200:
        return

    return (
        min_teeth_x,
        min_teeth_y,
        max_teeth_x - min_teeth_x,
        max_teeth_y - min_teeth_y,
    )


def find_incisor_distance_teeth(
    teethmap, bounding_box_teeth, threshold=200, margin_x=0.5
):
    """Find incisor distance from teeth map.

    Iterate from teethmap x1 to x1 + width, starting from
    the half of it, try finding points with MAXIMAL distance
    as long as they are within bounding_box_teeth and their
    value is above 200 (well-detected teeth, to avoid
    diasthemes which would probably be the highest distance
    points, but that's not what we're looking for...).
    """

    y_mid = bounding_box_teeth[1] + bounding_box_teeth[3] / 2
    min_he = bounding_box_teeth[1]
    max_he = bounding_box_teeth[1] + bounding_box_teeth[3]

    x_start = int(
        bounding_box_teeth[0]
        + bounding_box_teeth[2] / 2
        - margin_x * bounding_box_teeth[2] / 2
    )
    x_end = int(
        bounding_box_teeth[0]
        + bounding_box_teeth[2] / 2
        + margin_x * bounding_box_teeth[2] / 2
    )

    found_values = []
    for x in range(x_start, x_end):
        upper_y, lower_y = y_mid, y_mid

        while upper_y > min_he:
            upper_y -= 1
            value = teethmap.getpixel((x, upper_y))
            if value >= threshold:
                break

        if upper_y <= min_he:
            # No upper teeth found!
            continue

        while lower_y < max_he:
            lower_y += 1
            value = teethmap.getpixel((x, lower_y))
            if value >= threshold:
                break

        if lower_y >= max_he:
            # No lower teeth found!
            continue

        distance = lower_y - upper_y
        found_values.append((distance, x, upper_y, lower_y))

    if not found_values:
        return

    found_values.sort()
    _, x, y1, y2 = found_values.pop()
    return (x, y1, x, y2)


def _pick_side(xs, ys, x_midline, use_left):
    """Keep only the left or right half of pixel coordinates.

    Used to avoid the septum (dark gap between central incisors).
    The caller decides which side to use so that upper and lower
    centroids are on the same side (left-to-left or right-to-right).
    """
    if use_left:
        mask = xs < x_midline
    else:
        mask = xs >= x_midline
    return xs[mask], ys[mask]


def find_incisor_centroids(
    teethmap,
    bounding_box_teeth,
    threshold=200,
    margin_x=0.5,
    min_pixels=50,
    centroid_margin_x=0.5,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Find centroids of upper and lower incisor surfaces.

    Returns ((upper_cx, upper_cy), (lower_cx, lower_cy)) in teethmap coordinates,
    or None if centroids cannot be determined.
    """
    bb_x, bb_y, bb_w, bb_h = bounding_box_teeth
    y_mid = bb_y + bb_h / 2

    # Determine the gap midline using median of per-column midpoints
    x_start = int(bb_x + bb_w / 2 - margin_x * bb_w / 2)
    x_end = int(bb_x + bb_w / 2 + margin_x * bb_w / 2)
    min_he = bb_y
    max_he = bb_y + bb_h

    gap_midpoints = []
    for x in range(x_start, x_end):
        upper_y = y_mid
        while upper_y > min_he:
            upper_y -= 1
            if teethmap.getpixel((x, int(upper_y))) >= threshold:
                break
        else:
            continue

        lower_y = y_mid
        while lower_y < max_he:
            lower_y += 1
            if teethmap.getpixel((x, int(lower_y))) >= threshold:
                break
        else:
            continue

        if upper_y > min_he and lower_y < max_he:
            gap_midpoints.append((upper_y + lower_y) / 2)

    if not gap_midpoints:
        return None

    gap_midline = sorted(gap_midpoints)[len(gap_midpoints) // 2]  # median

    # Extract ROI restricted to central incisor strip (excludes lateral teeth
    # that form an arch and would pull centroids into the mouth cavity)
    arr = numpy.array(teethmap)
    cx_start = int(bb_x + bb_w / 2 - centroid_margin_x * bb_w / 2)
    cx_end = int(bb_x + bb_w / 2 + centroid_margin_x * bb_w / 2)
    roi = arr[bb_y : bb_y + bb_h, cx_start:cx_end]

    ys, xs = numpy.where(roi >= threshold)
    if len(ys) == 0:
        return None

    # Convert to teethmap coordinates
    abs_xs = xs + cx_start
    abs_ys = ys + bb_y

    # Split into upper and lower groups
    upper_mask = abs_ys < gap_midline
    lower_mask = abs_ys > gap_midline

    upper_xs = abs_xs[upper_mask]
    upper_ys = abs_ys[upper_mask]
    lower_xs = abs_xs[lower_mask]
    lower_ys = abs_ys[lower_mask]

    if len(upper_xs) < min_pixels or len(lower_xs) < min_pixels:
        return None

    # Avoid the septum (dark gap between central incisors): split pixels
    # into left/right halves at the X midline, then keep the same side for
    # both upper and lower groups. This ensures the centroid pair measures
    # left-upper to left-lower (or right to right), not across the septum.
    # The side is chosen by total pixel count across both groups.
    x_midline = (cx_start + cx_end) / 2

    upper_left = numpy.sum(upper_xs < x_midline)
    upper_right = numpy.sum(upper_xs >= x_midline)
    lower_left = numpy.sum(lower_xs < x_midline)
    lower_right = numpy.sum(lower_xs >= x_midline)

    use_left = (upper_left + lower_left) >= (upper_right + lower_right)

    upper_xs, upper_ys = _pick_side(upper_xs, upper_ys, x_midline, use_left)
    lower_xs, lower_ys = _pick_side(lower_xs, lower_ys, x_midline, use_left)

    if len(upper_xs) == 0 or len(lower_xs) == 0:
        return None

    upper_centroid = (float(numpy.mean(upper_xs)), float(numpy.mean(upper_ys)))
    lower_centroid = (float(numpy.mean(lower_xs)), float(numpy.mean(lower_ys)))

    return (upper_centroid, lower_centroid)


def sample_depth_at_point(
    depthmap, point_x, point_y, photo_width, photo_height, kernel_size=3
) -> int | None:
    """Sample depth map at a photo-space coordinate using median filtering.

    Translates from photo-space to depth-map-space and returns the median
    value of a kernel_size x kernel_size region.
    """
    depth_x = round(point_x * depthmap.width / photo_width)
    depth_y = round(point_y * depthmap.height / photo_height)

    half = kernel_size // 2
    values = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            sx = depth_x + dx
            sy = depth_y + dy
            if 0 <= sx < depthmap.width and 0 <= sy < depthmap.height:
                px = depthmap.getpixel((sx, sy))
                # Multi-channel depth maps (e.g. RGB): take first channel
                if isinstance(px, tuple):
                    px = px[0]
                values.append(px)

    if not values:
        return None

    values.sort()
    return values[len(values) // 2]
