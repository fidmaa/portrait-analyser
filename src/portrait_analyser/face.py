import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy
from PIL import Image

from .exceptions import MultipleFacesDetected, NoFacesDetected


@dataclass
class IncisorMeasurement:
    # Historical field names retained for API compatibility.  The points are
    # robust representatives of the facing incisal edges, not whole-tooth means.
    upper_centroid: tuple[float, float]  # (x, y) in photo/teethmap coordinates
    lower_centroid: tuple[float, float]  # (x, y) in photo/teethmap coordinates
    upper_depth_raw: int | None = None  # raw pixel value from depth map
    lower_depth_raw: int | None = None
    upper_distance_cm: float | None = None  # physical distance from camera (cm)
    lower_distance_cm: float | None = None
    distance_3d_mm: float | None = None  # 3D Euclidean distance between centroids
    pixel_distance_y: float = 0.0  # legacy-style vertical pixel gap
    # "upper"/"lower" when that arch was found only as a weak matte blob.
    weak_arch: str | None = None
    # "upper"/"lower" when that edge's depth was implausible and the other
    # arch's depth was used for it (coplanar incisal edges assumed).
    depth_assumed: str | None = None


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
                eyes.append(
                    Rectangle(
                        int(eye_x - eye_w / 2),
                        int(eye_y - eye_h / 2),
                        int(eye_w),
                        int(eye_h),
                    )
                )
    return eyes


def get_face_parameters(input_image: Image.Image, raise_opencv_exceptions=False):
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
            eyes.append(
                Eye(
                    None,  # face reference set below
                    int(eye_x_rel - eye_w / 2),
                    int(eye_y_rel - eye_h / 2),
                    int(eye_w),
                    int(eye_h),
                )
            )

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
    skinmap,
    scan_start_y,
    scan_end_y,
    threshold=1,
):
    """Find the first stable neck-width basin in an anatomical search band.

    Widths are median-smoothed vertically before looking for the first
    prominent local minimum. This prefers the neck basin immediately below the
    face over a later, globally narrower collar/shoulder matte artefact.

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

    ys = numpy.arange(scan_start_y, scan_end_y, dtype=numpy.int32)
    widths = numpy.full(len(ys), numpy.nan, dtype=numpy.float64)
    lefts = numpy.zeros(len(ys), dtype=numpy.int32)
    rights = numpy.zeros(len(ys), dtype=numpy.int32)

    for index, y in enumerate(ys):
        row = arr[y, :]
        skin_cols = numpy.where(row >= threshold)[0]
        if len(skin_cols) == 0:
            continue

        left = int(skin_cols[0])
        right = int(skin_cols[-1])
        width = right - left

        if width <= 0:
            continue
        widths[index] = width
        lefts[index] = left
        rights[index] = right

    valid = numpy.isfinite(widths)
    if not numpy.any(valid):
        return None

    span = len(ys)
    smooth_radius = max(2, min(15, round(span * 0.02)))
    smooth = numpy.full(span, numpy.nan, dtype=numpy.float64)
    for index in numpy.flatnonzero(valid):
        start = max(0, index - smooth_radius)
        stop = min(span, index + smooth_radius + 1)
        local = widths[start:stop]
        if numpy.count_nonzero(numpy.isfinite(local)) >= smooth_radius + 1:
            smooth[index] = float(numpy.nanmedian(local))

    smooth_valid = numpy.isfinite(smooth)
    if not numpy.any(smooth_valid):
        smooth = widths.copy()
        smooth_valid = valid.copy()

    comparison_radius = max(3, min(25, round(span * 0.05)))
    typical_width = float(numpy.nanmedian(smooth[smooth_valid]))
    minimum_prominence = max(2.0, typical_width * 0.015)
    local_minima = []
    for index in range(comparison_radius, span - comparison_radius):
        if not smooth_valid[index]:
            continue
        before = smooth[index - comparison_radius : index]
        after = smooth[index + 1 : index + comparison_radius + 1]
        if not numpy.any(numpy.isfinite(before)) or not numpy.any(
            numpy.isfinite(after)
        ):
            continue
        left_level = float(numpy.nanmedian(before))
        right_level = float(numpy.nanmedian(after))
        if (
            smooth[index] <= numpy.nanmin(before)
            and smooth[index] <= numpy.nanmin(after)
            and min(left_level, right_level) - smooth[index] >= minimum_prominence
        ):
            local_minima.append(index)

    if local_minima:
        selected = local_minima[0]
    else:
        # Monotonic/noisy profiles have no clear basin. Retain width as the
        # dominant signal but weakly penalise rows near the bottom of the band.
        normalized_width = smooth / max(typical_width, 1.0)
        depth_penalty = numpy.linspace(0.0, 0.12, span)
        score = normalized_width + depth_penalty
        score[~smooth_valid] = numpy.inf
        selected = int(numpy.argmin(score))

    best_y = int(ys[selected])
    return (int(lefts[selected]), best_y, int(rights[selected]), best_y)


def find_neck_measurement_point(
    skinmap,
    face_location=None,
    threshold=1,
    face=None,
    smooth_sigma=3.0,
    eyes=None,
    image_width=None,
    scan_start_y=None,
    scan_end_y=None,
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
            skinmap,
            scan_start_y,
            scan_end_y,
            threshold=threshold,
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


# Apple's semantic teeth matte is a 0-255 confidence map whose overall gain
# varies a lot between captures: on some photos the incisors saturate at 255,
# on others the whole matte peaks at ~100-140 even though teeth are clearly
# visible.  A fixed ``> 200`` cut therefore misses many real teeth.  The
# adaptive threshold below scales with the matte's own peak, is capped at the
# historical 200 (strong mattes behave exactly as before whenever their peak
# is high enough) and never drops below a floor that sits above the faint
# lip/mouth-contour halo Apple paints around an open mouth (~20-40) and above
# faint "teeth-like" blobs seen on mattes without visible teeth (max ~59).
TEETH_THRESHOLD_CAP = 200
TEETH_THRESHOLD_FLOOR = 64
TEETH_PEAK_FRACTION = 0.5
# The matte "peak" is the value of the K-th brightest pixel, K being this
# fraction of the image area (~1000 px on a 2320x3087 matte).  Using a rank
# statistic instead of ``max`` keeps a handful of saturated noise pixels from
# setting the threshold; using an area fraction instead of a percentile of
# non-zero pixels keeps it independent of how large the faint halo is.
TEETH_PEAK_AREA_FRACTION = 1.4e-4

# Bounding-box sanity limits, as fractions of the matte size (the historical
# fixed values were 100 px margins and a 200 px minimum height on ~2300x3100
# mattes, which also rejected photos where only one arch is in the matte).
TEETH_BBOX_MARGIN_FRACTION = 0.03
TEETH_BBOX_MIN_HEIGHT_FRACTION = 0.002
TEETH_BBOX_MIN_AREA_FRACTION = 1.5e-5
# Connected components smaller than this fraction of the largest one are
# treated as speckle and do not extend the bounding box.
TEETH_BBOX_COMPONENT_FRACTION = 0.1
# Inter-incisal measurement needs both arches inside the box.  A box shorter
# than this fraction of the matte height (the historical 200 px on ~3100 px
# mattes) holds a single arch, whose internal notches would otherwise be
# mistaken for the gap between upper and lower incisors.
TEETH_MEASUREMENT_MIN_HEIGHT_FRACTION = 0.065

# Weak opposite arch (see ``_find_weak_arch``).  On real mattes the weak arch
# peaks at ~20-35 on a background of exactly 0, while noise stays <= ~5.
TEETH_WEAK_ARCH_SEARCH_FRACTION = 0.2  # search reach, fraction of height
TEETH_WEAK_ARCH_MIN_GAP_FRACTION = 0.01  # min empty gap, fraction of height
TEETH_WEAK_ARCH_GAP_FILL = 0.02  # max share of weak pixels in a "gap" row
TEETH_WEAK_ARCH_PEAK_FRACTION = 0.5
TEETH_WEAK_ARCH_FLOOR = 8
TEETH_WEAK_ARCH_MIN_PEAK = 15
TEETH_WEAK_ARCH_MIN_WIDTH_FRACTION = 0.3  # of the searched central columns
TEETH_WEAK_ARCH_CENTRAL_FRACTION = 0.5  # searched share of strong-arch columns
TEETH_WEAK_ARCH_MIN_AREA_FRACTION = 1.5e-5  # of the image area (~107 px)


def _teeth_array(teethmap):
    """Return the teeth matte as a 2-D numpy array (first channel if RGB)."""
    arr = numpy.asarray(teethmap)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def teeth_threshold(
    teethmap,
    floor=None,
    cap=None,
    peak_fraction=None,
    peak_area_fraction=None,
) -> int:
    """Return a per-matte confidence threshold for the Apple teeth matte.

    Pixels ``>= teeth_threshold(teethmap)`` are treated as teeth.  The value is
    ``peak_fraction`` of the matte's robust peak (the K-th brightest pixel,
    ``K = peak_area_fraction * width * height``), clipped to ``[floor, cap]``.
    Arguments left as ``None`` take the ``TEETH_*`` module constants.
    """
    floor = TEETH_THRESHOLD_FLOOR if floor is None else floor
    cap = TEETH_THRESHOLD_CAP if cap is None else cap
    peak_fraction = TEETH_PEAK_FRACTION if peak_fraction is None else peak_fraction
    if peak_area_fraction is None:
        peak_area_fraction = TEETH_PEAK_AREA_FRACTION
    if not 0 <= floor <= cap:
        raise ValueError("floor must be non-negative and not above cap")
    if not 0 < peak_fraction <= 1:
        raise ValueError("peak_fraction must be in the (0, 1] range")
    if not 0 < peak_area_fraction < 1:
        raise ValueError("peak_area_fraction must be in the (0, 1) range")

    flat = _teeth_array(teethmap).ravel()
    if flat.size == 0:
        return int(cap)
    k = min(flat.size, max(1, round(flat.size * peak_area_fraction)))
    peak = float(numpy.partition(flat, flat.size - k)[flat.size - k])
    return round(min(cap, max(floor, peak * peak_fraction)))


def _resolve_threshold(teethmap, threshold):
    return teeth_threshold(teethmap) if threshold is None else threshold


@dataclass
class TeethArches:
    """Teeth found in the Apple teeth matte, possibly with one weak arch.

    ``mask`` is a boolean array (matte shape) of all teeth pixels: the strong
    arch(es) found with ``threshold`` plus, when ``weak_side`` is set, a weak
    opposite arch found with its own ``weak_threshold``.  ``bbox`` is
    ``(x, y, width, height)`` of ``mask``.
    """

    mask: numpy.ndarray
    bbox: tuple[int, int, int, int]
    threshold: int
    weak_threshold: int | None = None
    weak_side: str | None = None  # "upper", "lower" or None

    def mask_image(self) -> Image.Image:
        """``mask`` as an L-mode image (teeth 255, background 0)."""
        return Image.fromarray(self.mask.astype(numpy.uint8) * 255)


def _margin(value, size):
    if value is None:
        return round(size * TEETH_BBOX_MARGIN_FRACTION)
    return int(value)


def _mask_bbox(mask):
    rows = numpy.flatnonzero(mask.any(axis=1))
    cols = numpy.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def _significant_components(mask):
    """Drop speckle: keep components >= TEETH_BBOX_COMPONENT_FRACTION of the largest."""
    import cv2

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(numpy.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return numpy.zeros_like(mask, dtype=bool)
    keep = numpy.flatnonzero(areas >= TEETH_BBOX_COMPONENT_FRACTION * areas.max()) + 1
    return numpy.isin(labels, keep)


def _thresholded_teeth(arr, mask, margin_x, margin_y, min_height, min_area):
    """Apply margins, speckle removal and sanity limits to a teeth mask.

    Returns ``(kept_mask, (x0, y0, x1, y1))`` or ``None``.
    """
    height, width = arr.shape
    mx = _margin(margin_x, width)
    my = _margin(margin_y, height)
    if min_height is None:
        min_height = max(1, round(height * TEETH_BBOX_MIN_HEIGHT_FRACTION))
    if min_area is None:
        min_area = max(1, round(height * width * TEETH_BBOX_MIN_AREA_FRACTION))

    roi = numpy.zeros_like(mask, dtype=bool)
    roi[my : height - my, mx : width - mx] = True
    mask = mask & roi
    if not mask.any():
        return None

    kept = _significant_components(mask)
    if int(numpy.count_nonzero(kept)) < min_area:
        return None

    x0, y0, x1, y1 = _mask_bbox(kept)
    if y1 == height - my - 1:
        # Teeth reach the bottom margin: the matte is cut off, reject.
        return None
    if y1 - y0 < min_height:
        return None
    return kept, (x0, y0, x1, y1)


def _find_weak_arch(arr, strong, strong_box, side):
    """Look for a weak arch across a near-empty gap from the strong arch.

    Apple's matte frequently gives one arch full confidence and the opposite
    one only ~10-35.  That weak arch is still a separate blob on a zero
    background, split from the strong arch by the dark mouth cavity.  The
    search runs in the central columns of the strong arch (so the faint lip
    contour at the mouth corners cannot bridge the two arches), on ``side``
    ("upper"/"lower") of it, with a threshold relative to that side's own
    robust peak.  Weak-level blobs connected to the strong arch are its soft
    skirt and are ignored; what remains must be sizeable, wide enough and
    separated from the strong arch by a real gap.

    Returns ``(weak_mask, weak_threshold)`` (full-size mask) or ``None``.
    """
    import cv2

    height, width = arr.shape
    x0, y0, x1, y1 = strong_box
    strong_width = x1 - x0 + 1
    c0 = x0 + round(strong_width * (1 - TEETH_WEAK_ARCH_CENTRAL_FRACTION) / 2)
    c1 = max(c0 + 1, x1 + 1 - round(strong_width * (1 - TEETH_WEAK_ARCH_CENTRAL_FRACTION) / 2))
    reach = round(height * TEETH_WEAK_ARCH_SEARCH_FRACTION)
    min_gap = max(2, round(height * TEETH_WEAK_ARCH_MIN_GAP_FRACTION))
    my = _margin(None, height)
    if side == "upper":
        side_top, side_bottom = max(my, y0 - reach), y0  # rows [top, bottom)
    else:
        side_top, side_bottom = y1 + 1, min(height - my, y1 + 1 + reach)
    if side_bottom - side_top <= min_gap:
        return None

    side_values = arr[side_top:side_bottom, c0:c1]
    values = side_values[side_values > 0]
    if values.size == 0:
        return None
    peak = float(numpy.percentile(values, 99))
    if peak < TEETH_WEAK_ARCH_MIN_PEAK:
        return None
    weak_threshold = max(TEETH_WEAK_ARCH_FLOOR, round(peak * TEETH_WEAK_ARCH_PEAK_FRACTION))

    # Label weak-level pixels over the side window *and* the strong arch rows,
    # so blobs touching the strong arch can be recognised and dropped.
    top = min(side_top, y0)
    bottom = max(side_bottom, y1 + 1)
    window = arr[top:bottom, c0:c1]
    strong_window = strong[top:bottom, c0:c1]
    level = (window >= weak_threshold) | strong_window
    _, labels = cv2.connectedComponents(level.astype(numpy.uint8), connectivity=8)
    skirt_labels = numpy.unique(labels[strong_window])
    candidate = level & ~numpy.isin(labels, skirt_labels)
    if side == "upper":
        candidate[y0 - top :] = False
    else:
        candidate[: y1 + 1 - top] = False
    if not candidate.any():
        return None
    weak = _significant_components(candidate)

    min_area = max(1, round(height * width * TEETH_WEAK_ARCH_MIN_AREA_FRACTION))
    if int(numpy.count_nonzero(weak)) < min_area:
        return None
    wx0, wy0, wx1, wy1 = _mask_bbox(weak)
    if wx1 - wx0 + 1 < TEETH_WEAK_ARCH_MIN_WIDTH_FRACTION * (c1 - c0):
        return None

    # Real gap: a run of rows with ~no weak-level signal between the weak
    # arch's facing edge and the strong arch's skirt.
    if side == "upper":
        span = level[wy1 + 1 : y0 - top]
    else:
        span = level[y1 + 1 - top : wy0]
    empty_rows = numpy.count_nonzero(span, axis=1) <= TEETH_WEAK_ARCH_GAP_FILL * (c1 - c0)
    longest = max((end - start for start, end in _true_runs(empty_rows)), default=0)
    if longest < min_gap:
        return None

    full = numpy.zeros_like(strong, dtype=bool)
    full[top:bottom, c0:c1] = weak
    return full, int(weak_threshold)


def detect_teeth_arches(
    teethmap,
    threshold=None,
    margin_x=None,
    margin_y=None,
    min_height=None,
    min_area=None,
    find_weak_arch=True,
) -> TeethArches | None:
    """Detect teeth in the Apple teeth matte, per arch.

    Strong teeth are pixels ``>= threshold`` (default :func:`teeth_threshold`).
    When they form a single arch (box shorter than
    ``TEETH_MEASUREMENT_MIN_HEIGHT_FRACTION`` of the height) and
    ``find_weak_arch`` is true, the opposite arch is searched for across the
    mouth gap with its own, lower threshold -- upper side first, as a weak
    upper arch is the common case.  A weak blob on its own never produces a
    detection: a strong arch must be present first.
    """
    arr = _teeth_array(teethmap)
    if threshold is None:
        threshold = teeth_threshold(arr)
    found = _thresholded_teeth(arr, arr >= threshold, margin_x, margin_y, min_height, min_area)
    if found is None:
        return None
    mask, (x0, y0, x1, y1) = found
    weak_threshold = None
    weak_side = None

    height = arr.shape[0]
    if find_weak_arch and y1 - y0 < round(height * TEETH_MEASUREMENT_MIN_HEIGHT_FRACTION):
        for side in ("upper", "lower"):
            weak = _find_weak_arch(arr, mask, (x0, y0, x1, y1), side)
            if weak is not None:
                weak_mask, weak_threshold = weak
                mask = mask | weak_mask
                weak_side = side
                x0, y0, x1, y1 = _mask_bbox(mask)
                break

    return TeethArches(
        mask=mask,
        bbox=(x0, y0, x1 - x0, y1 - y0),
        threshold=int(threshold),
        weak_threshold=weak_threshold,
        weak_side=weak_side,
    )


def find_bounding_box_teeth(
    teethmap,
    margin_x=None,
    margin_y=None,
    min_value=None,
    min_height=None,
    min_area=None,
):
    """Return the teeth bounding box ``(x, y, width, height)`` or ``None``.

    ``min_value=None`` (default) uses :func:`detect_teeth_arches`: adaptive
    threshold (pixels ``>= threshold``) plus a weak opposite arch when one is
    present.  An explicit ``min_value`` keeps the historical single strict
    ``> min_value`` comparison and no weak-arch search.  Margins default to
    ``TEETH_BBOX_MARGIN_FRACTION`` of the matte size.  Only connected
    components at least ``TEETH_BBOX_COMPONENT_FRACTION`` of the largest one
    contribute to the box, so isolated speckles cannot inflate it.  ``None``
    is returned when nothing is found, the teeth touch the bottom margin, the
    box is shorter than ``min_height`` or the teeth cover fewer than
    ``min_area`` pixels (both default to fractions of the matte size).
    """
    if min_value is None:
        arches = detect_teeth_arches(
            teethmap,
            margin_x=margin_x,
            margin_y=margin_y,
            min_height=min_height,
            min_area=min_area,
        )
        return None if arches is None else arches.bbox

    arr = _teeth_array(teethmap)
    found = _thresholded_teeth(arr, arr > min_value, margin_x, margin_y, min_height, min_area)
    if found is None:
        return None
    _, (x0, y0, x1, y1) = found
    return (x0, y0, x1 - x0, y1 - y0)


def find_incisor_distance_teeth(
    teethmap, bounding_box_teeth, threshold=None, margin_x=0.5
):
    """Find incisor distance from teeth map.

    Iterate from teethmap x1 to x1 + width, starting from
    the half of it, try finding points with MAXIMAL distance
    as long as they are within bounding_box_teeth and their
    value is at least ``threshold`` (well-detected teeth, to avoid
    diasthemes which would probably be the highest distance
    points, but that's not what we're looking for...).
    ``threshold=None`` uses the adaptive :func:`teeth_threshold`.
    """
    threshold = _resolve_threshold(teethmap, threshold)

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


def _true_runs(values):
    """Yield half-open runs where a one-dimensional boolean array is true."""
    run_start = None
    for index, value in enumerate(values):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            yield run_start, index
            run_start = None
    if run_start is not None:
        yield run_start, len(values)


def _find_incisor_gap(
    mask,
    min_pixels,
    min_gap_height,
    max_gap_foreground_fraction,
):
    """Find the widest low-confidence horizontal band between tooth regions.

    ``mask`` is the central part of the teeth matte.  A valid gap must have
    enough foreground support both above and below it; this prevents one
    continuous bright region from being split at the bounding-box midpoint.
    """
    if mask.size == 0 or mask.shape[1] == 0:
        return None

    row_counts = numpy.count_nonzero(mask, axis=1)
    if not numpy.any(row_counts):
        return None

    # Scale the allowed gap noise to the strongest tooth row, not the full ROI
    # width.  Otherwise a valid but narrower lower incisor could itself be
    # classified as gap merely because the upper incisor is much wider.
    max_gap_foreground = int(numpy.max(row_counts) * max_gap_foreground_fraction)
    low_foreground_rows = row_counts <= max_gap_foreground
    tooth_rows = numpy.flatnonzero(~low_foreground_rows)
    if len(tooth_rows) < 2:
        return None

    first_tooth_row = int(tooth_rows[0])
    last_tooth_row = int(tooth_rows[-1])
    candidates = []
    for start, end in _true_runs(low_foreground_rows):
        if start <= first_tooth_row or end - 1 >= last_tooth_row:
            continue
        if end - start < min_gap_height:
            continue

        upper_support = int(numpy.count_nonzero(mask[:start]))
        lower_support = int(numpy.count_nonzero(mask[end:]))
        if upper_support < min_pixels or lower_support < min_pixels:
            continue

        # Prefer the widest genuine gap.  When two gaps have the same width,
        # the one nearest the vertical centre is the more plausible mouth gap.
        centre_offset = abs((start + end) / 2.0 - mask.shape[0] / 2.0)
        candidates.append((end - start, -centre_offset, start, end))

    if not candidates:
        return None

    _, _, start, end = max(candidates)
    return start, end


def _robust_edge_pair(xs, upper_ys, lower_ys, min_edge_columns):
    """Return a real paired edge point nearest the robust boundary centroid."""
    xs = numpy.asarray(xs, dtype=float)
    upper_ys = numpy.asarray(upper_ys, dtype=float)
    lower_ys = numpy.asarray(lower_ys, dtype=float)
    if len(xs) < min_edge_columns:
        return None

    keep = numpy.ones(len(xs), dtype=bool)
    for values in (upper_ys, lower_ys, lower_ys - upper_ys):
        median = float(numpy.median(values))
        mad = float(numpy.median(numpy.abs(values - median)))
        tolerance = max(2.0, 3.0 * 1.4826 * mad)
        keep &= numpy.abs(values - median) <= tolerance

    if numpy.count_nonzero(keep) < min_edge_columns:
        return None

    xs = xs[keep]
    upper_ys = upper_ys[keep]
    lower_ys = lower_ys[keep]

    # The mathematical centroid of a curved/disconnected boundary may lie in
    # the dark mouth cavity.  Snap it to the nearest *paired* boundary sample,
    # so both returned points are actual teeth pixels in the same column.
    target_x = float(numpy.mean(xs))
    target_upper_y = float(numpy.mean(upper_ys))
    target_lower_y = float(numpy.mean(lower_ys))
    costs = (
        (xs - target_x) ** 2
        + (upper_ys - target_upper_y) ** 2
        + (lower_ys - target_lower_y) ** 2
    )
    best = int(numpy.argmin(costs))
    x = float(xs[best])
    points = (x, float(upper_ys[best])), (x, float(lower_ys[best]))
    return int(len(xs)), points


def _edge_pair_for_side(
    mask,
    x_offset,
    side_start,
    side_end,
    gap_start,
    gap_end,
    min_pixels,
    min_edge_columns,
):
    """Build a representative upper/lower incisal-edge pair for one side."""
    side = mask[:, side_start:side_end]
    if side.size == 0:
        return None

    upper = side[:gap_start]
    lower = side[gap_end:]
    upper_support = int(numpy.count_nonzero(upper))
    lower_support = int(numpy.count_nonzero(lower))
    if upper_support < min_pixels or lower_support < min_pixels:
        return None

    xs = []
    upper_ys = []
    lower_ys = []
    for local_x in range(side.shape[1]):
        upper_rows = numpy.flatnonzero(upper[:, local_x])
        lower_rows = numpy.flatnonzero(lower[:, local_x])
        if len(upper_rows) == 0 or len(lower_rows) == 0:
            continue
        xs.append(x_offset + side_start + local_x)
        upper_ys.append(int(upper_rows[-1]))
        lower_ys.append(gap_end + int(lower_rows[0]))

    robust_result = _robust_edge_pair(xs, upper_ys, lower_ys, min_edge_columns)
    if robust_result is None:
        return None
    robust_column_count, points = robust_result

    # Rank sides by paired-column coverage first and balanced tooth support
    # second.  A side with one noisy lower pixel must not beat a well-supported
    # upper/lower pair merely because its upper tooth region is large.
    score = (
        robust_column_count,
        min(upper_support, lower_support),
        upper_support + lower_support,
    )
    return score, points


def find_incisor_centroids(
    teethmap,
    bounding_box_teeth,
    threshold=None,
    margin_x=0.5,
    min_pixels=50,
    centroid_margin_x=0.5,
    min_gap_fraction=0.01,
    max_gap_foreground_fraction=0.1,
    min_edge_columns=5,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Find representative points on facing upper/lower incisal edges.

    The public name and return shape retain the historical ``centroid`` API,
    but the points are now derived from the facing edges rather than the full
    visible tooth surfaces.  Boundary samples are robustly centred and then
    snapped to a real paired mask column.  This measures the inter-incisal gap
    and guarantees that depth is sampled on tooth pixels, not in the cavity.

    Returns ``((upper_x, upper_y), (lower_x, lower_y))`` in teethmap
    coordinates, or ``None`` when two separated, sufficiently supported
    incisal surfaces cannot be identified.  ``threshold=None`` uses the
    adaptive :func:`teeth_threshold`.
    """
    if not 0 < margin_x <= 1 or not 0 < centroid_margin_x <= 1:
        raise ValueError("margin fractions must be in the (0, 1] range")
    if min_pixels < 1 or min_edge_columns < 1:
        raise ValueError("minimum support values must be positive")
    if not 0 <= max_gap_foreground_fraction < 1:
        raise ValueError("max_gap_foreground_fraction must be in the [0, 1) range")
    if min_gap_fraction < 0:
        raise ValueError("min_gap_fraction must not be negative")

    bb_x, bb_y, bb_w, bb_h = (int(value) for value in bounding_box_teeth)
    image_width, image_height = teethmap.size
    bb_end_x = min(image_width, bb_x + bb_w)
    bb_end_y = min(image_height, bb_y + bb_h)
    bb_x = max(0, bb_x)
    bb_y = max(0, bb_y)
    bb_w = bb_end_x - bb_x
    bb_h = bb_end_y - bb_y
    if bb_w < 2 or bb_h < 2:
        return None

    arr = _teeth_array(teethmap)
    threshold = _resolve_threshold(arr, threshold)

    # Locate a real low-confidence band between upper and lower teeth using a
    # central strip.  This replaces the old unconditional split at bbox/2.
    gap_x_start = int(bb_x + bb_w / 2 - margin_x * bb_w / 2)
    gap_x_end = int(bb_x + bb_w / 2 + margin_x * bb_w / 2)
    gap_mask = arr[bb_y:bb_end_y, gap_x_start:gap_x_end] >= threshold
    min_gap_height = max(2, int(round(bb_h * min_gap_fraction)))
    gap = _find_incisor_gap(
        gap_mask,
        min_pixels=min_pixels,
        min_gap_height=min_gap_height,
        max_gap_foreground_fraction=max_gap_foreground_fraction,
    )
    if gap is None:
        return None
    gap_start, gap_end = gap

    # Restrict edge measurement to the central incisors.  Lateral teeth form
    # an arch and would otherwise pull the representative points sideways.
    cx_start = int(bb_x + bb_w / 2 - centroid_margin_x * bb_w / 2)
    cx_end = int(bb_x + bb_w / 2 + centroid_margin_x * bb_w / 2)
    mask = arr[bb_y:bb_end_y, cx_start:cx_end] >= threshold
    if mask.size == 0:
        return None

    side_midpoint = mask.shape[1] // 2
    candidates = []
    for side_start, side_end in ((0, side_midpoint), (side_midpoint, mask.shape[1])):
        candidate = _edge_pair_for_side(
            mask,
            x_offset=cx_start,
            side_start=side_start,
            side_end=side_end,
            gap_start=gap_start,
            gap_end=gap_end,
            min_pixels=min_pixels,
            min_edge_columns=min_edge_columns,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return None

    _, (upper_point, lower_point) = max(candidates, key=lambda item: item[0])
    return (
        (upper_point[0], upper_point[1] + bb_y),
        (lower_point[0], lower_point[1] + bb_y),
    )


def sample_depth_at_point(
    depthmap,
    point_x,
    point_y,
    photo_width,
    photo_height,
    kernel_size=3,
    support_mask=None,
    support_threshold=None,
    inward_y=0,
) -> int | None:
    """Sample depth map at a photo-space coordinate using median filtering.

    Translates from photo-space to depth-map-space and returns the median value
    of a ``kernel_size`` square.  When ``support_mask`` is supplied, only depth
    pixels whose centres map to foreground mask pixels are included.  An
    ``inward_y`` offset in depth pixels can move an incisal-edge sample into the
    tooth surface (negative for an upper tooth, positive for a lower tooth).
    ``support_threshold=None`` uses :func:`teeth_threshold` of the mask.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    if photo_width < 1 or photo_height < 1 or depthmap.width < 1 or depthmap.height < 1:
        return None
    if not 0 <= point_x <= photo_width - 1 or not 0 <= point_y <= photo_height - 1:
        return None

    # Match image endpoints exactly.  Multiplying by depthmap.width/photo_width
    # maps the last valid photo pixel beyond the last depth pixel after rounding.
    depth_x = (
        0
        if photo_width == 1
        else round(point_x * (depthmap.width - 1) / (photo_width - 1))
    )
    depth_y = (
        0
        if photo_height == 1
        else round(point_y * (depthmap.height - 1) / (photo_height - 1))
    )
    depth_y += int(inward_y)

    if support_mask is not None and support_threshold is None:
        support_threshold = teeth_threshold(support_mask)

    half = kernel_size // 2
    values = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            sx = depth_x + dx
            sy = depth_y + dy
            if 0 <= sx < depthmap.width and 0 <= sy < depthmap.height:
                if support_mask is not None:
                    mask_x = (
                        0
                        if depthmap.width == 1
                        else round(sx * (support_mask.width - 1) / (depthmap.width - 1))
                    )
                    mask_y = (
                        0
                        if depthmap.height == 1
                        else round(sy * (support_mask.height - 1) / (depthmap.height - 1))
                    )
                    mask_value = support_mask.getpixel((mask_x, mask_y))
                    if isinstance(mask_value, tuple):
                        mask_value = mask_value[0]
                    if mask_value < support_threshold:
                        continue

                px = depthmap.getpixel((sx, sy))
                # Multi-channel depth maps (e.g. RGB): take first channel
                if isinstance(px, tuple):
                    px = px[0]
                values.append(px)

    if not values:
        return None

    values.sort()
    return values[len(values) // 2]
