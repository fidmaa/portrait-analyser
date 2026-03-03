"""Neck/chin detection via MediaPipe Selfie Segmentation.

Fallback method for when FaceLandmarker/PoseLandmarker fail (e.g., extended
neck poses). Derives neck midpoint and chin from the person silhouette
width profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .pose import NeckMidpoint, PortraitPose, _download_model

if TYPE_CHECKING:
    from PIL import Image

_SELFIE_SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/latest/"
    "selfie_segmenter.tflite"
)
_SELFIE_SEGMENTER_FILENAME = "selfie_segmenter.tflite"


@dataclass(frozen=True)
class SegmentationDebug:
    """Debug data from segmentation-based neck detection."""

    mask: np.ndarray  # binary mask (H, W), dtype bool
    width_profile: np.ndarray  # smoothed width at each row in ROI
    roi_top: int  # top y of the ROI used for width profile
    neck_y: int | None  # detected neck row in image coords
    chin_y: int | None  # detected chin row in image coords
    midline_x: float | None  # midline x at neck_y
    shoulder_y: int | None = None  # detected shoulder row in image coords
    skin_mask: np.ndarray | None = None  # binary skin mask for debug viz
    ear_y: int | None = None  # detected ear/jaw level (widest skin row)
    neck_left_x: float | None = None  # left edge x at neck_y
    neck_right_x: float | None = None  # right edge x at neck_y
    neck_width_front_arc_mm: float | None = None  # 3D front arc width
    neck_width_straight_mm: float | None = None  # 3D straight-line width


def _clean_mask(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Clean segmentation mask with median blur and morphological close.

    Uses OpenCV (available as a transitive dependency of mediapipe).
    """
    import cv2

    mask_u8 = (mask * 255).astype(np.uint8)

    # Median blur to remove salt-and-pepper noise
    mask_u8 = cv2.medianBlur(mask_u8, kernel_size)

    # Morphological close to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    return mask_u8 > 127


def _compute_width_profile(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the width, left edge, and right edge for each row.

    Returns:
        widths: array of widths per row (0 if no pixels in that row)
        lefts: array of leftmost x per row
        rights: array of rightmost x per row
    """
    h = mask.shape[0]
    widths = np.zeros(h, dtype=np.float64)
    lefts = np.zeros(h, dtype=np.float64)
    rights = np.zeros(h, dtype=np.float64)

    for y in range(h):
        row_pixels = np.where(mask[y])[0]
        if len(row_pixels) > 0:
            lefts[y] = row_pixels[0]
            rights[y] = row_pixels[-1]
            widths[y] = rights[y] - lefts[y]

    return widths, lefts, rights


def _make_empty_debug(mask, roi_top=0, skin_mask=None):
    return SegmentationDebug(
        mask=mask,
        width_profile=np.array([]),
        roi_top=roi_top,
        neck_y=None,
        chin_y=None,
        midline_x=None,
        shoulder_y=None,
        skin_mask=skin_mask,
    )


def detect_neck_midpoint_from_segmentation(
    image: Image.Image,
    threshold: float = 0.5,
    jaw_flare_fraction: float = 0.15,
    smoothing_window: int = 15,
) -> tuple[NeckMidpoint | None, SegmentationDebug | None]:
    """Detect neck midpoint and chin from the person silhouette.

    Uses MediaPipe SelfieSegmentation to create a person mask, then
    analyzes the width profile to find the narrowest point (neck) and
    where the jaw flares out above it (chin).

    Args:
        image: PIL Image of the portrait.
        threshold: Segmentation confidence threshold (0-1).
        jaw_flare_fraction: Minimum width increase fraction to detect jaw flare.
        smoothing_window: Window size for smoothing the width profile.

    Returns:
        2-tuple of (NeckMidpoint | None, SegmentationDebug | None).
    """
    import mediapipe as mp

    model_path = _download_model(_SELFIE_SEGMENTER_URL, _SELFIE_SEGMENTER_FILENAME)

    image_array = np.array(image)
    h, w = image_array.shape[:2]

    # Ensure RGB (3-channel) input
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

    # Run selfie segmentation using the Tasks API (ImageSegmenter)
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        output_confidence_masks=True,
        output_category_mask=False,
    )
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp_image)

    if not result.confidence_masks:
        return None, None

    # confidence_masks[0] is the person mask; shape (H, W, 1), float32
    raw_mask = result.confidence_masks[0].numpy_view()
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[:, :, 0]
    binary_mask = _clean_mask((raw_mask > threshold).astype(np.float64))

    return _detect_from_mask(binary_mask, h, w, jaw_flare_fraction, smoothing_window)


def _get_segmentation_mask(
    image: Image.Image, threshold: float = 0.5
) -> np.ndarray | None:
    """Run MediaPipe selfie segmentation and return cleaned binary mask."""
    import mediapipe as mp

    model_path = _download_model(_SELFIE_SEGMENTER_URL, _SELFIE_SEGMENTER_FILENAME)

    image_array = np.array(image)

    # Ensure RGB (3-channel) input
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        output_confidence_masks=True,
        output_category_mask=False,
    )
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp_image)

    if not result.confidence_masks:
        return None

    raw_mask = result.confidence_masks[0].numpy_view()
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[:, :, 0]
    return _clean_mask((raw_mask > threshold).astype(np.float64))


def _find_chin_from_depth(
    skin_binary: np.ndarray,
    depthmap_arr: np.ndarray,
) -> tuple[int | None, float | None]:
    """Find chin as the point closest to camera in the upper face.

    The chin is the pixel with the **maximum depth value** (highest = closest
    to camera) within the skin area in the upper 2/3 of the image, restricted
    to the central 30% of width.

    Args:
        skin_binary: (H, W) boolean mask of skin pixels.
        depthmap_arr: (H_d, W_d) uint8 depth map — high values = close.

    Returns:
        (chin_y, midline_x) or (None, None) if no valid pixel found.
    """
    import cv2

    h, w = skin_binary.shape[:2]

    # Resize depth map to match skin mask dimensions
    if depthmap_arr.shape[:2] != (h, w):
        depthmap_resized = cv2.resize(
            depthmap_arr, (w, h), interpolation=cv2.INTER_LINEAR
        )
    else:
        depthmap_resized = depthmap_arr

    # Restrict to upper 2/3 of rows
    max_row = int(h * 2 / 3)

    # Restrict to middle 30% of width (35%–65%)
    left_col = int(w * 0.35)
    right_col = int(w * 0.65)

    # Build a combined mask: skin AND within ROI
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[:max_row, left_col:right_col] = True
    combined = skin_binary & roi_mask

    if not np.any(combined):
        return None, None

    # Zero out depth where there's no skin in the ROI
    masked_depth = depthmap_resized.astype(np.float64)
    masked_depth[~combined] = -1

    # Find pixel with maximum depth value (closest to camera)
    flat_idx = int(np.argmax(masked_depth))
    chin_y, chin_x = divmod(flat_idx, w)

    return chin_y, float(chin_x)


def _make_depth_body_mask(
    depthmap_arr: np.ndarray,
    skin_binary: np.ndarray,
    hair_arr: np.ndarray | None = None,
    threshold_fraction: float = 0.15,
    hair_threshold: int = 100,
) -> np.ndarray:
    """Create a body-presence mask from the depth map with hair removed.

    High depth values = close to camera = body.  Thresholding at a fraction
    of the max depth filters out the background while keeping the body.
    Hair pixels are zeroed out from the depth map before thresholding so
    that hanging hair does not inflate the width profile.

    Args:
        depthmap_arr: (H_d, W_d) uint8 depth map — high values = close.
        skin_binary: (H, W) boolean mask used only for target dimensions.
        hair_arr: (H_h, W_h) uint8 hair matte or None. High = hair.
        threshold_fraction: Fraction of max depth value to use as threshold.
        hair_threshold: Minimum hair matte value to consider as hair.

    Returns:
        Boolean mask (H, W) where True = body pixel (excluding hair).
    """
    import cv2

    h, w = skin_binary.shape[:2]

    # Resize depth map to match skin mask dimensions
    if depthmap_arr.shape[:2] != (h, w):
        depthmap_resized = cv2.resize(
            depthmap_arr, (w, h), interpolation=cv2.INTER_LINEAR
        ).copy()
    else:
        depthmap_resized = depthmap_arr.copy()

    # Zero out hair pixels in the depth map
    if hair_arr is not None:
        if hair_arr.shape[:2] != (h, w):
            hair_resized = cv2.resize(
                hair_arr, (w, h), interpolation=cv2.INTER_LINEAR
            )
        else:
            hair_resized = hair_arr
        depthmap_resized[hair_resized >= hair_threshold] = 0

    max_depth = float(np.max(depthmap_resized))
    if max_depth <= 0:
        return np.zeros((h, w), dtype=bool)

    threshold = max_depth * threshold_fraction
    return depthmap_resized > threshold


def _find_shoulders_from_segmentation(
    seg_mask: np.ndarray,
    below_y: int,
    width_increase_factor: float = 1.25,
    smoothing_window: int = 15,
) -> int | None:
    """Find shoulder_y from the segmentation mask below a given row.

    Looks for where the silhouette width dramatically increases
    (shoulders appearing).

    Returns:
        shoulder_y in image coordinates, or None.
    """
    h = seg_mask.shape[0]
    if below_y >= h - 10:
        return None

    widths, _, _ = _compute_width_profile(seg_mask)

    # Smooth
    if smoothing_window > 1 and len(widths) > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        widths_smooth = np.convolve(widths, kernel, mode="same")
    else:
        widths_smooth = widths.copy()

    # Compute the typical "neck width" as the minimum width in a small band
    # just below chin_y
    neck_search_start = below_y
    neck_search_end = min(below_y + int(h * 0.15), h)
    neck_band = widths_smooth[neck_search_start:neck_search_end]

    nonzero_band = neck_band[neck_band > 0]
    if len(nonzero_band) == 0:
        return None

    neck_width = float(np.min(nonzero_band))
    shoulder_threshold = neck_width * width_increase_factor

    # Scan downward from chin looking for dramatic widening
    for y in range(below_y, min(h, below_y + int(h * 0.4))):
        if widths_smooth[y] > shoulder_threshold:
            return y

    return None


def compute_neck_width_3d(
    depthmap: "Image.Image",
    neck_y: int,
    neck_left_x: float,
    neck_right_x: float,
    photo_width: int,
    photo_height: int,
    float_min: float,
    float_max: float,
    n_samples: int = 25,
) -> tuple[float | None, float | None]:
    """Compute 3D neck width by sampling points across the neck row.

    Samples N evenly-spaced points at neck_y from left to right edge,
    converts each to 3D coordinates using depth calibration, and:
    - Sums consecutive 3D distances → front arc length
    - Computes straight-line 3D distance from first to last point

    Args:
        depthmap: PIL Image depth map.
        neck_y: Row index of the neck (narrowest point).
        neck_left_x: Left edge x-coordinate at neck_y.
        neck_right_x: Right edge x-coordinate at neck_y.
        photo_width: Width of the photo image.
        photo_height: Height of the photo image.
        float_min: EXIF FloatMinValue from depth metadata.
        float_max: EXIF FloatMaxValue from depth metadata.
        n_samples: Number of sample points across the neck.

    Returns:
        (front_arc_mm, straight_width_mm) or (None, None).
    """
    from .face import sample_depth_at_point
    from .incisor import depth_raw_to_distance_cm, pixel_to_mm, vector_length_3d

    # Inset edges by 5% to avoid unreliable edge depths
    span = neck_right_x - neck_left_x
    inset = span * 0.05
    left = neck_left_x + inset
    right = neck_right_x - inset

    # Generate evenly-spaced x-coordinates
    xs = [left + (right - left) * i / (n_samples - 1) for i in range(n_samples)]

    # Convert each sample point to 3D (mm)
    points_3d = []
    for x in xs:
        depth_raw = sample_depth_at_point(
            depthmap, x, neck_y, photo_width, photo_height
        )
        if depth_raw is None or depth_raw == 0:
            continue

        z_cm = depth_raw_to_distance_cm(depth_raw, float_min, float_max)
        if z_cm is None:
            continue

        # Convert pixel coords to mm at this depth
        x_mm = pixel_to_mm(x, z_cm)
        y_mm = pixel_to_mm(float(neck_y), z_cm)
        if x_mm is None or y_mm is None:
            continue

        z_mm = z_cm * 10.0
        points_3d.append((x_mm, y_mm, z_mm))

    if len(points_3d) < 2:
        return None, None

    # Sum consecutive 3D distances → front arc
    front_arc_mm = sum(
        vector_length_3d(
            points_3d[i][0], points_3d[i][1], points_3d[i][2],
            points_3d[i + 1][0], points_3d[i + 1][1], points_3d[i + 1][2],
        )
        for i in range(len(points_3d) - 1)
    )

    # Straight-line 3D distance from first to last point
    straight_width_mm = vector_length_3d(
        points_3d[0][0], points_3d[0][1], points_3d[0][2],
        points_3d[-1][0], points_3d[-1][1], points_3d[-1][2],
    )

    return front_arc_mm, straight_width_mm


def _detect_landmarks_from_depth_profile(
    depthmap_arr: np.ndarray,
    skin_binary: np.ndarray,
    chin_y: int,
    hair_arr: np.ndarray | None = None,
    smoothing_window: int = 15,
) -> tuple[int | None, int | None, int | None, float | None, float | None]:
    """Detect ear, neck, and shoulder levels from depth map below chin.

    Algorithm — finds two peaks in the width profile below chin:
    1. Build depth body mask (with hair removed), compute smoothed width profile
    2. Peak 1 — EARS: widest row below chin (jaw/ear level)
    3. Valley: narrowest point after ear peak
    4. Peak 2 — SHOULDERS: widest row after the valley
    5. neck_y = midpoint between the two peaks

    Returns:
        (ear_y, neck_y, shoulder_y, neck_left_x, neck_right_x) — any may be None.
    """
    depth_mask = _make_depth_body_mask(depthmap_arr, skin_binary, hair_arr=hair_arr)
    h = depth_mask.shape[0]

    if chin_y >= h - 20:
        return None, None, None, None, None

    widths, lefts, rights = _compute_width_profile(depth_mask)

    # Smooth the entire profile
    if smoothing_window > 1 and len(widths) > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        widths_smooth = np.convolve(widths, kernel, mode="same")
    else:
        widths_smooth = widths.copy()

    # --- Divide area below chin into 3 equal parts ---
    # Upper third: ears/jaw (widest point)
    # Middle third: neck region
    # Lower third: shoulders (widest point)
    below_chin = widths_smooth[chin_y:]
    n = len(below_chin)
    if n < 3 or not np.any(below_chin > 0):
        return None, None, None, None, None

    third = n // 3

    # EARS — widest row in the upper third
    upper = below_chin[:third]
    ear_y = None
    if np.any(upper > 0):
        ear_y = chin_y + int(np.argmax(upper))

    # SHOULDERS — widest row in the lower third
    lower = below_chin[2 * third :]
    shoulder_y = None
    if np.any(lower > 0):
        shoulder_y = chin_y + 2 * third + int(np.argmax(lower))

    # neck_y = midpoint between ears and shoulders
    if ear_y is not None and shoulder_y is not None:
        neck_y = (ear_y + shoulder_y) // 2
    elif ear_y is not None:
        # No shoulders — use narrowest nonzero row below ears as fallback
        after_ear = below_chin[ear_y - chin_y :]
        nz = after_ear > 0
        if np.any(nz):
            fallback = after_ear.copy()
            fallback[~nz] = np.inf
            neck_y = (ear_y - chin_y) + int(np.argmin(fallback)) + chin_y
        else:
            neck_y = ear_y + third
    else:
        return None, None, None, None, None

    # Extract neck edge x-coordinates at neck_y
    neck_left_x = float(lefts[neck_y]) if lefts[neck_y] > 0 else None
    neck_right_x = float(rights[neck_y]) if rights[neck_y] > 0 else None

    return ear_y, neck_y, shoulder_y, neck_left_x, neck_right_x


def _find_neck_from_skin(
    skin_binary: np.ndarray,
    search_top: int,
    search_bottom: int,
    smoothing_window: int = 11,
) -> int | None:
    """Find the neck row as the narrowest skin row in a given range.

    Args:
        skin_binary: (H, W) boolean mask of skin pixels.
        search_top: Start row for the search (e.g. ear_y or chin_y).
        search_bottom: End row for the search (e.g. shoulder_y).
        smoothing_window: Window size for smoothing the width profile.

    Returns:
        Row index of the narrowest skin point, or None if no skin found.
    """
    widths, _, _ = _compute_width_profile(skin_binary)

    # Extract the region
    region = widths[search_top:search_bottom]
    if len(region) == 0:
        return None

    # Smooth the width profile
    if smoothing_window > 1 and len(region) > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        region = np.convolve(region, kernel, mode="same")

    # Only consider rows with nonzero skin width
    nonzero = region > 0
    if not np.any(nonzero):
        return None

    # Find the row with minimum width among nonzero rows
    search = region.copy()
    search[~nonzero] = np.inf
    min_idx = int(np.argmin(search))

    return search_top + min_idx


def detect_neck_midpoint_from_dual_mask(
    image: Image.Image,
    skinmap: Image.Image,
    depthmap: Image.Image,
    hairmap: Image.Image | None = None,
    threshold: float = 0.5,
    skin_threshold: int = 30,
    float_min: float | None = None,
    float_max: float | None = None,
) -> tuple[NeckMidpoint | None, SegmentationDebug | None]:
    """Detect neck midpoint using skin matte, depth map, and silhouette.

    Uses dual-mask approach:
    - Depth map + skin matte: find chin as the closest-to-camera skin pixel
    - Depth map with hair removed: find neck/shoulders from width profile

    The neck midpoint sits between chin and shoulders.

    Args:
        image: PIL Image of the portrait (RGB).
        skinmap: PIL Image "L" mode skin segmentation from iOS HEIC.
        depthmap: PIL Image "L" mode depth map from iOS HEIC.
        hairmap: PIL Image "L" mode hair matte from iOS HEIC, or None.
        threshold: Segmentation confidence threshold (0-1).
        skin_threshold: Minimum pixel value in skinmap to count as skin.
        float_min: EXIF FloatMinValue for depth calibration, or None.
        float_max: EXIF FloatMaxValue for depth calibration, or None.

    Returns:
        2-tuple of (NeckMidpoint | None, SegmentationDebug | None).
    """
    # Get segmentation mask
    seg_mask = _get_segmentation_mask(image, threshold)
    if seg_mask is None:
        return None, None

    h, w = seg_mask.shape[:2]

    # Convert skinmap to binary numpy mask
    skin_arr = np.array(skinmap)
    if skin_arr.ndim == 3:
        skin_arr = skin_arr[:, :, 0]
    skin_binary = skin_arr >= skin_threshold

    # Convert depthmap to numpy array
    depthmap_arr = np.array(depthmap)
    if depthmap_arr.ndim == 3:
        depthmap_arr = depthmap_arr[:, :, 0]

    # Convert hairmap to numpy array
    hair_arr = None
    if hairmap is not None:
        hair_arr = np.array(hairmap)
        if hair_arr.ndim == 3:
            hair_arr = hair_arr[:, :, 0]

    # Step 1: Find chin from depth map + skin mask
    chin_y, midline_x = _find_chin_from_depth(skin_binary, depthmap_arr)
    if chin_y is None or midline_x is None:
        print("  Dual-mask: could not find chin from depth map, falling back")
        return _detect_from_mask(seg_mask, h, w)

    print(f"  Dual-mask: chin_y={chin_y}, midline_x={midline_x:.1f}")

    # Step 2: Detect ear, neck, shoulder from depth profile below chin
    #         (hair removed from depth map so hanging hair doesn't inflate widths)
    ear_y, depth_neck_y, shoulder_y, neck_left_x, neck_right_x = (
        _detect_landmarks_from_depth_profile(
            depthmap_arr, skin_binary, chin_y, hair_arr=hair_arr
        )
    )
    print(
        f"  Dual-mask depth profile: ear_y={ear_y},"
        f" neck_y={depth_neck_y}, shoulder_y={shoulder_y},"
        f" neck_left_x={neck_left_x}, neck_right_x={neck_right_x}"
    )

    # Fallback: shoulders from segmentation if depth failed
    if shoulder_y is None:
        shoulder_y = _find_shoulders_from_segmentation(seg_mask, chin_y)
        if shoulder_y is not None:
            print(f"  Dual-mask: shoulder_y={shoulder_y} (seg fallback)")

    if shoulder_y is None:
        print("  Dual-mask: could not find shoulders, falling back")
        return _detect_from_mask(seg_mask, h, w)

    # Use depth-based neck_y, or fall back to skin-based
    neck_y = depth_neck_y
    if neck_y is None:
        neck_search_top = ear_y if ear_y is not None else chin_y
        neck_search_top = max(neck_search_top, chin_y)
        neck_y = _find_neck_from_skin(skin_binary, neck_search_top, shoulder_y)
    if neck_y is None:
        neck_y = int((chin_y + shoulder_y) / 2)  # fallback

    # Compute midline_x at neck_y from segmentation mask
    neck_row = seg_mask[neck_y]
    neck_pixels = np.where(neck_row)[0]
    if len(neck_pixels) > 0:
        neck_midline_x = float((neck_pixels[0] + neck_pixels[-1]) / 2)
    else:
        neck_midline_x = midline_x

    # Compute width profile for debug
    person_rows = np.where(np.any(seg_mask, axis=1))[0]
    if len(person_rows) == 0:
        return None, _make_empty_debug(seg_mask, skin_mask=skin_binary)

    person_top = person_rows[0]
    person_bottom = person_rows[-1]
    person_height = person_bottom - person_top
    roi_top = person_top
    roi_bottom = person_top + int(person_height * 0.75)
    roi_mask = seg_mask[roi_top:roi_bottom]
    widths, _, _ = _compute_width_profile(roi_mask)
    kernel_w = 15
    if kernel_w > 1 and len(widths) > kernel_w:
        k = np.ones(kernel_w) / kernel_w
        widths_smooth = np.convolve(widths, k, mode="same")
    else:
        widths_smooth = widths.copy()

    # Estimate head landmarks from skin mask
    head_top = person_top
    head_height = chin_y - head_top
    if head_height < 10:
        debug = SegmentationDebug(
            mask=seg_mask,
            width_profile=widths_smooth,
            roi_top=roi_top,
            neck_y=neck_y,
            chin_y=chin_y,
            midline_x=neck_midline_x,
            shoulder_y=shoulder_y,
            skin_mask=skin_binary,
            ear_y=ear_y,
        )
        return None, debug

    nose_y = head_top + head_height * 0.65
    nose_x = midline_x
    mouth_y = head_top + head_height * 0.80

    # Estimate mouth width from skin mask at mouth level
    skin_widths, _, _ = _compute_width_profile(skin_binary)
    mouth_row_idx = int(mouth_y)
    if 0 <= mouth_row_idx < len(skin_widths):
        head_width_at_mouth = skin_widths[mouth_row_idx]
    else:
        head_width_at_mouth = 50.0
    mouth_half_width = max(head_width_at_mouth * 0.2, 5.0)
    mouth_left = (midline_x - mouth_half_width, mouth_y)
    mouth_right = (midline_x + mouth_half_width, mouth_y)

    # Chin x from skin mask at chin_y
    chin_row_pixels = np.where(skin_binary[chin_y])[0]
    if len(chin_row_pixels) > 0:
        chin_x = float((chin_row_pixels[0] + chin_row_pixels[-1]) / 2)
    else:
        chin_x = midline_x

    # Compute 3D neck width if calibration data and neck edges are available
    neck_width_front_arc_mm = None
    neck_width_straight_mm = None
    if (
        float_min is not None
        and float_max is not None
        and neck_left_x is not None
        and neck_right_x is not None
    ):
        neck_width_front_arc_mm, neck_width_straight_mm = compute_neck_width_3d(
            depthmap,
            neck_y,
            neck_left_x,
            neck_right_x,
            image.size[0],
            image.size[1],
            float_min,
            float_max,
        )
        if neck_width_front_arc_mm is not None:
            print(
                f"  Neck width 3D: arc={neck_width_front_arc_mm:.1f} mm,"
                f" straight={neck_width_straight_mm:.1f} mm"
            )

    debug = SegmentationDebug(
        mask=seg_mask,
        width_profile=widths_smooth,
        roi_top=roi_top,
        neck_y=neck_y,
        chin_y=chin_y,
        midline_x=neck_midline_x,
        shoulder_y=shoulder_y,
        skin_mask=skin_binary,
        ear_y=ear_y,
        neck_left_x=neck_left_x,
        neck_right_x=neck_right_x,
        neck_width_front_arc_mm=neck_width_front_arc_mm,
        neck_width_straight_mm=neck_width_straight_mm,
    )

    neck_midpoint = NeckMidpoint(
        nose=(nose_x, nose_y),
        mouth_left=mouth_left,
        mouth_right=mouth_right,
        chin=(chin_x, float(chin_y)),
        neck_extended=True,
        face_flatness_ratio=None,
        pose=PortraitPose.EXTENDED_NECK,
        mouth_open_ratio=None,
        x=neck_midline_x,
        y=float(neck_y),
        left_shoulder=None,
        right_shoulder=None,
        left_shoulder_visibility=None,
        right_shoulder_visibility=None,
        nose_visibility=None,
        interpolation_ratio=None,
    )

    print(
        f"  Dual-mask result: neck_y={neck_y}, chin_y={chin_y},"
        f" ear_y={ear_y}, shoulder_y={shoulder_y},"
        f" midline_x={neck_midline_x:.1f}"
    )

    return neck_midpoint, debug


def _detect_from_mask(
    binary_mask: np.ndarray,
    h: int,
    w: int,
    jaw_flare_fraction: float = 0.15,
    smoothing_window: int = 15,
) -> tuple[NeckMidpoint | None, SegmentationDebug | None]:
    """Core detection logic operating on a binary mask.

    Separated from ``detect_neck_midpoint_from_segmentation`` so it can
    be unit-tested with synthetic masks without requiring MediaPipe.
    """
    # Check if mask has enough content
    person_pixels = np.sum(binary_mask)
    if person_pixels < (h * w * 0.01):  # less than 1% of image
        return None, _make_empty_debug(binary_mask)

    # Find vertical extent of person
    row_has_pixels = np.any(binary_mask, axis=1)
    person_rows = np.where(row_has_pixels)[0]
    person_top = person_rows[0]
    person_bottom = person_rows[-1]
    person_height = person_bottom - person_top

    if person_height < 50:
        return None, _make_empty_debug(binary_mask, person_top)

    # Analyze upper 75% of person (head + neck + upper torso)
    roi_top = person_top
    roi_bottom = person_top + int(person_height * 0.75)
    roi_mask = binary_mask[roi_top:roi_bottom]

    widths, lefts, rights = _compute_width_profile(roi_mask)

    # Smooth the width profile
    if smoothing_window > 1 and len(widths) > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        widths_smooth = np.convolve(widths, kernel, mode="same")
    else:
        widths_smooth = widths.copy()

    # Search for neck in the 25%-75% band of ROI height
    roi_h = roi_bottom - roi_top
    search_start = int(roi_h * 0.25)
    search_end = int(roi_h * 0.75)

    # Only consider rows with nonzero width
    search_band = widths_smooth[search_start:search_end]
    nonzero_mask = search_band > 0

    if not np.any(nonzero_mask):
        debug = SegmentationDebug(
            mask=binary_mask,
            width_profile=widths_smooth,
            roi_top=roi_top,
            neck_y=None,
            chin_y=None,
            midline_x=None,
        )
        return None, debug

    # Find minimum width in search band (among nonzero rows)
    search_values = search_band.copy()
    search_values[~nonzero_mask] = np.inf
    neck_idx_in_band = int(np.argmin(search_values))
    neck_idx_in_roi = search_start + neck_idx_in_band
    neck_y_abs = roi_top + neck_idx_in_roi
    neck_width = widths_smooth[neck_idx_in_roi]

    # Find chin by scanning upward from neck until jaw flare
    jaw_threshold = neck_width * (1 + jaw_flare_fraction)
    chin_idx_in_roi = neck_idx_in_roi
    for y_idx in range(neck_idx_in_roi - 1, -1, -1):
        if widths_smooth[y_idx] > jaw_threshold:
            chin_idx_in_roi = y_idx
            break
    chin_y_abs = roi_top + chin_idx_in_roi

    # Compute midline at neck
    neck_row = roi_mask[neck_idx_in_roi]
    neck_pixels = np.where(neck_row)[0]
    if len(neck_pixels) == 0:
        midline_x = float(w / 2)
    else:
        midline_x = float((neck_pixels[0] + neck_pixels[-1]) / 2)

    # Compute midline at chin
    chin_row = roi_mask[chin_idx_in_roi]
    chin_pixels = np.where(chin_row)[0]
    if len(chin_pixels) > 0:
        chin_x = float((chin_pixels[0] + chin_pixels[-1]) / 2)
    else:
        chin_x = midline_x

    # Estimate head landmarks from the silhouette (rough, for cursor placement only)
    head_top = person_top
    head_height = chin_y_abs - head_top

    if head_height < 10:
        debug = SegmentationDebug(
            mask=binary_mask,
            width_profile=widths_smooth,
            roi_top=roi_top,
            neck_y=neck_y_abs,
            chin_y=chin_y_abs,
            midline_x=midline_x,
        )
        return None, debug

    nose_y = head_top + head_height * 0.65
    nose_x = midline_x
    mouth_y = head_top + head_height * 0.80

    # Estimate mouth width as ~40% of head width at mouth level
    mouth_row_idx = int(mouth_y - roi_top)
    if 0 <= mouth_row_idx < len(widths):
        head_width_at_mouth = widths[mouth_row_idx]
    else:
        head_width_at_mouth = 50.0
    mouth_half_width = head_width_at_mouth * 0.2
    mouth_left = (midline_x - mouth_half_width, mouth_y)
    mouth_right = (midline_x + mouth_half_width, mouth_y)

    debug = SegmentationDebug(
        mask=binary_mask,
        width_profile=widths_smooth,
        roi_top=roi_top,
        neck_y=neck_y_abs,
        chin_y=chin_y_abs,
        midline_x=midline_x,
    )

    neck_midpoint = NeckMidpoint(
        nose=(nose_x, nose_y),
        mouth_left=mouth_left,
        mouth_right=mouth_right,
        chin=(chin_x, float(chin_y_abs)),
        neck_extended=True,
        face_flatness_ratio=None,
        pose=PortraitPose.EXTENDED_NECK,
        mouth_open_ratio=None,
        x=midline_x,
        y=float(neck_y_abs),
        left_shoulder=None,
        right_shoulder=None,
        left_shoulder_visibility=None,
        right_shoulder_visibility=None,
        nose_visibility=None,
        interpolation_ratio=None,
    )

    print(
        f"  Segmentation fallback: neck_y={neck_y_abs},"
        f" chin_y={chin_y_abs}, midline_x={midline_x:.1f}"
    )

    return neck_midpoint, debug
