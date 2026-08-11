"""Automatic neck circumference estimation via dense 3D arc integration.

Samples the visible front arc of the neck using the skin segmentation map
and depth data, then multiplies by an empirical factor (~3) to estimate
full circumference. Provides both the metric result and 2D/3D point lists
so that a GUI can paint the sampled arc.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .depth_sampling import median_filter_depthmap, sample_filtered_depth
from .face import find_neck_measurement_point, sample_depth_at_point
from .incisor import depth_raw_to_distance_cm, pixel_to_mm, vector_length_3d


def _ellipse_circumference(a: float, b: float) -> float:
    """Approximate ellipse perimeter using Ramanujan's formula.

    Parameters:
        a: semi-axis in mm (horizontal half-width)
        b: semi-axis in mm (front-to-back half-depth)

    Returns perimeter in mm.
    """
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


@dataclass
class NeckMeasurement:
    # The Y-level in photo-space where the neck was measured
    neck_y: int

    # Left and right skin edges at neck_y (photo-space pixels)
    left_x: int
    right_x: int

    # List of 3D arc points: [(x_mm, y_mm, z_mm), ...]
    # These are the sampled points along the neck surface in physical units.
    # The GUI can use arc_points_photo for painting overlays.
    arc_points_3d: list[tuple[float, float, float]]

    # List of 2D arc points in photo-space pixels: [(x, y), ...]
    # Same points as arc_points_3d but in pixel coordinates for GUI painting.
    arc_points_photo: list[tuple[int, int]]

    # Sum of segment lengths across the visible front arc (mm)
    front_arc_length_mm: float

    # Estimated full circumference = front_arc * circumference_multiplier (mm)
    circumference_mm: float

    # The multiplier used (kept explicit so it can be tuned)
    circumference_multiplier: float = 2.7

    # Raw silhouette coordinates before depth-stability correction.
    mask_left_x: int | None = None
    mask_right_x: int | None = None


def find_stable_depth_x_from_edge(
    depthmap,
    edge_x,
    y,
    direction,
    photo_width,
    photo_height,
    max_distance,
    stability_run=4,
):
    """Find the first stable depth patch while walking in from a skin edge.

    ``direction`` is ``1`` at the left edge and ``-1`` at the right edge.
    Sampling advances by approximately one native depth pixel. The returned
    coordinate is centred within the first run whose consecutive depth changes
    match the quiet part of the profile, avoiding the TrueDepth silhouette wall.
    """
    filtered_depthmap = median_filter_depthmap(depthmap, size=3)
    return _find_stable_depth_x_from_edge(
        filtered_depthmap,
        edge_x,
        y,
        direction,
        photo_width,
        photo_height,
        max_distance,
        stability_run,
    )


def _find_stable_depth_x_from_edge(
    filtered_depthmap,
    edge_x,
    y,
    direction,
    photo_width,
    photo_height,
    max_distance,
    stability_run,
):
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if stability_run < 3:
        raise ValueError("stability_run must be at least 3")
    if max_distance <= 0:
        return None

    native_step = max(1.0, (photo_width - 1) / max(1, filtered_depthmap.width - 1))
    sample_distances = list(np.arange(0.0, max_distance + native_step * 0.25, native_step))
    if sample_distances[-1] < max_distance:
        sample_distances.append(float(max_distance))
    positions = [
        edge_x + direction * distance for distance in sample_distances
    ]
    values = [
        sample_filtered_depth(
            filtered_depthmap,
            position,
            y,
            photo_width,
            photo_height,
        )
        for position in positions
    ]

    differences = [
        abs(right - left)
        for left, right in zip(values, values[1:])
        if left is not None and right is not None
    ]
    if len(differences) < stability_run:
        return None

    sorted_differences = sorted(differences)
    quiet_differences = sorted_differences[: max(2, len(sorted_differences) // 2)]
    quiet_median = float(np.median(quiet_differences))
    quiet_mad = float(np.median(np.abs(np.asarray(quiet_differences) - quiet_median)))
    stable_change = max(0.75, quiet_median + 3.0 * 1.4826 * quiet_mad)

    for start in range(1, len(values) - stability_run + 1):
        window = values[start : start + stability_run]
        if any(value is None for value in window):
            continue
        window_differences = [
            abs(right - left) for left, right in zip(window, window[1:])
        ]
        if max(window_differences, default=0.0) > stable_change:
            continue
        if max(window) - min(window) > stable_change * (stability_run - 1):
            continue
        stable_index = start + stability_run // 2
        return round(positions[stable_index])
    return None


def estimate_face_from_skinmap(
    skinmap: Image.Image, threshold: int = 1,
) -> tuple[int, int, int, int] | None:
    """Estimate a face bounding box from the skin segmentation map.

    Scans every row to find the widest horizontal skin extent (jaw/chin area).
    Returns a synthetic (x, y, w, h) tuple where the bottom edge sits at the
    widest row, or None if no skin pixels are found.
    """
    width, height = skinmap.size
    widest_row_y = None
    widest_width = 0
    widest_left = 0
    top_y = None

    for y in range(height):
        left_x = None
        right_x = None
        for x in range(width):
            if skinmap.getpixel((x, y)) >= threshold:
                if left_x is None:
                    left_x = x
                right_x = x

        if left_x is None:
            continue

        if top_y is None:
            top_y = y

        row_width = right_x - left_x
        if row_width > widest_width:
            widest_width = row_width
            widest_row_y = y
            widest_left = left_x

    if widest_row_y is None or top_y is None:
        return None

    face_height = widest_row_y - top_y
    if face_height <= 0:
        # Widest row is the very first skin row — no useful face region
        face_height = 1

    return (widest_left, top_y, widest_width, face_height)


def _depth_amplitude_at_sag(
    depthmap, sample_xs, neck_y, x_left, x_right,
    photo_width, photo_height, sag,
):
    """Compute max-min depth amplitude along a half-sine arc at given sag.

    *sag* is in photo-space pixels (how far the arc center dips below neck_y).
    Returns the amplitude (max depth - min depth), or None if fewer than 2
    valid depth samples could be read.
    """
    depths = []
    span = x_right - x_left
    if span <= 0:
        return None

    for sx in sample_xs:
        t = (sx - x_left) / span
        sample_y = neck_y + round(sag * math.sin(math.pi * t))
        raw = sample_depth_at_point(depthmap, sx, sample_y, photo_width, photo_height)
        if raw is not None and raw > 0:
            depths.append(raw)

    if len(depths) < 2:
        return None
    return max(depths) - min(depths)


def _find_best_sag(
    depthmap, sample_xs, neck_y, x_left, x_right,
    photo_width, photo_height,
    max_sag_photo=300, sag_step=5,
) -> int:
    """Find the arc sag (photo pixels) that minimises depth amplitude.

    Sweeps sag from 0 to *max_sag_photo* in steps of *sag_step*.
    Returns the sag whose depth amplitude is lowest.
    Returns 0 when no valid measurements can be taken.
    """
    best_sag = 0
    best_amp = None

    for sag in range(0, max_sag_photo + 1, sag_step):
        amp = _depth_amplitude_at_sag(
            depthmap, sample_xs, neck_y, x_left, x_right,
            photo_width, photo_height, sag,
        )
        if amp is None:
            continue
        if best_amp is None or amp < best_amp:
            best_amp = amp
            best_sag = sag

    return best_sag


def compute_neck_circumference(
    skinmap,  # PIL Image "L" — skin segmentation, same size as photo
    depthmap,  # PIL Image — depth map (different resolution)
    photo_width,  # int — photo width in pixels
    photo_height,  # int — photo height in pixels
    float_min,  # float — EXIF depth calibration
    float_max,  # float — EXIF depth calibration
    face_location=None,  # tuple (x, y, w, h) or None for auto-estimate from skinmap
    n_samples=25,  # number of points to sample across the neck
    skin_threshold=1,    # any nonzero pixel is skin
    circumference_multiplier=2.7,
    arc_sag=None,  # None=auto-detect; int=fixed sag in depth-map px
    face=None,  # Face object — enables eye-anchored neck search
    eyes=None,  # list of Rectangle — standalone eye detections (no face)
    image_width=None,  # int — image width for standalone eye mode
    scan_start_y=None,  # int — top of MediaPipe-bounded search range (mouth Y)
    scan_end_y=None,  # int — bottom of search range (neck midpoint Y)
    neck_midpoint_y=None,  # float — MediaPipe neck midpoint Y for arc center
) -> NeckMeasurement | None:
    """Compute neck circumference by densely sampling the front arc.

    Algorithm:
    1. Find the narrowest neck row using skin matte (existing function)
    2. Sample N evenly-spaced points across the skin width at that row
    3. For each point, read depth → convert to 3D (x_mm, y_mm, z_mm)
    4. Sum Euclidean distances between consecutive 3D points = front arc
    5. Multiply by circumference_multiplier → estimated circumference

    Returns NeckMeasurement with all data, or None if the neck cannot
    be located (e.g. no skin detected below the face).
    """
    # Neutralise white borders that some skinmaps have — paint a
    # 30-pixel black frame so border pixels are never mistaken for skin.
    skinmap = skinmap.copy()
    draw = ImageDraw.Draw(skinmap)
    border = 30
    w, h = skinmap.size
    draw.rectangle([0, 0, w - 1, border - 1], fill=0)          # top
    draw.rectangle([0, h - border, w - 1, h - 1], fill=0)      # bottom
    draw.rectangle([0, 0, border - 1, h - 1], fill=0)          # left
    draw.rectangle([w - border, 0, w - 1, h - 1], fill=0)      # right

    # Auto-estimate face location from skin map when not provided
    if face_location is None:
        face_location = estimate_face_from_skinmap(skinmap, skin_threshold)
        if face_location is None:
            return None

    # Auto-detect Face object passed as face_location (common in fidmaa-gui)
    if face is None and hasattr(face_location, "eyes"):
        face = face_location

    # Step 1: Find the narrowest neck row.
    # find_neck_measurement_point returns (x_left, y, x_right, y) — the left
    # and right skin edges at the narrowest horizontal line below the face.
    try:
        neck_pts = find_neck_measurement_point(
            skinmap, face_location, threshold=skin_threshold, face=face,
            eyes=eyes, image_width=image_width,
            scan_start_y=scan_start_y, scan_end_y=scan_end_y,
        )
    except (IndexError, ValueError):
        # No valid neck measurement found (e.g. no skin rows below face)
        return None

    x_left, neck_y, x_right, _ = neck_pts
    mask_left_x = x_left
    mask_right_x = x_right

    # Sanity check: need at least a few pixels of skin width
    if x_right <= x_left:
        return None

    neck_width = x_right - x_left
    amplitude = None
    if neck_midpoint_y is not None:
        # Arc from narrowest row (edges) down to neck midpoint (center).
        # The narrowest row is above the midpoint; the arc sags to the
        # midpoint level at the center of the neck.
        # Push the edge points 1/3 down toward the midpoint so the arc
        # doesn't start too high at the skin edges.
        full_gap = max(0, round(neck_midpoint_y) - neck_y)
        edge_offset = round(full_gap / 3)
        neck_y = neck_y + edge_offset
        amplitude = full_gap - edge_offset

    # Median-filter once, then walk inward from both segmentation edges until
    # the depth profile settles. This replaces the fixed 5% inset, which can
    # stop either inside a broad silhouette wall or unnecessarily far inward.
    filtered_depthmap = median_filter_depthmap(depthmap, size=3)
    max_edge_search = max(6, round(neck_width * 0.20))
    stable_left = _find_stable_depth_x_from_edge(
        filtered_depthmap,
        x_left,
        neck_y,
        1,
        photo_width,
        photo_height,
        max_edge_search,
        4,
    )
    stable_right = _find_stable_depth_x_from_edge(
        filtered_depthmap,
        x_right,
        neck_y,
        -1,
        photo_width,
        photo_height,
        max_edge_search,
        4,
    )
    fallback_inset = round(neck_width * 0.05)
    x_left = stable_left if stable_left is not None else x_left + fallback_inset
    x_right = stable_right if stable_right is not None else x_right - fallback_inset
    if x_right <= x_left:
        return None

    # Step 2: Generate n_samples evenly-spaced x-coordinates from x_left to x_right.
    step = (x_right - x_left) / max(n_samples - 1, 1)
    sample_xs = [round(x_left + i * step) for i in range(n_samples)]

    if amplitude is None:
        if arc_sag is None:
            amplitude = _find_best_sag(
                depthmap,
                sample_xs,
                neck_y,
                x_left,
                x_right,
                photo_width,
                photo_height,
            ) // 2
        else:
            # Manual arc_sag is in depth-map pixels; scale to photo resolution.
            amplitude = arc_sag * photo_height / depthmap.size[1]

    # Step 3: For each sample point, compute Y via half-sine arc,
    # read depth and convert to 3D coordinates.
    #
    # Depth is read from the same-size median-filtered copy of the depth map,
    # bilinearly sampled at the (fractional) native-resolution coordinate.
    # This smooths TrueDepth sensor noise before it can accumulate across
    # the many points walked along the arc -- the same fix applied to
    # fidmaa-gui's surface_vector_filtered() for straight-line measurements.
    arc_points_3d = []
    arc_points_photo = []

    for sx in sample_xs:
        # Half-sine arc: edges at neck_y, center dips by amplitude
        t = (sx - x_left) / (x_right - x_left)
        sample_y = neck_y + round(amplitude * math.sin(math.pi * t))

        raw_depth = sample_filtered_depth(
            filtered_depthmap, sx, sample_y, photo_width, photo_height,
        )
        if raw_depth is None:
            # Skip points where depth data is missing or zero (invalid disparity)
            continue

        # Convert raw depth pixel value to physical distance in cm
        z_cm = depth_raw_to_distance_cm(raw_depth, float_min, float_max)
        if z_cm is None:
            continue

        # Convert pixel coordinates to physical mm at this depth
        x_mm = pixel_to_mm(sx, z_cm, photo_width)
        y_mm = pixel_to_mm(sample_y, z_cm, photo_height)
        if x_mm is None or y_mm is None:
            continue

        # Z in mm for consistent units
        z_mm = z_cm * 10.0

        arc_points_3d.append((x_mm, y_mm, z_mm))
        arc_points_photo.append((sx, sample_y))

    # Need at least 2 points to compute any arc length
    if len(arc_points_3d) < 2:
        return None

    # Step 4: Sum Euclidean distances between consecutive 3D points.
    # This gives the front arc length across the visible neck surface.
    front_arc_length_mm = 0.0
    for i in range(1, len(arc_points_3d)):
        p0 = arc_points_3d[i - 1]
        p1 = arc_points_3d[i]
        front_arc_length_mm += vector_length_3d(
            p0[0], p0[1], p0[2],
            p1[0], p1[1], p1[2],
        )

    # Step 5: Estimate full circumference via empirical multiplier.
    # front_arc_mm * 2.7 ≈ circumference_mm (i.e. front_arc_mm * 0.27 = circumference_cm)
    circumference_mm = front_arc_length_mm * circumference_multiplier

    return NeckMeasurement(
        neck_y=neck_y,
        left_x=x_left,
        right_x=x_right,
        arc_points_3d=arc_points_3d,
        arc_points_photo=arc_points_photo,
        front_arc_length_mm=front_arc_length_mm,
        circumference_mm=circumference_mm,
        circumference_multiplier=circumference_multiplier,
        mask_left_x=mask_left_x,
        mask_right_x=mask_right_x,
    )
