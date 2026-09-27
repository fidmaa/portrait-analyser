"""Neck width (and width-based circumference) on TrueDepth capture-app photos.

For frontal portraits from the TrueDepth capture app (full-precision float
depth with *absolute* accuracy + the file's camera intrinsics). Camera-app
(legacy 8-bit) files are not measured here: :func:`measure_neck_width`
returns None for them and the older neck detectors
(:mod:`~portrait_analyser.neck`, :mod:`~portrait_analyser.extended_neck`) are
untouched.

Why width and not the front arc
-------------------------------
On a frontal portrait the throat and the true sides of the neck are rarely
visible in the same image row: just under the jaw the sides are clean
silhouettes but the chin (and a beard) hides the throat; lower down the
throat is visible but the sides are hidden by the shirt collar. Front-arc and
sagitta based circumference models therefore swing by 15-30 % with the
chosen row, while the *lateral width* just under the jaw varies by about 1 %
between neighbouring rows of one photo. This module measures only that
width. The float depth itself does not show usable lateral edges (collar and
shirt sit at the neck's depth), so:

* **edges** come from the portrait **skin matte**: per row, the outer 50 %
  crossings of the skin run around the facial midline (the chin column;
  small gaps such as a dark beard or shadow are bridged);
* **depth only validates** each edge: if the depth just *outside* the edge
  is nearer than the depth just *inside* it (by
  :data:`OCCLUDER_MARGIN_CM`), the edge belongs to an occluder -- a collar,
  a hand -- not to the neck silhouette, and the row is rejected. Between
  :data:`COLLAR_CLOSE_MARGIN_CM` and that margin the row is kept but the
  result is marked low quality with a "collar close to the neck edge"
  warning;
* the **tangent depth** of each edge is a median read :data:`EDGE_INSET_MM`
  inside it, and the width is the 3-D distance of the two edge points
  (pinhole, file intrinsics).

The inside reference for the occluder test is that same inset depth, not the
depth at the edge itself: on real maps the depth within ~1 mm of the
silhouette is mixed with whatever lies behind or in front of it (IMG_2389,
right edge: 47.0-47.5 cm at the edge against 43.5-45.0 cm 3 mm inside, the
background being 48-50 cm), so an edge reading would hide a collar in front
and be biased by the background behind.

Width bias
----------
``width_mm`` reads **low** by construction. The inset depth is nearer than
the true tangent point, and the tangent chord of a round neck is
``2 R cos(alpha)`` (``alpha`` = half the angle the neck subtends, ~0.8 %).
On an ideal circular cylinder (R = 65 mm at 45 cm) the method gives about
124 mm instead of 130 mm (-4.5 %); on real, smoothed TrueDepth maps the
inset matters less (IMG_2389: 136.1 / 134.5 / 133.4 / 132.0 mm at 2 / 3 / 5 /
10 mm inset), so the real shortfall is roughly 0-5 %. No cylinder correction
is applied.

The band searched runs from the chin (MediaPipe FaceMesh landmark 152) down
to Apple Vision's ``neck_1_joint`` (the neck base between the shoulders; see
:mod:`~portrait_analyser.apple_vision`), or -- without Vision (not macOS, no
person found) -- down to :data:`CHIN_OFFSET_BAND_MM` below the chin. The
reported width is the median over the topmost contiguous run of clean rows,
after dropping rows off that run's median by more than
:data:`WIDTH_STABILITY_TOLERANCE`; the run must span at least
:data:`MIN_SUPPORT_MM` vertically. Rows are 1 mm apart but each reads depth
over a ~5-7 mm window, so neighbouring rows are not independent: support is
stated in mm, not in rows. The slope, midline, symmetry, jaw-width and
stability thresholds are **provisional heuristics** tuned on eight photos of
one person; ``quality`` (``"good"``/``"low"`` with reasons) is the outlet for
borderline cases.

Circumference
-------------
Width alone does not fix the circumference; two width-based values are given:

* ``circumference_circle_mm`` = pi * W: the circle model (a circle of
  diameter W). It is **not** a bound: a neck is wider than deep, which makes
  pi * W high, but W itself reads low (above).
* ``circumference_ellipse_mm``: Ramanujan's ellipse perimeter with
  ``a = W / 2`` and ``b = r * a`` for ``r`` in
  :data:`ELLIPSE_AXIS_RATIO_RANGE` (0.85-0.90), i.e. about 0.93-0.95 * pi * W.
  The ratio ``b / a`` is a **population assumption**, not measured on the
  photo.

The b/a prior and the width's inset bias are currently **co-calibrated on one
person** (collar size 41 cm; manual width 130.3 mm on IMG_2386): the ellipse
range 39-40 cm matches that person only because both errors are whatever
they are on his photos. Treat the circumference as indicative.

Repeatability: on the same person, IMG_2389 gives 134.5 mm and IMG_2363 (a
collar-limited photo, now rejected by the symmetry check) gave 128.9 mm --
about 4 % between photos, i.e. roughly +-1.6 cm of circumference. The ~1 %
above is only the row-to-row stability within one photo. Validation:
IMG_2389 134.5 mm (Vision band); IMG_2386 carries no intrinsics
(``"no-camera"`` by default; 133.4 mm with the prototype's assumed fx 2765
passed as ``camera=``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from .apple_vision import BodyPose
from .exceptions import AppleVisionError, AppleVisionUnavailable
from .incisor import distance_3d_from_cm
from .neck import ellipse_circumference

logger = logging.getLogger(__name__)

# -- statuses ------------------------------------------------------------------

STATUS_OK = "ok"
STATUS_EDGES_OCCLUDED = "edges-occluded"
STATUS_NO_FACE = "no-face"
STATUS_NO_DEPTH = "no-depth"
# Float depth but no usable camera intrinsics: a metric width would need
# invented intrinsics, so none is given.
STATUS_NO_CAMERA = "no-camera"
# Depth accuracy is not "absolute" (relative depth has an unknown scale):
# never measured, not even with an explicit camera.
STATUS_RELATIVE_DEPTH = "relative-depth"
# The measurement raised (only set by IOSPortrait.neck_width, which caches it).
STATUS_ERROR = "error"

QUALITY_GOOD = "good"
QUALITY_LOW = "low"

BAND_SOURCE_VISION = "vision-neck"
BAND_SOURCE_CHIN_OFFSET = "chin-offset"
BAND_SOURCE_MANUAL = "manual"

EDGES_OCCLUDED_MESSAGE = (
    "neck edges not visible -- take the photo from further away / include the neck below the chin"
)

# -- parameters (metric; converted to pixels with the camera and depth) --------

# FaceMesh landmarks: chin (menton), nose tip, jaw angles (gonion region)
# and cheek extremes. The neck is searched straight down the photo, so the
# nose -> chin direction must be within 45 degrees of "down".
FACE_MESH_CHIN_INDEX = 152
FACE_MESH_NOSE_INDEX = 1
FACE_MESH_JAW_INDICES = (172, 397)

# Rows start this far below the chin (the chin's own outline is not the neck).
CHIN_CLEARANCE_MM = 6.5

# Band bottom without Apple Vision: this far below the chin. The clean
# silhouette rows on the validation photos lie 9-40 mm below the chin, and
# Vision's neck joint 100-135 mm.
CHIN_OFFSET_BAND_MM = 60.0

# Vision's neck joint is used only with at least this confidence (0.60-0.71
# on the validation photos).
VISION_NECK_MIN_CONFIDENCE = 0.3

# Vertical spacing of the evaluated rows.
ROW_STEP_MM = 1.0

# The skin matte is averaged over this many rows (centred) before the 50 %
# crossing is taken.
SKIN_ROW_AVERAGE = 9
SKIN_MATTE_ON = 128

# Gaps in the skin run narrower than this are bridged (dark beard / shadow
# under the chin drops out of the matte).
SKIN_GAP_BRIDGE_MM = 10.0

# Skin runs are joined only within this distance of the midline.
SKIN_SEARCH_HALF_WIDTH_MM = 115.0

# A row needs at least this much skin in total.
SKIN_MIN_SPAN_MM = 8.0

# The tangent depth is read this far inside each edge (see "Width bias") ...
EDGE_INSET_MM = 3.0
# ... and the occluder check reads the depth this far outside it.
OUTSIDE_OFFSET_MM = 10.0

# Median window radius (depth pixels) on the 3x3-median-filtered float map.
DEPTH_WINDOW_RADIUS = 2

# An edge is occluded when the outside is nearer than the inside by at least
# this much (a collar in front of the neck side) ...
OCCLUDER_MARGIN_CM = 0.5
# ... and "close" (kept, low quality, warning) when nearer by more than this.
COLLAR_CLOSE_MARGIN_CM = 0.2

# The inside-edge depth must lie within this of the chin depth (else the
# "edge" samples background or a shoulder, not the neck).
MAX_EDGE_BEHIND_CHIN_CM = 20.0

# Provisional heuristic: a neck silhouette just below the jaw is close to
# vertical; an edge that runs more obliquely than this (|dx/dy|, ~19 degrees)
# is the V of an open collar or the jaw line. Measured over
# +-EDGE_SLOPE_BASELINE_MM of the row.
MAX_EDGE_SLOPE = 0.35
EDGE_SLOPE_BASELINE_MM = 2.0

# Provisional heuristic: the mid-point of the two edges may be off the chin
# column by at most this fraction of the edge distance.
MAX_MIDLINE_OFFSET_FRACTION = 0.2

# Provisional heuristic: the larger half-width (chin column to an edge) may
# be at most this multiple of the smaller one; a hand or other skin next to
# one side of the neck widens one half only. Validation: 1.00-1.10 on clean
# frontal photos, 1.38 for a synthetic 25 mm hand. Above the "low" ratio the
# result is marked low quality.
MAX_HALF_WIDTH_RATIO = 1.3
LOW_QUALITY_HALF_WIDTH_RATIO = 1.15

# Provisional heuristic: the neck (edge distance, px) may be at most this
# multiple of the FaceMesh jaw width (landmarks 172-397). Validation photos:
# 0.93-1.03.
MAX_NECK_TO_JAW_WIDTH = 1.1

# Left/right tangent depths differing by more than this reject the row
# (one edge is not on the neck); a manual measurement only warns.
MAX_TANGENT_DEPTH_ASYMMETRY_CM = 4.0

# Provisional heuristic: rows whose width is off the run median by more than
# this are dropped.
WIDTH_STABILITY_TOLERANCE = 0.04

# A width needs clean, stable rows spanning at least this much vertically
# (a single passing row is usually a coincidence where a collar edge happens
# to look like a silhouette) ...
MIN_SUPPORT_MM = 2.0
# ... and is "low" quality below this (IMG_2389's run spans 3.7 mm).
GOOD_SUPPORT_MM = 5.0

# A clean run is broken by a gap (rows without skin edges) taller than this.
MAX_RUN_GAP_MM = 3.0

# Horizontal rows measure W / cos(roll) on a rolled head/neck: warn (and
# mark low quality) above this.
MAX_ROLL_DEG = 8.0

# Left/right tangent depths differing by more than this -> "head rotated".
TANGENT_DEPTH_ASYMMETRY_WARN_CM = 3.0

# b / a of the neck cross-section ellipse (population assumption).
ELLIPSE_AXIS_RATIO_RANGE = (0.85, 0.90)


@dataclass
class NeckWidthRow:
    """One evaluated image row (for overlays: green = ok, magenta = rejected).

    ``left_ok`` / ``right_ok``: that edge passes the per-edge checks (depth
    inside valid, outside not nearer, edge near vertical). ``reject_reason``
    is None for a clean row, else why the row was not used. Depths in cm;
    None where invalid. ``left_slope``/``right_slope``: signed ``dx/dy`` of
    the edge over +-:data:`EDGE_SLOPE_BASELINE_MM` (None = not evaluated).
    """

    y: float
    left_x: float
    right_x: float
    left_depth_cm: float | None
    right_depth_cm: float | None
    left_outside_cm: float | None
    right_outside_cm: float | None
    width_mm: float | None
    left_ok: bool
    right_ok: bool
    left_slope: float | None = None
    right_slope: float | None = None
    reject_reason: str | None = None

    @property
    def clean(self) -> bool:
        return self.reject_reason is None

    def outside_minus_inside_cm(self) -> list[float]:
        """``outside - inside`` depth (cm) of each edge where both are known."""
        return [
            outside - inside
            for inside, outside in (
                (self.left_depth_cm, self.left_outside_cm),
                (self.right_depth_cm, self.right_outside_cm),
            )
            if inside is not None and outside is not None
        ]


@dataclass
class NeckWidthResult:
    """Neck width at the silhouette just below the jaw, in photo pixels / mm.

    :ivar status: ``"ok"``, ``"edges-occluded"``, ``"no-face"``,
        ``"no-depth"``, ``"no-camera"``, ``"relative-depth"`` or ``"error"``;
        the measurement fields are None unless ``"ok"``.
    :ivar quality: ``"good"`` or ``"low"`` (only for ``"ok"``), with
        ``quality_reasons``
    :ivar row_y: the reported row (the used row whose width is nearest the
        median), photo px
    :ivar left_x: left silhouette edge at ``row_y``, photo px
    :ivar right_x: right silhouette edge at ``row_y``, photo px
    :ivar width_mm: 3-D width, median over ``rows_used``; reads low by
        0-5 % (see the module docstring, "Width bias")
    :ivar rows_used: y of the clean rows the median was taken over
    :ivar support_mm: vertical extent of ``rows_used``, mm at neck depth
    :ivar height_below_chin_mm: ``row_y`` below the chin landmark, mm in the
        image plane at the neck-edge depth
    :ivar roll_deg: head roll (nose -> chin from vertical, degrees, signed)
    :ivar neck_roll_deg: neck-axis roll from the used rows' edge slopes
    :ivar band: ``(top_y, bottom_y)`` searched, photo px (top = chin)
    :ivar band_source: ``"vision-neck"`` (bottom = Vision neck joint),
        ``"chin-offset"`` (bottom = chin + :data:`CHIN_OFFSET_BAND_MM`) or
        ``"manual"`` (:func:`neck_width_from_edges`)
    :ivar circumference_circle_mm: pi * width, the circle model (not a bound)
    :ivar circumference_ellipse_mm: ``(low, high)`` Ramanujan perimeter for
        ``b / a`` over :data:`ELLIPSE_AXIS_RATIO_RANGE` (a population
        assumption, not measured)
    :ivar left_depth_cm: tangent depth of the left edge at ``row_y``
    :ivar right_depth_cm: tangent depth of the right edge at ``row_y``
    :ivar warnings: human-readable caveats
    :ivar message: why there is no number (status != "ok"), else None
    :ivar rows: every evaluated row, for overlays
    :ivar chin: ``(x, y)`` chin landmark used, photo px
    :ivar neck_joint: Vision ``(x, y, confidence)`` neck joint, or None
    """

    status: str
    quality: str | None = None
    quality_reasons: list[str] = field(default_factory=list)
    row_y: float | None = None
    left_x: float | None = None
    right_x: float | None = None
    width_mm: float | None = None
    rows_used: list[float] = field(default_factory=list)
    support_mm: float | None = None
    height_below_chin_mm: float | None = None
    roll_deg: float | None = None
    neck_roll_deg: float | None = None
    band: tuple[float, float] | None = None
    band_source: str | None = None
    circumference_circle_mm: float | None = None
    circumference_ellipse_mm: tuple[float, float] | None = None
    left_depth_cm: float | None = None
    right_depth_cm: float | None = None
    warnings: list[str] = field(default_factory=list)
    message: str | None = None
    rows: list[NeckWidthRow] = field(default_factory=list)
    chin: tuple[float, float] | None = None
    neck_joint: tuple[float, float, float] | None = None


def circumference_circle(width_mm: float) -> float:
    """pi * W: perimeter of a circle of diameter ``width_mm`` (circle model)."""
    return math.pi * width_mm


def circumference_ellipse_range(
    width_mm: float, ratio_range=ELLIPSE_AXIS_RATIO_RANGE
) -> tuple[float, float]:
    """Ramanujan ellipse perimeters for ``a = W / 2``, ``b = r * a``, ``r`` in
    ``ratio_range``; returned as ``(low, high)``."""
    a = width_mm / 2.0
    values = [ellipse_circumference(a, ratio * a) for ratio in ratio_range]
    return min(values), max(values)


# -- helpers --------------------------------------------------------------------


def _px_per_mm(camera, z_cm: float) -> float:
    """Photo pixels per mm (x axis) at camera distance ``z_cm``."""
    return camera.fx / (z_cm * 10.0)


def _landmarks(face_mesh):
    """Landmark list from a ``FaceMeshDebug`` or a plain sequence."""
    landmarks = getattr(face_mesh, "landmarks", face_mesh)
    if landmarks is None or len(landmarks) <= FACE_MESH_CHIN_INDEX:
        raise ValueError(f"face_mesh needs at least {FACE_MESH_CHIN_INDEX + 1} FaceMesh landmarks")
    return landmarks


def _jaw_width_px(landmarks) -> float | None:
    """Distance between the FaceMesh jaw landmarks, or None if not available."""
    left, right = FACE_MESH_JAW_INDICES
    if len(landmarks) <= max(left, right):
        return None
    width = abs(float(landmarks[right][0]) - float(landmarks[left][0]))
    return width if width > 0 else None


def _skin_array(skinmap, photo_size):
    skin = skinmap.convert("L")
    if skin.size != tuple(photo_size):
        skin = skin.resize(tuple(photo_size))
    return np.asarray(skin, dtype=np.float32)


def skin_edges_at_row(skin, y, mid_x, *, gap_px, search_px, min_span_px):
    """Outer skin-matte edges of the neck at photo row ``y`` (sub-pixel), or None.

    The matte is averaged over :data:`SKIN_ROW_AVERAGE` rows; runs at or above
    :data:`SKIN_MATTE_ON` separated by less than ``gap_px`` are bridged; runs
    overlapping ``mid_x +- search_px`` are joined and must straddle
    ``mid_x``. The edges are the 50 % crossings of the joined span's outer
    ends. None when there is too little skin, the span does not straddle the
    midline, or it reaches the image border (edge not visible).
    """
    height, width = skin.shape
    half = SKIN_ROW_AVERAGE // 2
    y = round(y)
    if y - half < 0 or y + half >= height:
        return None
    profile = skin[y - half : y + half + 1].mean(axis=0)
    on = np.flatnonzero(profile >= SKIN_MATTE_ON)
    if on.size < min_span_px:
        return None
    breaks = np.flatnonzero(np.diff(on) > gap_px)
    starts = np.r_[on[0], on[breaks + 1]]
    ends = np.r_[on[breaks], on[-1]]
    runs = [
        (int(s), int(e))
        for s, e in zip(starts, ends)
        if e > mid_x - search_px and s < mid_x + search_px
    ]
    if not runs:
        return None
    a, b = runs[0][0], runs[-1][1]
    if not a < mid_x < b or a == 0 or b == width - 1:
        return None

    def crossing(i, step):
        # 50 % crossing between i - step (off) and i (on).
        j = i - step
        v0, v1 = profile[j], profile[i]
        return float(i) if v1 == v0 else j + step * (SKIN_MATTE_ON - v0) / (v1 - v0)

    return crossing(a, 1), crossing(b, -1)


def _depth_along(depth, x, y, ux, uy, offset_px):
    """Median depth (cm) at ``(x, y) + offset_px * (ux, uy)``, or None."""
    return depth.distance_cm(x + offset_px * ux, y + offset_px * uy, DEPTH_WINDOW_RADIUS)


def _edge_depths(depth, camera, x, y, ux, uy, z_ref_cm):
    """``(inside_cm, outside_cm)`` for an edge at ``(x, y)`` whose inward
    direction is the unit vector ``(ux, uy)``.

    Inside: :data:`EDGE_INSET_MM` inward (two passes: the offset in pixels is
    recomputed at the depth found). Outside: :data:`OUTSIDE_OFFSET_MM`
    outward.
    """
    inside = _depth_along(depth, x, y, ux, uy, EDGE_INSET_MM * _px_per_mm(camera, z_ref_cm))
    if inside is not None:
        refined = _depth_along(depth, x, y, ux, uy, EDGE_INSET_MM * _px_per_mm(camera, inside))
        inside = refined if refined is not None else inside
    z_out_ref = inside if inside is not None else z_ref_cm
    outside = _depth_along(depth, x, y, ux, uy, -OUTSIDE_OFFSET_MM * _px_per_mm(camera, z_out_ref))
    return inside, outside


def _edge_ok(inside, outside, z_ref_cm):
    if inside is None or inside > z_ref_cm + MAX_EDGE_BEHIND_CHIN_CM:
        return False
    return outside is None or outside - inside > -OCCLUDER_MARGIN_CM


def _evaluate_edges(depth, camera, photo_size, left_xy, right_xy, z_ref_cm):
    """A :class:`NeckWidthRow` for a pair of edge points (inward = towards
    each other)."""
    (xl, yl), (xr, yr) = left_xy, right_xy
    dx, dy = xr - xl, yr - yl
    norm = math.hypot(dx, dy)
    if norm == 0:
        raise ValueError("the two neck edge points coincide")
    ux, uy = dx / norm, dy / norm
    left_in, left_out = _edge_depths(depth, camera, xl, yl, ux, uy, z_ref_cm)
    right_in, right_out = _edge_depths(depth, camera, xr, yr, -ux, -uy, z_ref_cm)
    width = None
    if left_in is not None and right_in is not None:
        result = distance_3d_from_cm(
            (xl, yl), (xr, yr), left_in, right_in, photo_size[0], photo_size[1], camera
        )
        width = None if result is None else result[0]
    left_ok = _edge_ok(left_in, left_out, z_ref_cm)
    right_ok = _edge_ok(right_in, right_out, z_ref_cm)
    reason = None
    if width is None:
        reason = "no depth inside an edge"
    elif not (left_ok and right_ok):
        reason = "edge occluded (outside nearer) or edge depth not on the neck"
    return NeckWidthRow(
        y=0.5 * (yl + yr),
        left_x=xl,
        right_x=xr,
        left_depth_cm=left_in,
        right_depth_cm=right_in,
        left_outside_cm=left_out,
        right_outside_cm=right_out,
        width_mm=width,
        left_ok=left_ok,
        right_ok=right_ok,
        reject_reason=reason,
    )


def _half_width_ratio(row: NeckWidthRow, mid_x: float) -> float:
    left_half = mid_x - row.left_x
    right_half = row.right_x - mid_x
    if min(left_half, right_half) <= 0:
        return math.inf
    return max(left_half, right_half) / min(left_half, right_half)


def _apply_shape_checks(
    row: NeckWidthRow, edges_above, edges_below, baseline_px, mid_x, jaw_width_px
):
    """Reject rows whose edges are oblique, off the midline, asymmetric,
    wider than the jaw or at very different depths (see the module
    constants)."""
    if edges_above is not None and edges_below is not None:
        span = 2.0 * baseline_px
        row.left_slope = (edges_below[0] - edges_above[0]) / span
        row.right_slope = (edges_below[1] - edges_above[1]) / span
    else:
        row.left_slope = row.right_slope = None
    reasons = [] if row.reject_reason is None else [row.reject_reason]
    if row.left_slope is None:
        reasons.append("edge slope unknown (no skin edges next to the row)")
    else:
        for side, slope in (("left", row.left_slope), ("right", row.right_slope)):
            if abs(slope) > MAX_EDGE_SLOPE:
                reasons.append(f"{side} edge oblique (|dx/dy| {abs(slope):.2f}; collar V?)")
                if side == "left":
                    row.left_ok = False
                else:
                    row.right_ok = False
    edge_distance = row.right_x - row.left_x
    offset = abs(0.5 * (row.left_x + row.right_x) - mid_x)
    if edge_distance <= 0 or offset > MAX_MIDLINE_OFFSET_FRACTION * edge_distance:
        reasons.append("edges not centred on the facial midline")
    elif _half_width_ratio(row, mid_x) > MAX_HALF_WIDTH_RATIO:
        reasons.append(
            "edges asymmetric about the facial midline (hand or skin next to the neck, "
            "or head turned?)"
        )
    if jaw_width_px is not None and edge_distance > MAX_NECK_TO_JAW_WIDTH * jaw_width_px:
        reasons.append("neck wider than the jaw (hand or skin next to the neck?)")
    if (
        row.left_depth_cm is not None
        and row.right_depth_cm is not None
        and abs(row.left_depth_cm - row.right_depth_cm) > MAX_TANGENT_DEPTH_ASYMMETRY_CM
    ):
        reasons.append("left/right edge depths too different (one edge not on the neck)")
    row.reject_reason = "; ".join(reasons) if reasons else None


def _edge_warnings(row: NeckWidthRow) -> list[str]:
    warnings = []
    for side, inside, outside in (
        ("left", row.left_depth_cm, row.left_outside_cm),
        ("right", row.right_depth_cm, row.right_outside_cm),
    ):
        if inside is None:
            warnings.append(f"no depth inside the {side} neck edge")
        elif outside is not None and outside - inside <= -OCCLUDER_MARGIN_CM:
            warnings.append(
                f"{side} edge: outside nearer than neck edge (collar?) by {inside - outside:.1f} cm"
            )
        elif outside is not None and outside - inside < -COLLAR_CLOSE_MARGIN_CM:
            warnings.append(
                f"{side} edge: collar close to the neck edge (outside nearer by "
                f"{inside - outside:.1f} cm)"
            )
    if (
        row.left_depth_cm is not None
        and row.right_depth_cm is not None
        and abs(row.left_depth_cm - row.right_depth_cm) > TANGENT_DEPTH_ASYMMETRY_WARN_CM
    ):
        warnings.append(
            f"left/right edge depths differ by "
            f"{abs(row.left_depth_cm - row.right_depth_cm):.1f} cm (head rotated?)"
        )
    return warnings


def _has_occluded_edge(row: NeckWidthRow) -> bool:
    """Either edge has its outside nearer than its inside (an occluder)."""
    return any(d <= -OCCLUDER_MARGIN_CM for d in row.outside_minus_inside_cm())


def _topmost_clean_run(rows, max_gap_px):
    """The first contiguous run of clean rows. Rows without skin edges are
    not in ``rows``; such a gap breaks the run only when taller than
    ``max_gap_px``."""
    run = []
    for row in rows:
        if row.clean and (not run or row.y - run[-1].y <= max_gap_px):
            run.append(row)
        elif run:
            break
    return run


def _check_portrait(portrait, camera):
    """None if measurable, ``"legacy"`` for Camera-app files (nothing is
    measured), else ``(status, message)``."""
    depth = getattr(portrait, "depth", None)
    if depth is None:
        return STATUS_NO_DEPTH, "the portrait has no depth map"
    if not depth.is_float:
        return "legacy"
    if getattr(portrait, "depth_plausible", None) is False:
        return STATUS_NO_DEPTH, (
            "capture-app depth is implausible or not aligned with the photo "
            "(depth_plausible is False)"
        )
    accuracy = getattr(portrait, "depth_accuracy", None)
    if accuracy != "absolute":
        return STATUS_RELATIVE_DEPTH, (
            f"depth accuracy is {accuracy!r}, not 'absolute': its scale is unknown, "
            "so no metric neck width is given (not even with an explicit camera)"
        )
    if camera is None:
        if getattr(portrait, "focal_length_px", None) is None:
            return STATUS_NO_CAMERA, (
                "the file carries no camera intrinsics (calibration data); a metric "
                "neck width would need assumed intrinsics -- pass camera= explicitly "
                "to measure with them"
            )
        return STATUS_NO_CAMERA, (
            "the file's intrinsics are not usable (depth not absolute, not plausible "
            "or not aligned with the photo)"
        )
    return None


def _band_bottom(portrait, body_pose, use_vision, chin_y, z_chin_cm, camera, warnings):
    """``(bottom_y, band_source, neck_joint)``."""
    if body_pose is None and use_vision:
        from .apple_vision import detect_body_pose

        try:
            body_pose = detect_body_pose(portrait.photo)
        except AppleVisionUnavailable as exc:
            logger.info("Apple Vision unavailable (%s); neck band from the chin offset", exc)
        except AppleVisionError as exc:
            logger.warning(
                "Apple Vision body pose failed; neck band from the chin offset", exc_info=True
            )
            warnings.append(f"Apple Vision body pose failed ({exc}); band from chin offset")
    neck = None if body_pose is None else body_pose.neck(VISION_NECK_MIN_CONFIDENCE)
    min_bottom = chin_y + 2 * CHIN_CLEARANCE_MM * _px_per_mm(camera, z_chin_cm)
    if neck is not None and neck[1] > min_bottom:
        return float(neck[1]), BAND_SOURCE_VISION, neck
    if body_pose is not None:
        warnings.append("Vision neck joint not found below the chin; band from chin offset")
    bottom = chin_y + CHIN_OFFSET_BAND_MM * _px_per_mm(camera, z_chin_cm)
    return float(bottom), BAND_SOURCE_CHIN_OFFSET, neck


def _finish(result: NeckWidthResult, row: NeckWidthRow, width_mm: float):
    result.row_y = row.y
    result.left_x = row.left_x
    result.right_x = row.right_x
    result.left_depth_cm = row.left_depth_cm
    result.right_depth_cm = row.right_depth_cm
    result.width_mm = width_mm
    result.circumference_circle_mm = circumference_circle(width_mm)
    result.circumference_ellipse_mm = circumference_ellipse_range(width_mm)
    return result


def _mean_edge_depth_cm(row: NeckWidthRow) -> float:
    return 0.5 * (row.left_depth_cm + row.right_depth_cm)


# -- public API -----------------------------------------------------------------


def measure_neck_width(
    portrait,
    *,
    face_mesh=None,
    body_pose: BodyPose | None = None,
    camera=None,
    use_vision: bool = True,
) -> NeckWidthResult | None:
    """Automatic neck width just below the jaw (capture-app photos only).

    :param portrait: an :class:`~portrait_analyser.ios.IOSPortrait`
    :param face_mesh: FaceMesh landmarks (``FaceMeshDebug`` or a sequence of
        478 ``(x, y)`` photo points); None = run
        :func:`~portrait_analyser.pose.detect_face_mesh`
    :param body_pose: :class:`~portrait_analyser.apple_vision.BodyPose`; None
        = run Apple Vision when ``use_vision`` (macOS), else chin offset
    :param camera: intrinsics to measure with; None = ``portrait.camera``
        (the file's own). Passing assumed intrinsics is the caller's decision;
        relative depth is refused either way.
    :param use_vision: False skips Apple Vision (band from the chin offset)
    :returns: None for Camera-app (legacy 8-bit) files -- nothing is measured
        there. Otherwise a :class:`NeckWidthResult`; check ``status`` and
        ``quality``.
    """
    if camera is None:
        camera = getattr(portrait, "camera", None)
    problem = _check_portrait(portrait, camera)
    if problem == "legacy":
        return None
    if problem is not None:
        return NeckWidthResult(status=problem[0], message=problem[1])

    photo_size = portrait.photo.size
    if portrait.skinmap is None:
        return NeckWidthResult(
            status=STATUS_EDGES_OCCLUDED,
            message="the file has no skin matte, so the neck edges cannot be found",
        )

    if face_mesh is None:
        from .pose import detect_face_mesh

        face_mesh = detect_face_mesh(portrait.photo)
        if face_mesh is None:
            return NeckWidthResult(status=STATUS_NO_FACE, message="FaceMesh found no face")
    landmarks = _landmarks(face_mesh)
    chin_x, chin_y = (float(v) for v in landmarks[FACE_MESH_CHIN_INDEX])
    nose_x, nose_y = (float(v) for v in landmarks[FACE_MESH_NOSE_INDEX])
    if not chin_y - nose_y > abs(chin_x - nose_x):
        return NeckWidthResult(
            status=STATUS_NO_FACE,
            message="face not upright in the photo (rotated photo or head?); "
            "the neck is searched straight below the chin",
            chin=(chin_x, chin_y),
        )
    roll_deg = math.degrees(math.atan2(chin_x - nose_x, chin_y - nose_y))
    jaw_width_px = _jaw_width_px(landmarks)

    depth = portrait.depth.median_filtered(3)
    z_chin = depth.distance_cm(chin_x, chin_y, DEPTH_WINDOW_RADIUS)
    if z_chin is None:
        return NeckWidthResult(
            status=STATUS_NO_DEPTH, message="no depth at the chin", chin=(chin_x, chin_y)
        )

    warnings: list[str] = []
    bottom, band_source, neck_joint = _band_bottom(
        portrait, body_pose, use_vision, chin_y, z_chin, camera, warnings
    )
    bottom = min(bottom, photo_size[1] - 1 - SKIN_ROW_AVERAGE // 2)
    result = NeckWidthResult(
        status=STATUS_EDGES_OCCLUDED,
        band=(chin_y, bottom),
        band_source=band_source,
        chin=(chin_x, chin_y),
        neck_joint=neck_joint,
        roll_deg=roll_deg,
        warnings=warnings,
    )

    px_mm = _px_per_mm(camera, z_chin)
    skin = _skin_array(portrait.skinmap, photo_size)
    step = max(1.0, ROW_STEP_MM * px_mm)
    y = chin_y + CHIN_CLEARANCE_MM * px_mm
    baseline = EDGE_SLOPE_BASELINE_MM * px_mm

    def edges_at(row_y):
        return skin_edges_at_row(
            skin,
            row_y,
            chin_x,
            gap_px=SKIN_GAP_BRIDGE_MM * px_mm,
            search_px=SKIN_SEARCH_HALF_WIDTH_MM * px_mm,
            min_span_px=SKIN_MIN_SPAN_MM * px_mm,
        )

    while y <= bottom:
        row_y = float(round(y))
        edges = edges_at(row_y)
        if edges is not None:
            row = _evaluate_edges(
                depth, camera, photo_size, (edges[0], row_y), (edges[1], row_y), z_chin
            )
            _apply_shape_checks(
                row,
                edges_at(row_y - baseline),
                edges_at(row_y + baseline),
                baseline,
                chin_x,
                jaw_width_px,
            )
            result.rows.append(row)
        y += step

    run = _topmost_clean_run(result.rows, MAX_RUN_GAP_MM * px_mm)
    stable = []
    if run:
        median = float(np.median([r.width_mm for r in run]))
        stable = [r for r in run if abs(r.width_mm - median) <= WIDTH_STABILITY_TOLERANCE * median]
    support_mm = 0.0
    if stable:
        neck_z = float(np.median([_mean_edge_depth_cm(r) for r in stable]))
        support_mm = (stable[-1].y - stable[0].y) / _px_per_mm(camera, neck_z)
    if not stable or support_mm < MIN_SUPPORT_MM:
        result.message = (
            f"{EDGES_OCCLUDED_MESSAGE} ({len(result.rows)} rows with skin edges; clean, "
            f"stable rows span {support_mm:.1f} mm, {MIN_SUPPORT_MM:.0f} mm needed)"
        )
        occluded = sum(1 for r in result.rows if _has_occluded_edge(r))
        oblique = sum(1 for r in result.rows if r.reject_reason and "oblique" in r.reject_reason)
        if occluded:
            result.warnings.append(f"outside nearer than neck edge (collar?) on {occluded} rows")
        if oblique:
            result.warnings.append(f"oblique edges (collar V / jaw line) on {oblique} rows")
        return result

    width = float(np.median([r.width_mm for r in stable]))
    result.rows_used = [r.y for r in stable]
    result.support_mm = support_mm
    if len(stable) < len(run):
        result.warnings.append(
            f"{len(run) - len(stable)} clean rows dropped as unstable width "
            f"(> {WIDTH_STABILITY_TOLERANCE:.0%} off the median)"
        )
    # Report the used row nearest the median width (the first of ties).
    chosen = min(stable, key=lambda r: abs(r.width_mm - width))
    result.height_below_chin_mm = (
        (chosen.y - chin_y) * _mean_edge_depth_cm(chosen) * 10.0 / (camera.fy)
    )
    slopes = [
        0.5 * (r.left_slope + r.right_slope)
        for r in stable
        if r.left_slope is not None and r.right_slope is not None
    ]
    if slopes:
        result.neck_roll_deg = math.degrees(math.atan(float(np.median(slopes))))
    result.warnings.extend(_edge_warnings(chosen))
    _assess_quality(result, stable, chin_x)
    result.status = STATUS_OK
    return _finish(result, chosen, width)


def _assess_quality(result: NeckWidthResult, stable, chin_x):
    """Set ``quality``/``quality_reasons`` (and the matching warnings)."""
    reasons = []
    close = [
        d
        for r in stable
        for d in r.outside_minus_inside_cm()
        if -OCCLUDER_MARGIN_CM < d < -COLLAR_CLOSE_MARGIN_CM
    ]
    if close:
        reasons.append(
            f"collar close to the neck edge (outside nearer by up to {-min(close):.1f} cm)"
        )
        if not any("collar close" in w for w in result.warnings):
            result.warnings.append(reasons[-1])
    if result.support_mm < GOOD_SUPPORT_MM:
        reasons.append(
            f"clean rows span only {result.support_mm:.1f} mm (< {GOOD_SUPPORT_MM:.0f} mm)"
        )
    ratio = max(_half_width_ratio(r, chin_x) for r in stable)
    if ratio > LOW_QUALITY_HALF_WIDTH_RATIO:
        reasons.append(f"edges asymmetric about the facial midline (half-width ratio {ratio:.2f})")
    for label, angle in (("head", result.roll_deg), ("neck", result.neck_roll_deg)):
        if angle is not None and abs(angle) > MAX_ROLL_DEG:
            text = f"{label} roll {angle:.1f} deg: horizontal rows read the width / cos(roll)"
            reasons.append(text)
            result.warnings.append(text)
    result.quality_reasons = reasons
    result.quality = QUALITY_LOW if reasons else QUALITY_GOOD


def neck_width_from_edges(portrait, left_xy, right_xy, *, camera=None) -> NeckWidthResult | None:
    """Semi-automatic neck width from two user-clicked edge points (photo px).

    The tangent depth of each click is a median :data:`EDGE_INSET_MM` inward
    (towards the other click); the depth :data:`OUTSIDE_OFFSET_MM` outward
    drives the occluder warnings ("outside nearer than neck edge (collar?)",
    "collar close to the neck edge") but does not reject the measurement --
    the user chose the points. The width reads low like the automatic one
    (module docstring, "Width bias").

    :returns: None for Camera-app (legacy) files; otherwise a
        :class:`NeckWidthResult` with ``band_source="manual"``,
        ``rows_used=[row_y]``, width, the circle model and the ellipse range;
        status ``"no-depth"`` when either click has no depth inside it.
    """
    if camera is None:
        camera = getattr(portrait, "camera", None)
    problem = _check_portrait(portrait, camera)
    if problem == "legacy":
        return None
    if problem is not None:
        return NeckWidthResult(
            status=problem[0], message=problem[1], band_source=BAND_SOURCE_MANUAL
        )
    left_xy = (float(left_xy[0]), float(left_xy[1]))
    right_xy = (float(right_xy[0]), float(right_xy[1]))
    if left_xy[0] > right_xy[0]:
        left_xy, right_xy = right_xy, left_xy
    depth = portrait.depth.median_filtered(3)
    mid = (0.5 * (left_xy[0] + right_xy[0]), 0.5 * (left_xy[1] + right_xy[1]))
    # Reference distance for the mm -> px offsets: the edges themselves.
    refs = [
        z
        for z in (
            depth.distance_cm(*left_xy, DEPTH_WINDOW_RADIUS),
            depth.distance_cm(*right_xy, DEPTH_WINDOW_RADIUS),
            depth.distance_cm(*mid, DEPTH_WINDOW_RADIUS),
        )
        if z is not None
    ]
    result = NeckWidthResult(status=STATUS_NO_DEPTH, band_source=BAND_SOURCE_MANUAL)
    if not refs:
        result.message = "no depth at or between the clicked points"
        return result
    z_ref = min(refs)
    row = _evaluate_edges(depth, camera, portrait.photo.size, left_xy, right_xy, z_ref)
    result.rows = [row]
    result.warnings = _edge_warnings(row)
    if row.width_mm is None:
        result.message = "no depth inside one of the clicked edges"
        result.row_y, result.left_x, result.right_x = row.y, row.left_x, row.right_x
        return result
    result.status = STATUS_OK
    result.rows_used = [row.y]
    result.band = (row.y, row.y)
    manual_reasons = [
        w for w in result.warnings if "collar" in w or "differ by" in w or "no depth" in w
    ]
    result.quality_reasons = manual_reasons
    result.quality = QUALITY_LOW if manual_reasons else QUALITY_GOOD
    return _finish(result, row, row.width_mm)
