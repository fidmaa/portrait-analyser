"""The tape path around the front of the neck and circumference models in
its tilted cross-section plane (capture-app photos, after
:func:`~portrait_analyser.neck_width.measure_neck_width`).

A tape measure put around the neck does not follow an image row: it rests
on the neck sides just below the jaw angles and dips at the front under the
chin -- and under the lower border of a beard. A horizontal row through the
side points cuts through the chin/beard (IMG_2389: the front of that row
reads 37 cm against 45-46 cm at the sides), so no front-arc model on a row
can work. This module:

1. **Front point.** Walks down the facial midline (the FaceMesh jaw-centre
   column) from the chin landmark. With a beard -- no skin under the chin
   (the skin matte is off there; Apple's hair matte does not cover beards
   on the validation photos, so it is only used when it does) -- the tape
   has to pass below the beard's lower border: first row with skin (or no
   hair) for :data:`BEARD_BORDER_RUN_MM`, plus :data:`BEARD_CLEARANCE_MM`.
   With or without a beard it has to pass below the chin: on the depth
   profile along the midline (the integration map) the chin falls away
   steeply (dz/dy above :data:`CHIN_FALL_MIN_SLOPE`) and then levels off on
   the throat; the chin ends at the first row after the steepest fall with
   dz/dy below :data:`CHIN_END_SLOPE`. The front point is the lower of the
   two. Without either (no beard, no chin fall inside the band) it is
   inferred :data:`FRONT_FALLBACK_MM` below the chin and the result is low
   quality ("front point inferred").

2. **Tape path.** A curve from the left side point through the front point
   to the right side point, shaped like the FaceMesh lower-jaw contour
   (172 ... 152 ... 397): the jaw's sag below its jaw-angle chord, mapped
   across the side points and scaled to pass through the front point (the
   "clearing sag" idea: parallel to the jaw, lowered until it clears the
   chin/beard). Without a usable contour a parabola. Sampled every
   :data:`PATH_STEP_MM`.

3. **Plane.** The side points in 3-D (edge column, depth
   :data:`~portrait_analyser.neck_width.EDGE_INSET_MM` inside it, as for the
   width) and the front point (integration map) are the three key points;
   the path samples between the two inset columns are back-projected on the
   integration map (the board-validated smoothing of
   :meth:`~portrait_analyser.depth_map.DepthMap.integration_map`). The
   cross-section plane is the least-squares plane through the path samples
   and the front point (``plane_tilt_deg``); the plane through the three key
   points is reported too (``plane_tilt_3pt_deg``) but not used: the inset
   side depths sit well in front of the true lateral extremes (IMG_2389's
   right side reads the same depth as the front point), which tilts that
   plane by 10-25 degrees more and leaves the path samples 12 mm off it
   (0.5-2 mm off the least-squares plane). The samples are expressed
   in-plane: ``u`` along the (projected) side chord, ``v`` towards the front.

Circumference models (all reported, see :class:`NeckSagResult`):

* ``fitted_circle``: least-squares circle (algebraic start + Gauss-Newton
  refinement to the geometric fit) through the in-plane path samples (the
  front point is one of them), ``2 pi R``. **Headline**
  (:data:`HEADLINE_MODEL`).
* ``circle``: ``pi W`` (``W`` = 3-D distance of the side points).
* ``arc_circle``: the circle with chord ``W`` and arc = the 3-D path length
  (the GUI's arc-circle), ``2 pi R``.
* ``sagitta_ellipse``: Ramanujan ellipse, ``a = W / 2``, ``b`` = the front
  point's in-plane sagitta from the side chord (assumes the side points are
  the lateral extremes).
* ``fitted_ellipse``: ``a = W / 2``, centred on the side chord, ``b``
  least-squares fitted to the in-plane samples.
* the ``b/a``-prior ellipse range of :mod:`~portrait_analyser.neck_width`
  (``circumference_prior_ellipse_mm``), unchanged, for comparison.

Why the fitted circle
---------------------
Each model's sensitivity is measured on every result
(``sensitivity_detail[perturbation][model]`` = signed relative change,
``sensitivity[model]`` = the largest ``|change|``, None where the perturbed
model could not be computed). Perturbations: the side columns moved so the
pixel width changes by +-:data:`SENSITIVITY_WIDTH_FRACTION`; the front
point moved +-:data:`SENSITIVITY_FRONT_MM` along the midline; and
:data:`SENSITIVITY_END_DROP` of the samples dropped at each end of the arc.
To the side columns the fitted circle is the least sensitive model (only
the front-arc samples enter it): 0.8-2.5 % on IMG_2389/2363 and ~2 % on the
synthetic necks, against 4-5 % for pi W, 5-7 % for the arc circle and
7-10 % for the two ``a = W / 2`` ellipses. It is **not** insensitive to the
arc itself: its curvature comes mostly from the outer ~20 % of the arc
(where it turns towards the sides), so dropping 10 % of the samples at
each end moves it by -7.4 % (IMG_2389) / +4.8 % (IMG_2363), and moving the
front point by 5 mm by -6.3 / +4.2 % (IMG_2363; -1.4 % on IMG_2389). It is
also the only model consistent between the two photos of one person
(41.8 / 41.7 cm, collar size 41 cm; pi W 42.2 / 39.0, arc circle 42.5 /
42.0). The ``a = W / 2`` ellipses read 20-25 % low everywhere: the side
points are not the lateral extremes of the section in depth (see Plane).
A free ellipse fit (centre and both axes) is ill-conditioned on the ~140
degrees of arc visible: +40-60 % on noise-free synthetic data.

Bias of the fitted circle: it measures the curvature of the visible arc,
so its error depends on the (unseen) shape of the section. On synthetic
tilted elliptic necks (lateral 64 mm, axis leaning 15 degrees) against the
true perimeter of the plane section: front-back 48 / 56 / 60 / 64 / 72 mm
-> +22 / +4.5 / 0.0 / -4.7 / -16.5 % (pi W: +6 / -1 / -4 / -6 / -13 %);
leaning 10 / 20 / 25 degrees at 60 mm: -4 / +4 / +9 %. Within ~3 % only
where the section is near-circular; a section flatter at the front than at
the sides reads high, one elongated front to back reads low.

Quality gate
------------
``quality`` is ``"low"`` (with ``quality_reasons``) when the front point
was inferred, clamped to the band bottom, or its search saw a beard without
a lower border or two chin falls; when the front point is not below / in
front of the side points; when more than :data:`MAX_INVALID_SAMPLE_FRACTION`
of the path has no depth; when the headline falls back to pi W; and for the
fitted circle when (a) it and pi W disagree by more than
:data:`MAX_CIRCLE_VS_WIDTH` (flags the flat-front case: +15 % at
front-back 48 mm; IMG_2389 -1 %, IMG_2363 +7 %), (b) ``R / (W/2)`` is
outside :data:`CIRCLE_RADIUS_RATIO_RANGE` (validation 0.96-1.15; a
radius beyond :data:`MAX_FIT_RADIUS_FACTOR` x W/2 is no circle at all),
(c) the samples span less than :data:`MIN_CIRCLE_SPAN_DEG` of it
(validation 106-138 degrees) or lie more than :data:`MAX_CIRCLE_RMS_MM`
RMS off it (validation 0.2-1.2 mm), or (d) a perturbation moves it by more
than :data:`MAX_HEADLINE_SENSITIVITY`. A deep section (front-back 72 mm,
-16.5 %) passes the gate: nothing visible distinguishes it. The tilt of a real tape (front lower than the sides)
lengthens the section front to back, which tends to make it rounder. The
thresholds and the front-point rules are provisional, tuned on two photos
of one person (user annotations of the tape and the side points).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from . import neck_width as nw
from .neck import ellipse_circumference

SAG_STATUS_OK = "ok"
SAG_STATUS_FAILED = "failed"

FRONT_SOURCE_BEARD = "beard-border"
FRONT_SOURCE_CHIN = "chin-end"
FRONT_SOURCE_INFERRED = "inferred"

MODEL_FITTED_CIRCLE = "fitted_circle"
MODEL_SAGITTA_ELLIPSE = "sagitta_ellipse"
MODEL_FITTED_ELLIPSE = "fitted_ellipse"
MODEL_ARC_CIRCLE = "arc_circle"
MODEL_CIRCLE = "circle"
MODEL_PRIOR_ELLIPSE = "prior_ellipse"

# -- front point ----------------------------------------------------------------

# The midline skin/hair profile is averaged over this half-width (mm).
MIDLINE_HALF_WIDTH_MM = 2.0
# Vertical step of the midline walk.
MIDLINE_STEP_MM = 1.0
# "Beard": no skin (or hair) over this much right below the chin landmark.
BEARD_PROBE_MM = 5.0
# Hair matte value counted as hair (0-255).
HAIR_MATTE_ON = 128
# The beard's lower border: first row from which the midline is skin (and
# not hair) for this long ...
BEARD_BORDER_RUN_MM = 3.0
# ... and the tape passes this far below it (the beard hairs hang a few mm
# over the skin border). The user's tape on IMG_2389 runs ~9 mm below the
# matte border, but there the chin-end rule already puts the front point
# there (2466 vs the user's 2464, the border + 5 mm is 2439); on IMG_2363
# the border + 5 mm (2741) is the lower one, and a larger clearance would
# push it further down the throat than the chin end (2718) supports. 5 mm is
# the smallest clearance that keeps the tape off the beard hairs.
BEARD_CLEARANCE_MM = 5.0
# Depth slope along the midline, mm of depth per mm down (image plane at the
# depth read). The chin falls away at 2.5-6.5 on the validation photos; a
# profile whose steepest fall stays below CHIN_FALL_MIN_SLOPE has no chin
# edge in the band. The chin ends where the slope drops below CHIN_END_SLOPE
# after the steepest fall (IMG_2389: y 2474, the user's tape 2463; IMG_2363:
# 2716).
CHIN_FALL_MIN_SLOPE = 1.5
CHIN_END_SLOPE = 1.0
# The slope is a central difference over +- this.
SLOPE_BASELINE_MM = 2.0
# The midline walk covers at most this far below the chin (and never past
# the band bottom).
FRONT_SEARCH_MM = 70.0
# Without a beard border or chin end: the front point this far below the
# chin (both validation photos: ~40-45 mm), flagged low quality.
FRONT_FALLBACK_MM = 40.0

# -- path and plane -----------------------------------------------------------------

# Path sample spacing (mm, at the neck).
PATH_STEP_MM = 2.0
# More than this fraction of path samples without depth: low quality.
MAX_INVALID_SAMPLE_FRACTION = 0.1
# Fewer valid in-plane samples than this: no fitted models.
MIN_FIT_SAMPLES = 8

# Side-point placement sensitivity: side columns moved so the pixel width
# changes by +- this fraction (each side half of it).
SENSITIVITY_WIDTH_FRACTION = 0.03

# Front-point and end-sample perturbations of the sensitivity check.
SENSITIVITY_FRONT_MM = 5.0
SENSITIVITY_END_DROP = 0.10

# Circle fit: Gauss-Newton refinement steps after the algebraic start, and
# the largest radius accepted (x the half width; beyond it the samples are
# nearly collinear and the "circle" is a line).
CIRCLE_GN_ITERATIONS = 10
MAX_FIT_RADIUS_FACTOR = 3.0

# Quality gate of the headline (module docstring, "Quality gate").
MAX_CIRCLE_VS_WIDTH = 0.10
CIRCLE_RADIUS_RATIO_RANGE = (0.8, 1.4)
MIN_CIRCLE_SPAN_DEG = 100.0
MAX_CIRCLE_RMS_MM = 2.0
MAX_HEADLINE_SENSITIVITY = 0.10

# Headline model: the least sensitive to the side-point placement (module
# docstring, "Why the fitted circle").
HEADLINE_MODEL = MODEL_FITTED_CIRCLE


@dataclass
class NeckSagResult:
    """Tape path with a front sag and circumference models in its plane.

    Photo px for points; mm for lengths; degrees for angles.

    :ivar status: ``"ok"`` or ``"failed"`` (``message`` says why)
    :ivar quality: ``"good"``/``"low"`` with ``quality_reasons``: the tape's
        own (module docstring, "Quality gate"); the width's reasons are not
        repeated here -- ``NeckWidthResult.circumference_quality`` combines
        both
    :ivar left_xy/right_xy: side points (the width's edges)
    :ivar front_xy: front point of the tape (midline)
    :ivar front_source: ``"beard-border"``, ``"chin-end"`` or ``"inferred"``
    :ivar beard_border_y: midline skin border under a beard, or None
    :ivar chin_end_y: where the chin's depth fall levels off, or None
    :ivar path: the tape path, left -> right, photo px
    :ivar plane_tilt_deg: angle between the tape plane (least squares
        through the path samples and the front point) and the camera's
        horizontal x-z plane (a horizontal cut of an upright neck = 0)
    :ivar plane_tilt_3pt_deg: the same for the plane through the two side
        points and the front point (diagnostic; module docstring, "Plane")
    :ivar plane_normal: unit normal of the tape plane (camera space)
    :ivar plane_origin_mm: in-plane origin (the projected side-chord
        midpoint), camera mm
    :ivar front_drop_deg: signed angle of the front point below the side
        chord in the plane's front direction (positive = front lower)
    :ivar width_mm: 3-D side-point distance ``W``
    :ivar sagitta_mm: front point's in-plane distance from the side chord
    :ivar front_arc_mm: 3-D length of the path (side point to side point)
    :ivar off_plane_rms_mm: RMS distance of the path samples from the plane
    :ivar circle_radius_mm: fitted circle radius (in-plane)
    :ivar circle_rms_mm: geometric RMS of the samples off the fitted circle
    :ivar circle_span_deg: angle the samples subtend at its centre
    :ivar ellipse_b_mm: fitted-ellipse semi-axis towards the front
    :ivar circumferences_mm: ``{model: mm}`` for every model computed
    :ivar circumference_prior_ellipse_mm: ``(low, high)`` b/a-prior range
    :ivar headline_mm/headline_model: the reported circumference
    :ivar sensitivity: ``{model: max |relative change|}`` over the
        perturbations of ``sensitivity_detail`` (None = no perturbed value)
    :ivar sensitivity_detail: ``{perturbation: {model: signed relative
        change or None}}`` for the side columns (+-3 % pixel width), the
        front point (+-5 mm) and 10 % of the samples dropped at each end
    :ivar profile_mm: in-plane ``(u, v)`` of the samples used for the fits
        (the front point is among them)
    :ivar path_3d_mm: camera-space points of the path (side points
        included), for plots
    """

    status: str
    quality: str | None = None
    quality_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    message: str | None = None
    left_xy: tuple[float, float] | None = None
    right_xy: tuple[float, float] | None = None
    front_xy: tuple[float, float] | None = None
    front_source: str | None = None
    beard_border_y: float | None = None
    chin_end_y: float | None = None
    path: list[tuple[float, float]] = field(default_factory=list)
    plane_tilt_deg: float | None = None
    plane_tilt_3pt_deg: float | None = None
    plane_normal: tuple[float, float, float] | None = None
    plane_origin_mm: tuple[float, float, float] | None = None
    front_drop_deg: float | None = None
    width_mm: float | None = None
    sagitta_mm: float | None = None
    front_arc_mm: float | None = None
    off_plane_rms_mm: float | None = None
    circle_radius_mm: float | None = None
    circle_centre_mm: tuple[float, float] | None = None
    ellipse_b_mm: float | None = None
    circumferences_mm: dict[str, float] = field(default_factory=dict)
    circumference_prior_ellipse_mm: tuple[float, float] | None = None
    headline_mm: float | None = None
    headline_model: str | None = None
    circle_rms_mm: float | None = None
    circle_span_deg: float | None = None
    sensitivity: dict[str, float | None] = field(default_factory=dict)
    sensitivity_detail: dict[str, dict[str, float | None]] = field(default_factory=dict)
    profile_mm: list[tuple[float, float]] = field(default_factory=list)
    path_3d_mm: list[tuple[float, float, float]] = field(default_factory=list)


# -- geometry helpers ------------------------------------------------------------


def _back_project(camera, x, y, z_mm) -> np.ndarray:
    return np.array(
        [(x - camera.cx) * z_mm / camera.fx, (y - camera.cy) * z_mm / camera.fy, z_mm]
    )


def fit_circle_2d(points, max_radius=None) -> tuple[float, float, float] | None:
    """Least-squares circle ``(cx, cy, r)`` through 2-D points, or None
    (fewer than 3 points, degenerate/collinear, or ``r > max_radius``).

    The algebraic (Kasa) fit is the start; :data:`CIRCLE_GN_ITERATIONS`
    Gauss-Newton steps then minimise the geometric distances
    ``sum (|p - c| - r)^2``. On a partial arc Kasa is biased towards a
    smaller circle (IMG_2389/2363 and the synthetic necks: the geometric
    fit is 0.3-1 % larger); the refinement removes that bias and does not
    change the fit on exact circles."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    u, v = pts[:, 0], pts[:, 1]
    a = np.column_stack([u, v, np.ones_like(u)])
    rhs = u * u + v * v
    sol, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    cu, cv = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cu * cu + cv * cv
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = math.sqrt(r2)
    for _ in range(CIRCLE_GN_ITERATIONS):
        du, dv = u - cu, v - cv
        dist = np.hypot(du, dv)
        if np.any(dist < 1e-9):
            break
        jac = np.column_stack([-du / dist, -dv / dist, -np.ones_like(u)])
        step, *_ = np.linalg.lstsq(jac, -(dist - r), rcond=None)
        if not np.all(np.isfinite(step)):
            break
        cu, cv, r = cu + step[0], cv + step[1], r + step[2]
    if not (np.isfinite(r) and r > 0):
        return None
    if max_radius is not None and r > max_radius:
        return None
    return float(cu), float(cv), float(r)


def circle_fit_stats(points, circle) -> tuple[float, float]:
    """``(rms_mm, span_deg)``: geometric RMS residual of ``points`` from
    ``circle`` and the angle the points subtend at its centre."""
    pts = np.asarray(points, dtype=float)
    cu, cv, r = circle
    du, dv = pts[:, 0] - cu, pts[:, 1] - cv
    rms = float(np.sqrt(np.mean((np.hypot(du, dv) - r) ** 2)))
    ang = np.unwrap(np.arctan2(dv, du)[np.argsort(pts[:, 0])])
    return rms, float(math.degrees(ang.max() - ang.min()))


def fit_ellipse_b(points, a: float) -> float | None:
    """Semi-axis ``b`` of the ellipse ``u^2/a^2 + v^2/b^2 = 1`` (centred at
    the origin, ``a`` given) best fitting points with ``v > 0``:
    least squares of ``v`` against ``sqrt(1 - u^2/a^2)``."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return None
    g = 1.0 - (pts[:, 0] / a) ** 2
    keep = g > 0.05  # the ends are ill-conditioned (and at the inset edge)
    if keep.sum() < 3:
        return None
    s = np.sqrt(g[keep])
    b = float(np.dot(s, pts[keep, 1]) / np.dot(s, s))
    return b if b > 0 else None


def arc_circle_radius(chord: float, arc: float) -> float | None:
    """Radius of the circle on which an arc of length ``arc`` spans chord
    ``chord`` (``arc / chord = (t/2) / sin(t/2)``, ``t`` in (0, 2 pi)).
    None when ``arc <= chord``."""
    if chord <= 0 or arc <= chord:
        return None
    ratio = arc / chord
    lo, hi = 1e-6, math.pi - 1e-9  # half-angle
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mid / math.sin(mid) < ratio:
            lo = mid
        else:
            hi = mid
    half = 0.5 * (lo + hi)
    return chord / (2.0 * math.sin(half))


# -- front point --------------------------------------------------------------------


@dataclass
class _FrontSearch:
    y: float
    source: str
    beard_border_y: float | None
    chin_end_y: float | None
    flags: list[str] = field(default_factory=list)


def _column_mean(matte, x, y, half):
    height, width = matte.shape
    yi = min(max(round(y), 0), height - 1)
    x0, x1 = max(round(x - half), 0), min(round(x + half) + 1, width)
    return float(matte[yi, x0:x1].mean())


def find_front_point(
    skin, hair, smooth, camera, x, chin_y, bottom_y, z_chin_cm
) -> _FrontSearch:
    """Front point of the tape on the column ``x`` below ``chin_y`` (see the
    module docstring). ``skin``/``hair``: photo-size float mattes (hair may
    be None); ``smooth``: the integration map."""
    px_mm = camera.fy / (z_chin_cm * 10.0)
    step = max(1.0, MIDLINE_STEP_MM * px_mm)
    end = min(bottom_y, chin_y + FRONT_SEARCH_MM * px_mm, skin.shape[0] - 1)
    ys = np.arange(chin_y, end, step)
    half = MIDLINE_HALF_WIDTH_MM * px_mm

    def skin_on(y):
        on = _column_mean(skin, x, y, half) >= nw.SKIN_MATTE_ON
        if hair is not None and _column_mean(hair, x, y, half) >= HAIR_MATTE_ON:
            on = False
        return on

    flags: list[str] = []
    on = np.array([skin_on(y) for y in ys], dtype=bool)
    beard_border = None
    probe = max(1, round(BEARD_PROBE_MM / MIDLINE_STEP_MM))
    run = max(1, round(BEARD_BORDER_RUN_MM / MIDLINE_STEP_MM))
    if len(on) > probe and not on[:probe].any():
        for i in range(probe, len(on) - run + 1):
            if on[i : i + run].all():
                beard_border = float(ys[i])
                break
        if beard_border is None:
            flags.append(
                f"beard detected (no skin under the chin) but its lower border not "
                f"found within {(end - chin_y) / px_mm:.0f} mm"
            )

    chin_end = None
    z = np.array(
        [np.nan if (v := smooth.bilinear_cm(x, y)) is None else v * 10.0 for y in ys]
    )
    k = max(1, round(SLOPE_BASELINE_MM / MIDLINE_STEP_MM))
    if len(ys) > 2 * k + 1:
        # dz / dY: Y in mm in the image plane at the depth read.
        dz = z[2 * k :] - z[: -2 * k]
        dy_mm = (ys[2 * k :] - ys[: -2 * k]) * z[k:-k] / camera.fy
        slope = dz / dy_mm
        mid_ys = ys[k:-k]
        if np.isfinite(slope).any():
            peak = int(np.nanargmax(slope))
            if slope[peak] >= CHIN_FALL_MIN_SLOPE:
                after = np.flatnonzero(slope[peak:] < CHIN_END_SLOPE)
                if after.size:
                    chin_end = float(mid_ys[peak + after[0]])
                falls = _count_falls(slope)
                if falls > 1:
                    flags.append(
                        f"{falls} depth falls below the chin (double chin / skin fold?); "
                        "the front point follows the steepest"
                    )

    candidates = []
    if beard_border is not None:
        candidates.append(
            (beard_border + BEARD_CLEARANCE_MM * px_mm, FRONT_SOURCE_BEARD)
        )
    if chin_end is not None:
        candidates.append((chin_end, FRONT_SOURCE_CHIN))
    if candidates:
        y, source = max(candidates)
    else:
        y, source = chin_y + FRONT_FALLBACK_MM * px_mm, FRONT_SOURCE_INFERRED
    if y > bottom_y:
        flags.append(
            f"front point clamped to the band bottom ({(y - bottom_y) / px_mm:.0f} mm "
            "higher than found)"
        )
        y = bottom_y
    return _FrontSearch(float(y), source, beard_border, chin_end, flags)


def _count_falls(slope) -> int:
    """Separate runs of the midline slope at or above
    :data:`CHIN_FALL_MIN_SLOPE`, split by a stretch below
    :data:`CHIN_END_SLOPE` (a chin, then a second fold)."""
    falls, inside, settled = 0, False, True
    for value in slope:
        if not np.isfinite(value):
            continue
        if value >= CHIN_FALL_MIN_SLOPE:
            if not inside and settled:
                falls += 1
            inside, settled = True, False
        else:
            inside = False
            if value < CHIN_END_SLOPE:
                settled = True
    return falls


# -- tape path ----------------------------------------------------------------------


def _jaw_sag_shape(jaw):
    """``(u, c)``: the lower jaw contour's sag below its jaw-angle chord
    (px, positive down) against the normalised position ``u`` (0 = left
    jaw angle, 1 = right); None when unusable."""
    if jaw is None:
        return None
    pts = jaw.points[np.argsort(jaw.points[:, 0])]
    (x0, y0), (x1, y1) = jaw.left_angle, jaw.right_angle
    if x1 - x0 <= 0:
        return None
    u = (pts[:, 0] - x0) / (x1 - x0)
    c = pts[:, 1] - (y0 + (y1 - y0) * u)
    if not (np.diff(u) > 0).all() or c.max() <= 0:
        return None
    return u, c


def tape_path(left_xy, right_xy, front_xy, jaw, step_px) -> list[tuple[float, float]]:
    """The tape path through the side points and the front point, shaped
    like the lower jaw contour (module docstring, step 2). Includes the
    three points; left -> right."""
    (xl, yl), (xr, yr), (xf, yf) = left_xy, right_xy, front_xy
    if not xl < xf < xr:
        raise ValueError(
            f"front point x {xf:.0f} not between the side points ({xl:.0f}, {xr:.0f})"
        )
    n = max(3, math.ceil((xr - xl) / step_px) + 1)
    xs = np.union1d(np.linspace(xl, xr, n), [xf])
    u = (xs - xl) / (xr - xl)
    chord = yl + (yr - yl) * u
    uf = (xf - xl) / (xr - xl)
    shape = _jaw_sag_shape(jaw)

    def sag(uu):
        if shape is None:
            return 4.0 * uu * (1.0 - uu)
        return np.interp(uu, shape[0], shape[1], left=0.0, right=0.0)

    cf = float(sag(uf))
    if cf <= 1e-6:
        shape = None
        cf = float(sag(uf))
    scale = (yf - (yl + (yr - yl) * uf)) / cf if cf > 0 else 0.0
    ys = chord + scale * sag(u)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


# -- 3-D and models -----------------------------------------------------------------


def _side_points(depth, camera, left_xy, right_xy, z_ref_cm):
    """3-D side points (edge column, depth EDGE_INSET_MM inside) or None."""
    (xl, yl), (xr, yr) = left_xy, right_xy
    dx, dy = xr - xl, yr - yl
    norm = math.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm
    zl, _ = nw._edge_depths(depth, camera, xl, yl, ux, uy, z_ref_cm)
    zr, _ = nw._edge_depths(depth, camera, xr, yr, -ux, -uy, z_ref_cm)
    if zl is None or zr is None:
        return None
    return _back_project(camera, xl, yl, zl * 10.0), _back_project(
        camera, xr, yr, zr * 10.0
    )


@dataclass
class _Models:
    width: float
    sagitta: float
    arc: float | None
    tilt: float
    tilt3: float
    drop: float
    normal: tuple
    origin: tuple
    off_plane: float | None
    circle: tuple[float, float, float] | None
    circle_rms: float | None
    circle_span: float | None
    ellipse_b: float | None
    circumferences: dict
    profile: list
    path_3d: list
    invalid_fraction: float


def _plane_normal_3pt(p_left, p_right, p_front):
    normal = np.cross(p_right - p_left, p_front - p_left)
    norm = np.linalg.norm(normal)
    return None if norm < 1e-9 else normal / norm


def _models(
    depth, smooth, camera, left_xy, right_xy, front_xy, jaw, z_ref_cm, *, end_drop=0.0
):
    """All geometry and models for one placement of the three key points.
    ``end_drop``: fraction of the path samples dropped at *each* end before
    the plane and the fits (sensitivity check; the arc keeps them)."""
    sides = _side_points(depth, camera, left_xy, right_xy, z_ref_cm)
    z_front = smooth.bilinear_cm(*front_xy)
    if sides is None or z_front is None:
        return None
    p_left, p_right = sides
    p_front = _back_project(camera, front_xy[0], front_xy[1], z_front * 10.0)
    width = float(np.linalg.norm(p_right - p_left))
    if width <= 0:
        return None
    normal3 = _plane_normal_3pt(p_left, p_right, p_front)
    if normal3 is None:
        return None

    z_side_mm = 0.5 * (p_left[2] + p_right[2])
    px_mm = camera.fx / z_side_mm
    path = tape_path(left_xy, right_xy, front_xy, jaw, PATH_STEP_MM * px_mm)
    inset_px = nw.EDGE_INSET_MM * px_mm
    inner = [
        (x, y) for x, y in path if left_xy[0] + inset_px <= x <= right_xy[0] - inset_px
    ]
    samples, invalid, front_sampled = [], 0, False
    for x, y in inner:
        z = smooth.bilinear_cm(x, y)
        if z is None:
            invalid += 1
            continue
        samples.append(_back_project(camera, x, y, z * 10.0))
        front_sampled |= x == front_xy[0] and y == front_xy[1]
    invalid_fraction = invalid / len(inner) if inner else 1.0
    path_3d = [p_left, *samples, p_right]
    arc = None
    if invalid == 0 and samples:
        arc = float(sum(np.linalg.norm(b - a) for a, b in itertools.pairwise(path_3d)))

    # The fit cloud: the path samples, which include the front point itself
    # (tape_path puts a vertex there); added separately only if its sample
    # had no depth, so it is never counted twice.
    cloud = list(samples)
    if end_drop > 0 and cloud:
        k = round(end_drop * len(cloud))
        cloud = cloud[k : len(cloud) - k] if len(cloud) > 2 * k else []
    if (not front_sampled or end_drop > 0) and not any(
        np.allclose(p, p_front) for p in cloud
    ):
        cloud.append(p_front)

    # The plane the profile is expressed in: least squares through the fit
    # cloud (module docstring, step 3 "Plane"); the 3-point plane when there
    # are too few samples.
    normal = normal3
    fitted_plane = len(cloud) >= MIN_FIT_SAMPLES
    if fitted_plane:
        centroid = np.mean(cloud, axis=0)
        normal = np.linalg.svd(np.array(cloud) - centroid)[2][2]
    if normal[1] < 0:
        normal = -normal
    # In-plane frame: e1 along the side chord (projected), e2 towards the
    # front point; origin at the (projected) chord midpoint.
    mid = 0.5 * (p_left + p_right)
    if fitted_plane:
        mid = mid - np.dot(mid - centroid, normal) * normal
    chord = (p_right - p_left) - np.dot(p_right - p_left, normal) * normal
    e1 = chord / np.linalg.norm(chord)
    e2 = np.cross(normal, e1)
    if np.dot(p_front - mid, e2) < 0:
        e2 = -e2
    sagitta = float(np.dot(p_front - mid, e2))
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal[1])))))
    tilt3 = math.degrees(math.acos(min(1.0, abs(float(normal3[1])))))
    drop = math.degrees(math.atan2(float(e2[1]), -float(e2[2])))
    profile = [(float(np.dot(p - mid, e1)), float(np.dot(p - mid, e2))) for p in cloud]
    off_plane = (
        float(np.sqrt(np.mean([np.dot(p - mid, normal) ** 2 for p in samples])))
        if samples
        else None
    )

    circ = {
        MODEL_CIRCLE: math.pi * width,
        MODEL_SAGITTA_ELLIPSE: ellipse_circumference(width / 2.0, max(sagitta, 1e-6)),
    }
    circle = ellipse_b = rms = span = None
    if len(cloud) >= MIN_FIT_SAMPLES:
        circle = fit_circle_2d(profile, max_radius=MAX_FIT_RADIUS_FACTOR * width / 2.0)
        if circle is not None:
            circ[MODEL_FITTED_CIRCLE] = 2.0 * math.pi * circle[2]
            rms, span = circle_fit_stats(profile, circle)
        ellipse_b = fit_ellipse_b(profile, width / 2.0)
        if ellipse_b is not None:
            circ[MODEL_FITTED_ELLIPSE] = ellipse_circumference(width / 2.0, ellipse_b)
    if arc is not None:
        radius = arc_circle_radius(width, arc)
        if radius is not None:
            circ[MODEL_ARC_CIRCLE] = 2.0 * math.pi * radius
    return _Models(
        width,
        sagitta,
        arc,
        tilt,
        tilt3,
        drop,
        tuple(float(v) for v in normal),
        tuple(float(v) for v in mid),
        off_plane,
        circle,
        rms,
        span,
        ellipse_b,
        circ,
        profile,
        [tuple(float(v) for v in p) for p in path_3d],
        invalid_fraction,
    )


def _sensitivity(
    depth, smooth, camera, left_xy, right_xy, front_xy, jaw, z_ref_cm, base
):
    """``(worst, detail)``: ``detail[perturbation][model]`` = signed relative
    change (None when the perturbed model could not be computed), ``worst
    [model]`` = the largest ``|change|`` over the perturbations (None when
    none could be computed). Perturbations: the side columns moved to make
    the pixel width +-:data:`SENSITIVITY_WIDTH_FRACTION`, the front point
    moved +-:data:`SENSITIVITY_FRONT_MM` along the midline, and
    :data:`SENSITIVITY_END_DROP` of the samples dropped at each end."""
    delta = 0.5 * SENSITIVITY_WIDTH_FRACTION * (right_xy[0] - left_xy[0])
    z_front = smooth.bilinear_cm(*front_xy)
    front_px = SENSITIVITY_FRONT_MM * camera.fy / (z_front * 10.0)
    pct = f"{SENSITIVITY_WIDTH_FRACTION:.0%}"
    front_mm = f"{SENSITIVITY_FRONT_MM:.0f}mm"
    wider = ((left_xy[0] - delta, left_xy[1]), (right_xy[0] + delta, right_xy[1]))
    narrower = ((left_xy[0] + delta, left_xy[1]), (right_xy[0] - delta, right_xy[1]))
    lower = (front_xy[0], front_xy[1] + front_px)
    higher = (front_xy[0], front_xy[1] - front_px)
    # name: (left_xy, right_xy, front_xy, end_drop)
    variants = {
        f"side-width+{pct}": (*wider, front_xy, 0.0),
        f"side-width-{pct}": (*narrower, front_xy, 0.0),
        f"front+{front_mm}": (left_xy, right_xy, lower, 0.0),
        f"front-{front_mm}": (left_xy, right_xy, higher, 0.0),
        f"ends-{SENSITIVITY_END_DROP:.0%}": (
            left_xy,
            right_xy,
            front_xy,
            SENSITIVITY_END_DROP,
        ),
    }
    detail: dict[str, dict[str, float | None]] = {}
    for name, (left, right, front, end_drop) in variants.items():
        moved = _models(
            depth, smooth, camera, left, right, front, jaw, z_ref_cm, end_drop=end_drop
        )
        detail[name] = {}
        for model, value in base.circumferences.items():
            other = None if moved is None else moved.circumferences.get(model)
            detail[name][model] = None if other is None else other / value - 1.0
    worst: dict[str, float | None] = {}
    for model in base.circumferences:
        changes = [abs(d[model]) for d in detail.values() if d[model] is not None]
        worst[model] = max(changes) if changes else None
    return worst, detail


# -- public ---------------------------------------------------------------------------


def compute_neck_sag(
    portrait,
    width_result,
    landmarks,
    *,
    camera,
    depth=None,
    skin=None,
) -> NeckSagResult:
    """Tape path and tilted-plane circumference models for an ``"ok"``
    :class:`~portrait_analyser.neck_width.NeckWidthResult` (its ``left_x`` /
    ``right_x`` at ``row_y`` are the side points).

    :param landmarks: FaceMesh landmarks (photo px)
    :param camera: the intrinsics the width was measured with
    :param depth: the 3x3-median float map the width used (default
        ``portrait.depth.median_filtered(3)``)
    :param skin: photo-size float skin matte (default from the portrait)
    :returns: a :class:`NeckSagResult`; ``quality_reasons`` are the tape's
        own (the width's are in the width result; the combination is
        ``NeckWidthResult.circumference_quality``)
    """
    if width_result.status != nw.STATUS_OK:
        return NeckSagResult(status=SAG_STATUS_FAILED, message="no neck width")
    photo_size = portrait.photo.size
    if depth is None:
        depth = portrait.depth.median_filtered(3)
    smooth = depth.integration_map(camera)
    if skin is None:
        skin = nw._skin_array(portrait.skinmap, photo_size)
    hairmap = getattr(portrait, "hairmap", None)
    hair = None if hairmap is None else nw._skin_array(hairmap, photo_size)
    jaw = nw.lower_jaw_contour(landmarks)

    left_xy = (width_result.left_x, width_result.row_y)
    right_xy = (width_result.right_x, width_result.row_y)
    chin_x, chin_y = (float(v) for v in landmarks[nw.FACE_MESH_CHIN_INDEX][:2])
    front_x = chin_x
    if jaw is not None:
        front_x = 0.5 * (jaw.left_angle[0] + jaw.right_angle[0])
    z_chin = depth.distance_cm(chin_x, chin_y, nw.DEPTH_WINDOW_RADIUS)
    if z_chin is None:
        return NeckSagResult(status=SAG_STATUS_FAILED, message="no depth at the chin")
    if not left_xy[0] < front_x < right_xy[0]:
        return NeckSagResult(
            status=SAG_STATUS_FAILED,
            left_xy=left_xy,
            right_xy=right_xy,
            message="the facial midline is not between the neck sides (head turned?)",
        )
    bottom = width_result.band[1] if width_result.band else photo_size[1] - 1
    front = find_front_point(
        skin, hair, smooth, camera, front_x, chin_y, bottom, z_chin
    )
    front_xy = (float(front_x), front.y)

    result = NeckSagResult(
        status=SAG_STATUS_FAILED,
        left_xy=left_xy,
        right_xy=right_xy,
        front_xy=front_xy,
        front_source=front.source,
        beard_border_y=front.beard_border_y,
        chin_end_y=front.chin_end_y,
    )
    reasons: list[str] = []

    def flag(text):
        reasons.append(text)
        result.warnings.append(text)

    if front.source == FRONT_SOURCE_INFERRED:
        flag("front point inferred (no beard/chin boundary found)")
    for text in front.flags:
        flag(text)
    if front.y <= width_result.row_y:
        flag("front point not below the side points (tape plane tilts upward)")

    z_ref = width_result.left_depth_cm or z_chin
    base = _models(depth, smooth, camera, left_xy, right_xy, front_xy, jaw, z_ref)
    if base is None:
        result.message = "no depth at the side points or the front point"
        return result
    step = max(1.0, PATH_STEP_MM * camera.fx / base.path_3d[0][2])
    result.path = tape_path(left_xy, right_xy, front_xy, jaw, step)
    result.width_mm = base.width
    result.sagitta_mm = base.sagitta
    result.front_arc_mm = base.arc
    result.plane_tilt_deg = base.tilt
    result.plane_tilt_3pt_deg = base.tilt3
    result.plane_normal = base.normal
    result.plane_origin_mm = base.origin
    result.front_drop_deg = base.drop
    result.off_plane_rms_mm = base.off_plane
    if base.circle is not None:
        result.circle_centre_mm = (base.circle[0], base.circle[1])
        result.circle_radius_mm = base.circle[2]
        result.circle_rms_mm = base.circle_rms
        result.circle_span_deg = base.circle_span
    result.ellipse_b_mm = base.ellipse_b
    result.circumferences_mm = dict(base.circumferences)
    result.circumference_prior_ellipse_mm = nw.circumference_ellipse_range(base.width)
    result.profile_mm = base.profile
    result.path_3d_mm = base.path_3d
    if base.invalid_fraction > MAX_INVALID_SAMPLE_FRACTION:
        flag(f"{base.invalid_fraction:.0%} of the tape path has no depth")
    if base.sagitta <= 0:
        flag("front point not in front of the side points")
    result.sensitivity, result.sensitivity_detail = _sensitivity(
        depth, smooth, camera, left_xy, right_xy, front_xy, jaw, z_ref, base
    )
    model = HEADLINE_MODEL if HEADLINE_MODEL in base.circumferences else MODEL_CIRCLE
    if model != HEADLINE_MODEL:
        flag(
            f"no usable {HEADLINE_MODEL} (too few tape samples or degenerate); headline is pi*W"
        )
    else:
        _circle_quality(result, base, flag)
    result.headline_model = model
    result.headline_mm = base.circumferences[model]
    result.quality_reasons = reasons
    result.quality = nw.QUALITY_LOW if reasons else nw.QUALITY_GOOD
    result.status = SAG_STATUS_OK
    return result


def _circle_quality(result: NeckSagResult, base: _Models, flag):
    """The fitted-circle quality gate (module docstring, "Quality gate")."""
    fitted = base.circumferences[MODEL_FITTED_CIRCLE]
    pi_w = base.circumferences[MODEL_CIRCLE]
    disagreement = fitted / pi_w - 1.0
    if abs(disagreement) > MAX_CIRCLE_VS_WIDTH:
        flag(
            f"fitted circle and pi*W disagree by {disagreement:+.0%} (neck section "
            "not round: the circle reads high on a flat front, low on a deep one)"
        )
    ratio = base.circle[2] / (base.width / 2.0)
    low, high = CIRCLE_RADIUS_RATIO_RANGE
    if not low <= ratio <= high:
        flag(f"fitted radius {ratio:.2f}x the half width (outside {low}-{high})")
    if base.circle_span is not None and base.circle_span < MIN_CIRCLE_SPAN_DEG:
        flag(
            f"the tape samples span only {base.circle_span:.0f} deg of the fitted circle "
            f"(< {MIN_CIRCLE_SPAN_DEG:.0f})"
        )
    if base.circle_rms is not None and base.circle_rms > MAX_CIRCLE_RMS_MM:
        flag(
            f"tape samples {base.circle_rms:.1f} mm RMS off the fitted circle "
            f"(> {MAX_CIRCLE_RMS_MM:.1f} mm)"
        )
    worst = result.sensitivity.get(MODEL_FITTED_CIRCLE)
    if worst is not None and worst > MAX_HEADLINE_SENSITIVITY:
        flag(
            f"fitted circle moves by up to {worst:.0%} with the key points "
            f"(> {MAX_HEADLINE_SENSITIVITY:.0%}; see sensitivity_detail)"
        )
