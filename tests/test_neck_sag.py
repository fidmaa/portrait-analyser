"""Tape path, front point and tilted-plane circumference (synthetic only).

The synthetic neck is an elliptic cylinder (semi-axes A_MM lateral, B_MM
front-back) whose axis leans forward by TILT_DEG, seen by a pinhole camera:
float depth from exact ray/cylinder intersection, the skin matte from its
silhouette. Above the chin row sits a flat "face"; a "beard" is a region
under the chin with no skin whose depth stands HAIR_MM in front of the
neck. The FaceMesh lower-jaw contour is a parabola from the jaw angles
through the chin.
"""

import json
import logging
import math
import sys

import numpy as np
import pytest
from PIL import Image

from portrait_analyser import neck_sag, neck_width
from portrait_analyser.apple_vision import BodyPose
from portrait_analyser.ios import IOSPortrait
from portrait_analyser.neck_sag import (
    FRONT_SOURCE_BEARD,
    FRONT_SOURCE_CHIN,
    FRONT_SOURCE_INFERRED,
    arc_circle_radius,
    fit_circle_2d,
    fit_ellipse_b,
    tape_path,
)
from portrait_analyser.neck_width import STATUS_OK, measure_neck_width

PHOTO_W, PHOTO_H = 1200, 1600
DEPTH_SCALE = 2
FX = 1100.0
CX, CY = PHOTO_W / 2, PHOTO_H / 2
A_MM, B_MM = 64.0, 60.0
TILT_DEG = 15.0  # neck axis leaning forward (top nearer the camera)
AXIS_Z_MM = 510.0  # axis depth at the chin row
FACE_Z_MM = 400.0
BACKGROUND_Z_MM = 1500.0
CHIN_Y = 700.0
NECK_JOINT_Y = 1150.0
JAW_HALF_WIDTH_PX = 150.0
JAW_RISE_PX = 60.0
HAIR_MM = 8.0


def _axis():
    """Axis point at the chin row (camera mm) and unit direction (down)."""
    t = math.radians(TILT_DEG)
    y_chin_mm = (CHIN_Y - CY) * AXIS_Z_MM / FX
    return np.array([0.0, y_chin_mm, AXIS_Z_MM]), np.array([0.0, math.cos(t), math.sin(t)])


def _frame():
    _, d = _axis()
    ex = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(d, ex)  # perpendicular to the axis in the y-z plane
    if e2[2] > 0:
        e2 = -e2  # towards the camera
    return ex, e2


def ray_depth(us, vs):
    """Camera z (mm) of the near cylinder surface for photo px (NaN = miss)."""
    p0, _ = _axis()
    ex, e2 = _frame()
    rx = (us - CX) / FX
    ry = (vs - CY) / FX
    rays = np.stack([rx, ry, np.ones_like(rx)], axis=-1)
    s1r, s1c = rays @ ex, p0 @ ex
    s2r, s2c = rays @ e2, p0 @ e2
    # ((t s1r - s1c)/A)^2 + ((t s2r - s2c)/B)^2 = 1
    qa = (s1r / A_MM) ** 2 + (s2r / B_MM) ** 2
    qb = -2 * (s1r * s1c / A_MM**2 + s2r * s2c / B_MM**2)
    qc = (s1c / A_MM) ** 2 + (s2c / B_MM) ** 2 - 1
    disc = qb * qb - 4 * qa * qc
    with np.errstate(invalid="ignore"):
        t = (-qb - np.sqrt(disc)) / (2 * qa)
    return np.where(disc >= 0, t, np.nan)


def section_perimeter(normal, origin, n=4000):
    """Perimeter (mm) of the cylinder cut by the plane (normal, origin)."""
    p0, d = _axis()
    ex, e2 = _frame()
    normal, origin = np.asarray(normal), np.asarray(origin)
    th = np.linspace(0, 2 * np.pi, n + 1)
    base = p0 + A_MM * np.cos(th)[:, None] * ex + B_MM * np.sin(th)[:, None] * e2
    h = -((base - origin) @ normal) / (d @ normal)
    pts = base + h[:, None] * d
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _inside_jaw(xs, ys):
    """Inside the synthetic lower-jaw parabola (the face below the jaw angles)."""
    dx = np.abs(xs - CX) / JAW_HALF_WIDTH_PX
    return (dx <= 1.0) & (ys <= CHIN_Y - JAW_RISE_PX * dx**2)


def make_portrait(
    *,
    beard_mm=25.0,
    beard_depth_mm=HAIR_MM,
    beard_in_hair_matte=False,
    chin_step=True,
    flare_y=None,
    neck_under_jaw=False,
):
    """Synthetic portrait. ``beard_mm``: height of the beard under the chin
    (0 = none); ``beard_depth_mm``: how far its surface stands in front of
    the skin (0 = a matte dropout only, e.g. a shadow); with
    ``beard_in_hair_matte`` the beard is skin in the skin matte and only
    the hair matte marks it."""
    depth_w, depth_h = PHOTO_W // DEPTH_SCALE, PHOTO_H // DEPTH_SCALE
    us = np.arange(depth_w) * (PHOTO_W - 1) / (depth_w - 1)
    vs = np.arange(depth_h) * (PHOTO_H - 1) / (depth_h - 1)
    uu, vv = np.meshgrid(us, vs)
    neck = ray_depth(uu, vv)
    depth = np.where(np.isfinite(neck), neck, BACKGROUND_Z_MM)
    above = vv < CHIN_Y
    face = np.abs(uu - CX) <= 250
    if chin_step:
        depth[above & face] = FACE_Z_MM
    else:
        # No chin: the neck surface simply continues up under the face.
        depth[above & face] = np.where(np.isfinite(neck), neck, FACE_Z_MM)[above & face]

    xs, ys = np.meshgrid(np.arange(PHOTO_W, dtype=float), np.arange(PHOTO_H, dtype=float))
    neck_px = ray_depth(xs, ys)
    skin = np.where(np.isfinite(neck_px) & (ys >= CHIN_Y), 255, 0).astype(np.uint8)
    if neck_under_jaw:
        # A narrow face: the neck sides show from the jaw angles down (the
        # jaw's lower border runs inside them), as on a clean-shaven face.
        top = CHIN_Y - JAW_RISE_PX
        under = (vv >= top) & above & ~_inside_jaw(uu, vv) & np.isfinite(neck)
        depth[under] = neck[under]
        depth[above & ~_inside_jaw(uu, vv) & (vv >= top) & ~np.isfinite(neck)] = BACKGROUND_Z_MM
        skin[(ys >= top) & (ys < CHIN_Y) & np.isfinite(neck_px)] = 255
        skin[_inside_jaw(xs, ys)] = 255
        skin[(ys < top) & (np.abs(xs - CX) <= JAW_HALF_WIDTH_PX)] = 255
    else:
        skin[(ys < CHIN_Y) & (np.abs(xs - CX) <= 250)] = 255

    hair = None
    if beard_mm:
        beard_px = beard_mm * FX / 450.0
        beard = ((xs - CX) / 115.0) ** 2 + ((ys - CHIN_Y) / beard_px) ** 2 <= 1.0
        if beard_in_hair_matte:
            hair = np.where(beard, 255, 0).astype(np.uint8)
        else:
            skin[beard] = 0
        dbeard = ((uu - CX) / 115.0) ** 2 + ((vv - CHIN_Y) / beard_px) ** 2 <= 1.0
        depth[dbeard & ~above] = depth[dbeard & ~above] - beard_depth_mm

    if flare_y is not None:
        # Below flare_y each side flares 10 mm outwards (neck base / clavicle),
        # at the neck-edge depth.
        for y in range(int(flare_y), PHOTO_H):
            row = np.flatnonzero(skin[y])
            if row.size:
                extra = round(10.0 * FX / 500.0)
                skin[y, max(row[0] - extra, 0) : row[0]] = 255
                skin[y, row[-1] + 1 : row[-1] + 1 + extra] = 255
        flare_rows = vv >= flare_y
        for i in np.flatnonzero(flare_rows[:, 0]):
            finite = np.flatnonzero(np.isfinite(neck[i]))
            if finite.size:
                extra = round(10.0 * FX / 500.0 / DEPTH_SCALE) + 1
                z_edge = float(neck[i, finite[0]])
                depth[i, max(finite[0] - extra, 0) : finite[0]] = z_edge + 5.0
                depth[i, finite[-1] + 1 : finite[-1] + 1 + extra] = z_edge + 5.0

    return IOSPortrait(
        photo=Image.new("RGB", (PHOTO_W, PHOTO_H), (90, 90, 90)),
        skinmap=Image.fromarray(skin, "L"),
        hairmap=None if hair is None else Image.fromarray(hair, "L"),
        depth_m=(depth / 1000.0).astype(np.float32),
        depth_accuracy="absolute",
        depth_plausible=True,
        focal_length_px=(FX, FX),
        principal_point_px=(CX, CY),
    )


def face_mesh():
    landmarks = [(CX, CHIN_Y - 300.0)] * 478
    landmarks[152] = (CX, CHIN_Y)
    landmarks[1] = (CX, CHIN_Y - 300.0)
    left = (172, 136, 150, 149, 176, 148)
    right = (397, 365, 379, 378, 400, 377)
    for i, (li, ri) in enumerate(zip(left, right)):
        dx = JAW_HALF_WIDTH_PX * (1 - i / len(left))
        y = CHIN_Y - JAW_RISE_PX * (dx / JAW_HALF_WIDTH_PX) ** 2
        landmarks[li] = (CX - dx, y)
        landmarks[ri] = (CX + dx, y)
    return landmarks


def body_pose():
    return BodyPose(joints={"neck_1_joint": (CX, NECK_JOINT_Y, 0.7)})


def measure(portrait=None, **kwargs):
    portrait = portrait if portrait is not None else make_portrait(**kwargs)
    return measure_neck_width(portrait, face_mesh=face_mesh(), body_pose=body_pose())


# -- helpers -------------------------------------------------------------------------


def test_fit_circle_recovers_a_circle():
    t = np.linspace(0.2, 2.9, 40)
    cu, cv, r = fit_circle_2d(np.column_stack([3 + 50 * np.cos(t), -7 + 50 * np.sin(t)]))
    assert (cu, cv, r) == pytest.approx((3, -7, 50), abs=1e-6)
    assert fit_circle_2d([(0, 0), (1, 1)]) is None


def test_fit_ellipse_b_recovers_b():
    t = np.linspace(0.2, math.pi - 0.2, 40)
    pts = np.column_stack([60 * np.cos(t), 45 * np.sin(t)])
    assert fit_ellipse_b(pts, 60.0) == pytest.approx(45.0, rel=1e-6)


def test_arc_circle_of_a_semicircle():
    assert arc_circle_radius(100.0, 50.0 * math.pi) == pytest.approx(50.0, rel=1e-6)
    assert arc_circle_radius(100.0, 99.0) is None


def test_tape_path_passes_through_the_three_points_and_follows_the_jaw():
    jaw = neck_width.lower_jaw_contour(face_mesh())
    path = tape_path((400.0, 760.0), (800.0, 770.0), (CX, 900.0), jaw, 5.0)
    xs = [p[0] for p in path]
    assert path[0] == pytest.approx((400.0, 760.0))
    assert path[-1] == pytest.approx((800.0, 770.0))
    assert (CX, 900.0) in [(pytest.approx(x), pytest.approx(y)) for x, y in path]
    assert xs == sorted(xs)
    assert max(yy for _, yy in path) == pytest.approx(900.0, abs=1.0)


def test_tape_path_without_jaw_is_a_parabola():
    path = tape_path((0.0, 0.0), (100.0, 0.0), (50.0, 20.0), None, 10.0)
    ys = {round(x): y for x, y in path}
    assert ys[50] == pytest.approx(20.0)
    assert ys[20] == pytest.approx(20.0 * 4 * 0.2 * 0.8)


# -- the synthetic neck ------------------------------------------------------------


def _check_side_points(result):
    """Side points on the first clean rows below the chin, at the silhouette."""
    assert result.status == STATUS_OK, result.message
    rows_below = (result.row_y - CHIN_Y) / (FX / 450.0)  # mm, roughly
    assert neck_width.CHIN_CLEARANCE_MM <= rows_below < 25.0
    z = ray_depth(np.arange(PHOTO_W, dtype=float), np.full(PHOTO_W, result.row_y))
    cols = np.flatnonzero(np.isfinite(z))
    assert result.left_x == pytest.approx(cols[0], abs=3)
    assert result.right_x == pytest.approx(cols[-1] + 1, abs=3)


def test_beard_front_point_path_and_circumference():
    result = measure()
    _check_side_points(result)
    sag = result.sag
    assert sag.status == "ok", sag.message
    assert sag.front_source == FRONT_SOURCE_BEARD
    beard_bottom = CHIN_Y + 25.0 * FX / 450.0
    assert sag.beard_border_y == pytest.approx(beard_bottom, abs=6)
    assert sag.front_xy[1] > beard_bottom
    assert sag.front_xy[0] == pytest.approx(CX)
    # The path starts/ends at the side points and dips through the front.
    assert sag.path[0] == pytest.approx((result.left_x, result.row_y))
    assert sag.path[-1] == pytest.approx((result.right_x, result.row_y))
    assert max(y for _, y in sag.path) == pytest.approx(sag.front_xy[1], abs=0.5)
    # A tilted plane: the front is ~5 cm below the sides.
    assert 20.0 < sag.plane_tilt_deg < 60.0
    assert sag.off_plane_rms_mm < 3.0
    truth = section_perimeter(sag.plane_normal, sag.plane_origin_mm)
    assert sag.headline_model == "fitted_circle"
    assert result.circumference_mm == sag.headline_mm
    assert result.circumference_model == "fitted_circle"
    assert sag.headline_mm == pytest.approx(truth, rel=0.03)
    # Insensitive to the side-point placement (+-3 % pixel width) ...
    for name in ("side-width+3%", "side-width-3%"):
        assert abs(sag.sensitivity_detail[name]["fitted_circle"]) < 0.02
    # ... less so to the front point and the arc ends; all reported.
    assert set(sag.sensitivity_detail) == {
        "side-width+3%",
        "side-width-3%",
        "front+5mm",
        "front-5mm",
        "ends-10%",
    }
    assert sag.sensitivity["fitted_circle"] == max(
        abs(d["fitted_circle"]) for d in sag.sensitivity_detail.values()
    )
    json.dumps(sag.sensitivity_detail)  # no inf / NaN
    assert sag.quality == "good", sag.quality_reasons
    assert result.circumference_quality == "good"
    for model in ("circle", "sagitta_ellipse", "fitted_ellipse", "arc_circle"):
        assert model in sag.circumferences_mm
    assert sag.circumference_prior_ellipse_mm == neck_width.circumference_ellipse_range(
        sag.width_mm
    )


def test_perturbed_side_points_give_the_same_headline():
    portrait = make_portrait()
    result = measure(portrait)
    width_px = result.right_x - result.left_x
    values = []
    for f in (-0.015, 0.0, 0.015):
        moved = neck_width.NeckWidthResult(**result.__dict__)
        moved.left_x = result.left_x - f * width_px
        moved.right_x = result.right_x + f * width_px
        sag = neck_sag.compute_neck_sag(portrait, moved, face_mesh(), camera=portrait.camera)
        values.append(sag.headline_mm)
    base = values[1]
    assert all(abs(v / base - 1) < 0.02 for v in values)


def test_no_beard_front_point_at_the_chin_end():
    result = measure(beard_mm=0.0)
    sag = result.sag
    assert sag.status == "ok"
    assert sag.front_source == FRONT_SOURCE_CHIN
    assert sag.beard_border_y is None
    assert CHIN_Y < sag.front_xy[1] < CHIN_Y + 15.0 * FX / 450.0
    truth = section_perimeter(sag.plane_normal, sag.plane_origin_mm)
    assert sag.headline_mm == pytest.approx(truth, rel=0.03)


def test_no_beard_no_chin_front_point_is_inferred_low_quality():
    result = measure(beard_mm=0.0, chin_step=False)
    sag = result.sag
    assert sag.status == "ok"
    assert sag.front_source == FRONT_SOURCE_INFERRED
    assert sag.quality == "low"
    assert "front point inferred (no beard/chin boundary found)" in sag.quality_reasons
    assert any("front point inferred" in w for w in sag.warnings)


def test_decolletage_flare_rows_are_rejected():
    flare_y = CHIN_Y + 60.0
    result = measure(flare_y=flare_y)
    assert result.status == STATUS_OK
    assert result.row_y < flare_y
    # (Further down the synthetic neck recedes and narrows under the constant flare.)
    flared = [r for r in result.rows if flare_y + 10 < r.y < flare_y + 60]
    assert flared and all("flare" in r.reject_codes for r in flared)
    assert "flare" in result.reject_counts


def test_neck_sides_under_the_jaw_angles_above_the_chin():
    """Rows start just below the jaw angles; above the chin a row is a neck
    row once its edges lie outside the lower-jaw contour."""
    result = measure(neck_under_jaw=True, beard_mm=0.0)
    assert result.status == STATUS_OK, result.message
    assert CHIN_Y - JAW_RISE_PX < result.row_y < CHIN_Y
    # Signed: the side row is above the chin landmark.
    assert result.height_below_chin_mm < 0
    assert result.search_top_y < CHIN_Y - JAW_RISE_PX + 10
    jaw = neck_width.lower_jaw_contour(face_mesh())
    left_c, right_c = jaw.contour_x(result.row_y)
    assert result.left_x < left_c and result.right_x > right_c
    # The chin's own outline zone below the chin landmark is never a neck row.
    portrait = make_portrait(neck_under_jaw=True, beard_mm=0.0)
    z_chin = portrait.depth.median_filtered(3).distance_cm(CX, CHIN_Y, 2)
    clearance = neck_width.CHIN_CLEARANCE_MM * FX / (z_chin * 10.0)
    assert not [r for r in result.rows if CHIN_Y <= r.y < CHIN_Y + clearance - 1]


def test_jaw_line_rows_are_not_neck_rows():
    """Above the chin, a skin span bounded by the jaw contour itself (face
    wider than the neck, no neck showing beside it) is skipped."""
    portrait = make_portrait(beard_mm=0.0)
    skin = np.asarray(portrait.skinmap).copy()
    xs, ys = np.meshgrid(np.arange(PHOTO_W, dtype=float), np.arange(PHOTO_H, dtype=float))
    rows = ys < CHIN_Y
    skin[rows] = 0
    skin[rows & _inside_jaw(xs, ys)] = 255
    portrait.skinmap = Image.fromarray(skin, "L")
    result = measure(portrait)
    assert result.status == STATUS_OK
    assert result.row_y > CHIN_Y
    # (A row within the 9-row matte average of the chin can mix jaw and
    # neck; it is rejected, never measured.)
    assert all(not r.clean for r in result.rows if r.y < CHIN_Y)


def test_sag_is_only_for_ok_widths():
    result = neck_width.NeckWidthResult(status="edges-occluded")
    sag = neck_sag.compute_neck_sag(make_portrait(), result, face_mesh(), camera=None)
    assert sag.status == "failed" and sag.message == "no neck width"


def test_manual_edges_carry_no_sag():
    portrait = make_portrait()
    result = neck_width.neck_width_from_edges(portrait, (470, 780), (730, 780))
    assert result.sag is None and result.circumference_mm is None


# -- review follow-ups -----------------------------------------------------------


THIS = sys.modules[__name__]


@pytest.mark.parametrize(
    ("front_back_mm", "low", "high", "flagged"),
    [
        # Flat front: the circle reads high, and disagrees with pi*W -> low.
        (48.0, 0.15, 0.30, True),
        # Near-circular section: within 5 %.
        (56.0, -0.05, 0.05, False),
        # Deep section: the circle reads low; nothing visible flags it.
        (72.0, -0.22, -0.10, False),
    ],
)
def test_known_shape_bias_of_the_headline(monkeypatch, front_back_mm, low, high, flagged):
    monkeypatch.setattr(THIS, "B_MM", front_back_mm)
    sag = measure().sag
    truth = section_perimeter(sag.plane_normal, sag.plane_origin_mm)
    assert low < sag.headline_mm / truth - 1 < high
    disagree = any("disagree" in r for r in sag.quality_reasons)
    assert disagree is flagged
    assert (sag.quality == "low") is flagged


def test_circumference_quality_combines_width_and_tape_reasons(monkeypatch):
    monkeypatch.setattr(THIS, "B_MM", 48.0)
    result = measure()
    assert result.quality == "good"  # the width itself is fine
    assert result.circumference_quality == "low"
    assert result.circumference_quality_reasons == [
        *result.quality_reasons,
        *result.sag.quality_reasons,
    ]


def test_circle_fit_refines_to_the_geometric_fit_and_rejects_lines():
    rng = np.random.default_rng(1)
    t = np.linspace(0.4, 2.7, 60)
    pts = np.column_stack([60 * np.cos(t), 60 * np.sin(t)]) + rng.normal(0, 0.8, (60, 2))
    cu, cv, r = fit_circle_2d(pts)
    rms, span = neck_sag.circle_fit_stats(pts, (cu, cv, r))
    assert (
        r == pytest.approx(60.0, abs=1.0)
        and rms < 1.0
        and span == pytest.approx(math.degrees(2.3), abs=3)
    )
    line = np.column_stack([np.linspace(-50, 50, 20), np.full(20, 3.0)])
    assert fit_circle_2d(line + rng.normal(0, 0.01, line.shape), max_radius=200) is None


def test_front_point_outside_the_sides_is_refused():
    with pytest.raises(ValueError, match="not between the side points"):
        tape_path((400.0, 760.0), (800.0, 770.0), (850.0, 900.0), None, 5.0)
    portrait = make_portrait()
    result = measure(portrait)
    moved = neck_width.NeckWidthResult(**result.__dict__)
    moved.right_x = CX - 10
    moved.left_x = CX - 200
    sag = neck_sag.compute_neck_sag(portrait, moved, face_mesh(), camera=portrait.camera)
    assert sag.status == "failed" and "midline" in sag.message


def test_beard_in_the_hair_matte_only():
    sag = measure(beard_in_hair_matte=True).sag
    assert sag.front_source == FRONT_SOURCE_BEARD
    assert sag.beard_border_y == pytest.approx(CHIN_Y + 25.0 * FX / 450.0, abs=6)


def test_beard_without_lower_border_is_flagged():
    # The beard reaches past the 70 mm midline search window.
    sag = measure(beard_mm=85.0).sag
    assert sag.status == "ok"
    assert sag.beard_border_y is None
    assert sag.quality == "low"
    assert any("lower border not found" in r for r in sag.quality_reasons)


def test_false_beard_from_a_matte_dropout_moves_the_front_point_down():
    """A shadow under a clean-shaven chin that drops out of the skin matte
    (no depth change) looks like a beard: the front point goes to the end
    of the dropout + BEARD_CLEARANCE_MM instead of the chin end. Documented
    behaviour: the tape is lowered by that much, not rejected."""
    clean = measure(beard_mm=0.0).sag
    shadow = measure(beard_mm=15.0, beard_depth_mm=0.0).sag
    assert clean.front_source == FRONT_SOURCE_CHIN
    assert shadow.front_source == FRONT_SOURCE_BEARD
    dropout_end = CHIN_Y + 15.0 * FX / 450.0
    assert shadow.front_xy[1] > clean.front_xy[1]
    assert shadow.front_xy[1] == pytest.approx(
        dropout_end + neck_sag.BEARD_CLEARANCE_MM * FX / 400.0, abs=10
    )


def test_two_chin_falls_are_flagged():
    portrait = make_portrait(beard_mm=0.0)
    depth = portrait.depth_m.copy()
    rows, cols = depth.shape
    vs = np.arange(rows) * (PHOTO_H - 1) / (rows - 1)
    us = np.arange(cols) * (PHOTO_W - 1) / (cols - 1)
    # A skin fold 12 mm proud of the throat 15-30 mm below the chin.
    fold = (vs > CHIN_Y + 15 * FX / 450) & (vs < CHIN_Y + 30 * FX / 450)
    depth[np.ix_(fold, np.abs(us - CX) < 100)] -= 0.012
    portrait = IOSPortrait(
        photo=portrait.photo,
        skinmap=portrait.skinmap,
        depth_m=depth,
        depth_accuracy="absolute",
        depth_plausible=True,
        focal_length_px=(FX, FX),
        principal_point_px=(CX, CY),
    )
    sag = measure(portrait).sag
    assert any("depth falls below the chin" in r for r in sag.quality_reasons)
    assert sag.quality == "low"


def test_front_point_clamped_to_the_band_bottom():
    # Beard border ~29 px below the chin, + 5 mm clearance ~43 px: past a
    # band ending 40 px below the chin.
    pose = BodyPose(joints={"neck_1_joint": (CX, CHIN_Y + 40.0, 0.7)})
    portrait = make_portrait(beard_mm=12.0)
    result = measure_neck_width(portrait, face_mesh=face_mesh(), body_pose=pose)
    assert result.status == STATUS_OK, result.message
    sag = result.sag
    assert sag.front_source == FRONT_SOURCE_BEARD
    assert sag.front_xy[1] == pytest.approx(CHIN_Y + 40.0)
    assert any("clamped to the band bottom" in r for r in sag.quality_reasons)


def test_front_point_not_below_the_sides_is_flagged():
    portrait = make_portrait()
    result = measure(portrait)
    moved = neck_width.NeckWidthResult(**result.__dict__)
    moved.row_y = result.sag.front_xy[1] + 30.0
    sag = neck_sag.compute_neck_sag(portrait, moved, face_mesh(), camera=portrait.camera)
    assert "front point not below the side points (tape plane tilts upward)" in (
        sag.quality_reasons
    )


def test_tape_path_without_depth_is_flagged():
    portrait = make_portrait()
    depth = portrait.depth_m.copy()
    cols = depth.shape[1]
    us = np.arange(cols) * (PHOTO_W - 1) / (cols - 1)
    depth[:, (us > CX + 30) & (us < CX + 90)] = np.nan
    portrait = IOSPortrait(
        photo=portrait.photo,
        skinmap=portrait.skinmap,
        depth_m=depth,
        depth_accuracy="absolute",
        depth_plausible=True,
        focal_length_px=(FX, FX),
        principal_point_px=(CX, CY),
    )
    sag = measure(portrait).sag
    assert sag.status == "ok"
    assert sag.front_arc_mm is None
    assert any("of the tape path has no depth" in r for r in sag.quality_reasons)


def test_headline_falls_back_to_pi_w(monkeypatch):
    monkeypatch.setattr(neck_sag, "MIN_FIT_SAMPLES", 10**6)
    result = measure()
    sag = result.sag
    assert sag.headline_model == "circle"
    assert result.circumference_model == "circle"
    assert result.circumference_mm == pytest.approx(math.pi * sag.width_mm)
    assert any("headline is pi*W" in r for r in sag.quality_reasons)
    assert result.circumference_quality == "low"


def test_sag_failure_is_logged_and_the_width_stands(monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("tape exploded")

    monkeypatch.setattr(neck_sag, "compute_neck_sag", boom)
    with caplog.at_level(logging.ERROR):
        result = measure()
    assert result.status == STATUS_OK and result.width_mm is not None
    assert result.sag.status == "failed" and "tape exploded" in result.sag.message
    assert result.circumference_mm is None and result.circumference_quality is None
    assert any("no tape-plane circumference" in w for w in result.warnings)
    assert "neck sag / tape-plane computation failed" in caplog.text
