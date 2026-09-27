"""Tests for the width-based neck measurement (synthetic portraits only).

The synthetic portrait is a vertical cylinder "neck" of known radius under a
flat "face", seen by a pinhole camera: float depth (metres) from exact
ray/cylinder intersection, the skin matte from the cylinder's silhouette.
"""

import math
import re
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import portrait_analyser as pa
from portrait_analyser import apple_vision, neck_width
from portrait_analyser.apple_vision import BodyPose
from portrait_analyser.exceptions import AppleVisionError, AppleVisionUnavailable
from portrait_analyser.ios import IOSPortrait
from portrait_analyser.neck_width import (
    BAND_SOURCE_CHIN_OFFSET,
    BAND_SOURCE_MANUAL,
    BAND_SOURCE_VISION,
    STATUS_EDGES_OCCLUDED,
    STATUS_NO_CAMERA,
    STATUS_NO_FACE,
    STATUS_OK,
    circumference_circle,
    circumference_ellipse_range,
    measure_neck_width,
    neck_width_from_edges,
)

PHOTO_W, PHOTO_H = 1200, 1600
DEPTH_SCALE = 2  # photo px per depth px: ~0.9 mm at the neck (real files: ~1 mm)
FX = 1100.0  # pixels (the real 2771 px at 3024 wide, scaled to 1200)
CX, CY = PHOTO_W / 2, PHOTO_H / 2
NECK_RADIUS_MM = 65.0
NECK_AXIS_Z_MM = 510.0  # cylinder axis distance; front of the neck at 445 mm
FACE_Z_MM = 400.0
BACKGROUND_Z_MM = 1500.0
CHIN_Y = 700.0
NECK_JOINT_Y = 1150.0


def _ray_cylinder_depth(u):
    """Camera distance (mm) of the cylinder's near surface along column ``u``
    (photo px), or NaN outside the silhouette."""
    x = (np.asarray(u, dtype=float) - CX) / FX
    a = x * x + 1.0
    zc, r = NECK_AXIS_Z_MM, NECK_RADIUS_MM
    disc = zc * zc - a * (zc * zc - r * r)
    with np.errstate(invalid="ignore"):
        z = (zc - np.sqrt(disc)) / a
    return np.where(disc >= 0, z, np.nan)


def _silhouette_half_angle():
    return math.asin(NECK_RADIUS_MM / NECK_AXIS_Z_MM)


def _silhouette_columns():
    t = math.tan(_silhouette_half_angle())
    return CX - FX * t, CX + FX * t


def expected_method_width_mm():
    """What the method measures on an ideal cylinder: the silhouette columns,
    each at the depth read EDGE_INSET_MM inside it.

    On an ideal circular cylinder that inset depth is ~2 cm nearer than the
    tangent point (sqrt(2 R d)), so the method reads ~4.5 % below the
    diameter. Real TrueDepth maps are smooth at the silhouette and the width
    changes only ~1-1.5 % per mm of inset there (IMG_2389/IMG_2386).
    """
    left, right = _silhouette_columns()
    return (right - left) * _inset_depth_mm() / FX


def _inset_depth_mm():
    """Cylinder depth EDGE_INSET_MM inside the silhouette."""
    left, _ = _silhouette_columns()
    z = NECK_AXIS_Z_MM - NECK_RADIUS_MM
    for _ in range(3):
        inset_px = neck_width.EDGE_INSET_MM * FX / z
        z = float(_ray_cylinder_depth([left + inset_px])[0])
    return z


def make_portrait(*, collar=None, camera=True, face_width_px=500):
    """Synthetic capture-app portrait.

    :param collar: None, or ``(top_y, nearer_mm)``: from ``top_y`` down, a
        20 mm band just outside each neck edge sits ``nearer_mm`` nearer than
        the neck just inside the edge (a collar in front of the neck side).
    """
    depth_w, depth_h = PHOTO_W // DEPTH_SCALE, PHOTO_H // DEPTH_SCALE
    # Photo coordinates of each depth pixel (same mapping as FloatDepthMap).
    us = np.arange(depth_w) * (PHOTO_W - 1) / (depth_w - 1)
    vs = np.arange(depth_h) * (PHOTO_H - 1) / (depth_h - 1)
    neck_row = _ray_cylinder_depth(us)
    depth = np.full((depth_h, depth_w), BACKGROUND_Z_MM)
    below = vs >= CHIN_Y
    depth[below] = np.where(np.isfinite(neck_row), neck_row, BACKGROUND_Z_MM)
    face_cols = np.abs(us - CX) <= face_width_px / 2
    depth[np.ix_(~below, face_cols)] = FACE_Z_MM

    left, right = _silhouette_columns()
    if collar is not None:
        top_y, nearer_mm = collar
        edge_z = _inset_depth_mm()
        band_px = 20.0 * FX / edge_z
        rows = vs >= top_y
        cols = ((us < left) & (us > left - band_px)) | ((us > right) & (us < right + band_px))
        depth[np.ix_(rows, cols)] = edge_z - nearer_mm

    skin = np.zeros((PHOTO_H, PHOTO_W), dtype=np.uint8)
    xs = np.arange(PHOTO_W)
    skin[int(CHIN_Y) :, (xs >= left) & (xs <= right)] = 255
    skin[: int(CHIN_Y), np.abs(xs - CX) <= face_width_px / 2] = 255

    return IOSPortrait(
        photo=Image.new("RGB", (PHOTO_W, PHOTO_H), (90, 90, 90)),
        skinmap=Image.fromarray(skin, "L"),
        depth_m=(depth / 1000.0).astype(np.float32),
        depth_accuracy="absolute",
        depth_plausible=True,
        focal_length_px=(FX, FX) if camera else None,
        principal_point_px=(CX, CY) if camera else None,
    )


JAW_HALF_WIDTH_PX = 150.0  # neck silhouette is ~283 px wide


def face_mesh(roll_deg=0.0):
    landmarks = [(CX, CHIN_Y - 300.0)] * 478
    landmarks[152] = (CX, CHIN_Y)
    landmarks[1] = (CX - 300.0 * math.tan(math.radians(roll_deg)), CHIN_Y - 300.0)  # nose
    landmarks[172] = (CX - JAW_HALF_WIDTH_PX, CHIN_Y - 60.0)
    landmarks[397] = (CX + JAW_HALF_WIDTH_PX, CHIN_Y - 60.0)
    return landmarks


def body_pose():
    return BodyPose(joints={"neck_1_joint": (CX, NECK_JOINT_Y, 0.7)})


# -- circumference maths ------------------------------------------------------


def test_circle_model():
    assert circumference_circle(130.0) == pytest.approx(math.pi * 130.0)


def test_ellipse_range_matches_ramanujan():
    def ramanujan(a, b):
        h = ((a - b) / (a + b)) ** 2
        return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

    low, high = circumference_ellipse_range(130.0)
    assert low == pytest.approx(ramanujan(65.0, 0.85 * 65.0), rel=1e-9)
    assert high == pytest.approx(ramanujan(65.0, 0.90 * 65.0), rel=1e-9)
    # 0.93-0.95 x pi W, below the circle model.
    assert 0.92 * math.pi * 130 < low < high < 0.96 * math.pi * 130


def test_ellipse_of_a_circle_is_the_circle_model():
    low, high = circumference_ellipse_range(100.0, (1.0, 1.0))
    assert low == pytest.approx(math.pi * 100.0) and high == pytest.approx(math.pi * 100.0)


# -- automatic measurement ----------------------------------------------------


def test_cylinder_width_matches_the_method_definition():
    result = measure_neck_width(make_portrait(), face_mesh=face_mesh(), body_pose=body_pose())
    assert result.status == STATUS_OK, result.message
    assert result.band_source == BAND_SOURCE_VISION
    assert result.band == (CHIN_Y, NECK_JOINT_Y)
    assert result.width_mm == pytest.approx(expected_method_width_mm(), rel=0.02)
    # Known inset bias on an ideal cylinder (see expected_method_width_mm).
    assert 0.93 * 2 * NECK_RADIUS_MM < result.width_mm < 2 * NECK_RADIUS_MM
    left, right = _silhouette_columns()
    assert result.left_x == pytest.approx(left, abs=2)
    assert result.right_x == pytest.approx(right, abs=2)
    assert CHIN_Y < result.row_y <= NECK_JOINT_Y
    assert result.row_y in result.rows_used
    used = [r for r in result.rows if r.y in result.rows_used]
    nearest = min(used, key=lambda r: abs(r.width_mm - result.width_mm))
    assert result.row_y == nearest.y  # the row nearest the median, not the last
    assert result.quality == "good" and result.quality_reasons == []
    assert result.support_mm >= neck_width.GOOD_SUPPORT_MM
    z_edge = 0.5 * (result.left_depth_cm + result.right_depth_cm) * 10
    assert result.height_below_chin_mm == pytest.approx(
        (result.row_y - CHIN_Y) * z_edge / FX, rel=1e-6
    )
    assert result.height_below_chin_mm > neck_width.CHIN_CLEARANCE_MM
    assert result.roll_deg == pytest.approx(0.0)
    assert abs(result.neck_roll_deg) < 1.0
    assert result.circumference_circle_mm == pytest.approx(math.pi * result.width_mm)
    assert result.circumference_ellipse_mm == circumference_ellipse_range(result.width_mm)
    assert result.warnings == []


def test_collar_outside_nearer_is_rejected_with_warning():
    # Collar from just below the chin to the bottom: no clean row anywhere.
    portrait = make_portrait(collar=(CHIN_Y, 20.0))
    result = measure_neck_width(portrait, face_mesh=face_mesh(), body_pose=body_pose())
    assert result.status == STATUS_EDGES_OCCLUDED
    assert result.width_mm is None and result.circumference_circle_mm is None
    assert result.message.startswith(neck_width.ADVICE_COLLAR)
    assert result.reject_counts == {"occluded": len(result.rows)}
    assert any("outside nearer than neck edge (collar?)" in w for w in result.warnings)
    assert result.rows and not any(r.left_ok or r.right_ok for r in result.rows)


def test_collar_lower_down_leaves_the_clean_rows_above():
    top = CHIN_Y + 60
    result = measure_neck_width(
        make_portrait(collar=(top, 20.0)), face_mesh=face_mesh(), body_pose=body_pose()
    )
    assert result.status == STATUS_OK
    assert result.row_y < top
    assert result.width_mm == pytest.approx(expected_method_width_mm(), rel=0.02)


def test_collar_behind_the_edge_is_not_an_occluder():
    # "Outside" farther than the edge (e.g. a collar behind the neck side).
    result = measure_neck_width(
        make_portrait(collar=(CHIN_Y, -30.0)), face_mesh=face_mesh(), body_pose=body_pose()
    )
    assert result.status == STATUS_OK


def test_no_neck_skin_is_edges_occluded():
    portrait = make_portrait()
    skin = np.asarray(portrait.skinmap).copy()
    skin[int(CHIN_Y) :] = 0
    portrait.skinmap = Image.fromarray(skin, "L")
    result = measure_neck_width(portrait, face_mesh=face_mesh(), body_pose=body_pose())
    assert result.status == STATUS_EDGES_OCCLUDED
    assert "take the photo from further away" in result.message


def test_oblique_edges_are_rejected():
    """A V-shaped skin region (an open collar) is not a neck silhouette."""
    portrait = make_portrait()
    skin = np.zeros((PHOTO_H, PHOTO_W), dtype=np.uint8)
    left, right = _silhouette_columns()
    for y in range(int(CHIN_Y), PHOTO_H):
        inset = (y - CHIN_Y) * 0.8
        skin[y, int(left + inset) : int(right - inset) + 1] = 255
    portrait.skinmap = Image.fromarray(skin, "L")
    result = measure_neck_width(portrait, face_mesh=face_mesh(), body_pose=body_pose())
    assert result.status == STATUS_EDGES_OCCLUDED
    assert any("oblique" in w for w in result.warnings)


def _mesh_turned(jaw_shift_px=0.0, chin_shift_px=0.0):
    mesh = face_mesh()
    mesh[152] = (CX + chin_shift_px, CHIN_Y)
    mesh[172] = (mesh[172][0] + jaw_shift_px, mesh[172][1])
    mesh[397] = (mesh[397][0] + jaw_shift_px, mesh[397][1])
    return mesh


def _face_px_per_mm():
    """Pixels per mm at the jaw landmarks (the synthetic face plane)."""
    return FX / FACE_Z_MM


@pytest.mark.parametrize("fraction", [0.05, 0.06, 0.07, 0.08])
def test_small_head_turn_still_measures(fraction):
    """A turned head moves the chin ~10 cm in front of the neck axis by 5-8 %
    of the neck width and the jaw angles by about a third of that: the
    width is still measured (N1)."""
    left, right = _silhouette_columns()
    chin_shift = fraction * (right - left)
    mesh = _mesh_turned(jaw_shift_px=chin_shift / 3, chin_shift_px=chin_shift)
    result = measure_neck_width(make_portrait(), face_mesh=mesh, body_pose=body_pose())
    assert result.status == STATUS_OK, result.message
    assert result.width_mm == pytest.approx(expected_method_width_mm(), rel=0.02)


def test_moderate_jaw_offset_lowers_quality():
    shift = 7.0 * _face_px_per_mm()  # jaw centre 7 mm (3-D) from the neck centre
    result = measure_neck_width(
        make_portrait(), face_mesh=_mesh_turned(jaw_shift_px=shift), body_pose=body_pose()
    )
    assert result.status == STATUS_OK
    assert result.quality == "low"
    assert any("off the jaw centre" in r for r in result.quality_reasons)


def test_neck_far_off_the_jaw_centre_is_rejected_with_face_the_camera():
    shift = 15.0 * _face_px_per_mm()
    result = measure_neck_width(
        make_portrait(), face_mesh=_mesh_turned(jaw_shift_px=shift), body_pose=body_pose()
    )
    assert result.status == STATUS_EDGES_OCCLUDED
    assert result.message.startswith(neck_width.ADVICE_TURNED)
    assert any("off-centre" in w for w in result.warnings)


def test_chin_offset_band_without_vision_pose():
    result = measure_neck_width(make_portrait(), face_mesh=face_mesh(), use_vision=False)
    assert result.status == STATUS_OK
    assert result.band_source == BAND_SOURCE_CHIN_OFFSET
    z_chin_mm = float(_ray_cylinder_depth([CX])[0])
    expected_bottom = CHIN_Y + neck_width.CHIN_OFFSET_BAND_MM * FX / z_chin_mm
    assert result.band[1] == pytest.approx(expected_bottom, rel=0.01)
    assert result.width_mm == pytest.approx(expected_method_width_mm(), rel=0.02)


def test_chin_offset_band_when_vision_unavailable(monkeypatch):
    def unavailable():
        raise AppleVisionUnavailable("mocked: no Vision here")

    monkeypatch.setattr(apple_vision, "_import_backend", unavailable)
    result = measure_neck_width(make_portrait(), face_mesh=face_mesh())
    assert result.status == STATUS_OK
    assert result.band_source == BAND_SOURCE_CHIN_OFFSET
    assert result.neck_joint is None
    assert result.warnings == []


def test_chin_offset_band_when_vision_fails(monkeypatch, caplog):
    def failing(photo):
        raise AppleVisionError("mocked failure")

    monkeypatch.setattr(apple_vision, "detect_body_pose", failing)
    with caplog.at_level("WARNING"):
        result = measure_neck_width(make_portrait(), face_mesh=face_mesh())
    assert result.band_source == BAND_SOURCE_CHIN_OFFSET
    assert any("mocked failure" in w for w in result.warnings)
    assert "Apple Vision body pose failed" in caplog.text


def test_chin_offset_band_when_vision_finds_no_neck():
    pose = BodyPose(joints={"neck_1_joint": (CX, NECK_JOINT_Y, 0.05)})
    result = measure_neck_width(make_portrait(), face_mesh=face_mesh(), body_pose=pose)
    assert result.band_source == BAND_SOURCE_CHIN_OFFSET
    assert any("Vision neck joint not found" in w for w in result.warnings)


def test_non_darwin_detect_body_pose_raises_unavailable(monkeypatch):
    monkeypatch.setattr(apple_vision.sys, "platform", "linux")
    with pytest.raises(AppleVisionUnavailable):
        apple_vision.detect_body_pose(Image.new("RGB", (10, 10)))
    result = measure_neck_width(make_portrait(), face_mesh=face_mesh())
    assert result.status == STATUS_OK
    assert result.band_source == BAND_SOURCE_CHIN_OFFSET


def test_importing_the_package_does_not_import_vision():
    code = "import sys, portrait_analyser; print('Vision' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision needs macOS")
def test_vision_on_an_empty_canvas_finds_nobody():
    pytest.importorskip("Vision")
    assert apple_vision.detect_body_pose(Image.new("RGB", (300, 400), (40, 40, 40))) is None


def test_padded_canvas_mapping_round_trips():
    photo = Image.new("RGB", (300, 400))
    canvas, (ox, oy), scale = apple_vision.padded_canvas(photo)
    assert canvas.size == (150, 200)
    assert scale == pytest.approx(1 / 6)
    # Photo pixel (x, y) lands at canvas (ox + x * scale, oy + y * scale).
    assert (ox, oy) == ((150 - 50) // 2, (200 - 67) // 3)


def test_no_face(monkeypatch):
    from portrait_analyser import pose

    monkeypatch.setattr(pose, "detect_face_mesh", lambda image: None)
    result = measure_neck_width(make_portrait(), use_vision=False)
    assert result.status == STATUS_NO_FACE


def test_sideways_face_is_no_face():
    mesh = face_mesh()
    mesh[1] = (CX - 300.0, CHIN_Y)  # nose level with the chin: face rotated 90 degrees
    result = measure_neck_width(make_portrait(), face_mesh=mesh, use_vision=False)
    assert result.status == STATUS_NO_FACE
    assert "not upright" in result.message


def test_no_camera_is_reported_not_guessed():
    portrait = make_portrait(camera=False)
    result = measure_neck_width(portrait, face_mesh=face_mesh(), use_vision=False)
    assert result.status == STATUS_NO_CAMERA
    assert result.width_mm is None and "intrinsics" in result.message
    explicit = pa.CameraModel(fx=FX, fy=FX, cx=CX, cy=CY)
    result = measure_neck_width(portrait, face_mesh=face_mesh(), use_vision=False, camera=explicit)
    assert result.status == STATUS_OK


def test_implausible_depth_is_no_depth():
    portrait = make_portrait()
    portrait.depth_plausible = False
    result = measure_neck_width(portrait, face_mesh=face_mesh(), use_vision=False)
    assert result.status == "no-depth"


# -- legacy files ---------------------------------------------------------------


def test_legacy_portrait_returns_none():
    depth8 = Image.new("RGB", (60, 80), (128, 128, 128))
    legacy = IOSPortrait(
        Image.new("RGB", (600, 800)),
        depth8,
        floatValueMin=0.5,
        floatValueMax=3.5,
        skinmap=Image.new("L", (600, 800), 255),
    )
    assert legacy.depth.kind == "legacy"
    assert measure_neck_width(legacy, face_mesh=face_mesh()) is None
    assert neck_width_from_edges(legacy, (100, 400), (300, 400)) is None
    assert legacy.neck_width is None


def test_legacy_fixture_neck_width_is_none(heic_image_path):
    portrait = pa.load_image(str(heic_image_path))
    assert portrait.neck_width is None


# -- lazy property ----------------------------------------------------------------


def test_neck_width_property_is_lazy_and_cached(monkeypatch):
    calls = []

    def fake(portrait):
        calls.append(portrait)
        return "result"

    monkeypatch.setattr(neck_width, "measure_neck_width", fake)
    portrait = make_portrait()
    assert calls == []
    assert portrait.neck_width == "result"
    assert portrait.neck_width == "result"
    assert len(calls) == 1


# -- semi-automatic --------------------------------------------------------------


def test_width_from_clicked_edges():
    portrait = make_portrait()
    left, right = _silhouette_columns()
    y = CHIN_Y + 100
    result = neck_width_from_edges(portrait, (right, y), (left, y))  # any order
    assert result.status == STATUS_OK
    assert result.band_source == BAND_SOURCE_MANUAL
    assert result.left_x == pytest.approx(left) and result.right_x == pytest.approx(right)
    assert result.rows_used == [y]
    assert result.width_mm == pytest.approx(expected_method_width_mm(), rel=0.02)
    assert result.circumference_circle_mm == pytest.approx(math.pi * result.width_mm)
    assert result.warnings == []


def test_clicked_edges_on_a_collar_warn_but_measure():
    portrait = make_portrait(collar=(CHIN_Y, 20.0))
    left, right = _silhouette_columns()
    y = CHIN_Y + 100
    result = neck_width_from_edges(portrait, (left, y), (right, y))
    assert result.status == STATUS_OK
    assert result.width_mm == pytest.approx(expected_method_width_mm(), rel=0.02)
    assert sum("outside nearer than neck edge (collar?)" in w for w in result.warnings) == 2


def test_clicked_edges_without_depth():
    portrait = make_portrait()
    portrait.depth_m[:] = np.nan
    portrait.depth = None
    result = neck_width_from_edges(portrait, (100, 900), (300, 900))
    assert result.status == "no-depth"
    assert result.width_mm is None


# -- review follow-ups -----------------------------------------------------------


def _with(portrait, *, skin=None, depth_m=None):
    return IOSPortrait(
        photo=portrait.photo,
        skinmap=Image.fromarray(skin, "L") if skin is not None else portrait.skinmap,
        depth_m=depth_m if depth_m is not None else portrait.depth_m,
        depth_accuracy="absolute",
        depth_plausible=True,
        focal_length_px=(FX, FX),
        principal_point_px=(CX, CY),
    )


def _depth_coords(depth_m):
    rows, cols = depth_m.shape
    us = np.arange(cols) * (PHOTO_W - 1) / (cols - 1)
    vs = np.arange(rows) * (PHOTO_H - 1) / (rows - 1)
    return us, vs


def _hand_portrait(hand_mm=25.0, nearer_mm=10.0):
    """Skin (a hand) ``hand_mm`` wide against the right side of the neck,
    ``nearer_mm`` nearer than the neck just inside its edge."""
    base = make_portrait()
    skin = np.asarray(base.skinmap).copy()
    depth = base.depth_m.copy()
    _, right = _silhouette_columns()
    z = _inset_depth_mm()
    hand_px = hand_mm * FX / z
    skin[int(CHIN_Y) :, int(right) : int(right + hand_px)] = 255
    us, vs = _depth_coords(depth)
    cols = (us >= right) & (us < right + hand_px)
    depth[np.ix_(vs >= CHIN_Y, cols)] = (z - nearer_mm) / 1000.0
    return _with(base, skin=skin, depth_m=depth)


def test_hand_next_to_the_neck_is_rejected_with_advice():
    """The outside of the hand is background, so the occluder test passes;
    the depth step nearer towards the edge and the off-centre neck catch it,
    and the user is told why (N3)."""
    result = measure_neck_width(_hand_portrait(), face_mesh=face_mesh(), body_pose=body_pose())
    assert result.status == STATUS_EDGES_OCCLUDED
    assert result.warnings, "a rejection must never be silent"
    assert result.reject_counts["depth-step"] == len(result.rows)
    assert any("depth steps nearer towards an edge (hand, collar?)" in w for w in result.warnings)
    assert result.message.startswith(neck_width.ADVICE_BESIDE)


def test_hand_at_the_neck_depth_is_caught_by_the_off_centre_check():
    # 30 mm of skin flush with the neck side: centre ~15 mm off (> 12 mm).
    result = measure_neck_width(
        _hand_portrait(hand_mm=30.0, nearer_mm=0.0), face_mesh=face_mesh(), body_pose=body_pose()
    )
    assert result.status == STATUS_EDGES_OCCLUDED
    assert "off-centre" in result.reject_counts


@pytest.mark.parametrize("ratio", [1.25, 1.4])
def test_thick_neck_wider_than_the_jaw_is_measured(ratio):
    """Thick necks (the population screened by neck circumference) may be
    wider than the FaceMesh jaw: measured, low quality (N2)."""
    left, right = _silhouette_columns()
    mesh = face_mesh()
    half_jaw = (right - left) / ratio / 2
    mesh[172] = (CX - half_jaw, mesh[172][1])
    mesh[397] = (CX + half_jaw, mesh[397][1])
    result = measure_neck_width(make_portrait(), face_mesh=mesh, body_pose=body_pose())
    assert result.status == STATUS_OK
    assert result.width_mm == pytest.approx(expected_method_width_mm(), rel=0.02)
    assert result.quality == "low"
    assert any("jaw width" in r for r in result.quality_reasons)


def test_skin_span_far_wider_than_the_jaw_is_rejected():
    left, right = _silhouette_columns()
    mesh = face_mesh()
    half_jaw = (right - left) / 1.6 / 2
    mesh[172] = (CX - half_jaw, mesh[172][1])
    mesh[397] = (CX + half_jaw, mesh[397][1])
    result = measure_neck_width(make_portrait(), face_mesh=mesh, body_pose=body_pose())
    assert result.status == STATUS_EDGES_OCCLUDED
    assert "wider-than-jaw" in result.reject_counts
    assert result.message.startswith(neck_width.ADVICE_BESIDE)


def test_no_neck_rows_advises_distance():
    portrait = make_portrait()
    skin = np.asarray(portrait.skinmap).copy()
    skin[int(CHIN_Y) :] = 0
    portrait.skinmap = Image.fromarray(skin, "L")
    result = measure_neck_width(portrait, face_mesh=face_mesh(), body_pose=body_pose())
    assert result.message.startswith(neck_width.ADVICE_TOO_CLOSE)


def test_marginal_collar_is_kept_but_low_quality():
    # Outside nearer by ~0.3-0.45 cm (edge-dependent): below the 0.5 cm rejection.
    for nearer_mm in (2.0, 3.0):
        result = measure_neck_width(
            make_portrait(collar=(CHIN_Y, nearer_mm)), face_mesh=face_mesh(), body_pose=body_pose()
        )
        assert result.status == STATUS_OK
        assert result.quality == "low"
        assert any("collar close to the neck edge" in r for r in result.quality_reasons)
        # Two decimals: 0.48 must not print as the 0.5 cm rejection limit.
        assert re.search(r"by up to 0\.\d\d cm", result.quality_reasons[0])
        assert any("collar close to the neck edge" in w for w in result.warnings)


def test_head_roll_warns_and_lowers_quality():
    result = measure_neck_width(
        make_portrait(), face_mesh=face_mesh(roll_deg=12.0), body_pose=body_pose()
    )
    assert result.status == STATUS_OK
    assert result.roll_deg == pytest.approx(12.0, abs=0.01)
    assert result.quality == "low"
    assert any("head roll" in w for w in result.warnings)


def test_neck_roll_warns_and_lowers_quality():
    """The neck sheared by 12 degrees (edges within the slope limit)."""
    base = make_portrait()
    t = math.tan(math.radians(12.0))
    skin = np.asarray(base.skinmap).copy()
    for y in range(int(CHIN_Y), PHOTO_H):
        skin[y] = np.roll(skin[y], round((y - CHIN_Y) * t))
    depth = base.depth_m.copy()
    _us, vs = _depth_coords(depth)
    for i, y in enumerate(vs):
        if y >= CHIN_Y:
            depth[i] = np.roll(depth[i], round((y - CHIN_Y) * t / DEPTH_SCALE))
    result = measure_neck_width(
        _with(base, skin=skin, depth_m=depth), face_mesh=face_mesh(), use_vision=False
    )
    assert result.status == STATUS_OK
    assert result.neck_roll_deg == pytest.approx(12.0, abs=2.0)
    assert result.quality == "low"
    assert any("neck roll" in w for w in result.warnings)


def test_relative_depth_is_refused_even_with_a_camera():
    portrait = make_portrait()
    portrait.depth_accuracy = "relative"
    explicit = pa.CameraModel(fx=FX, fy=FX, cx=CX, cy=CY)
    for camera in (None, explicit):
        result = measure_neck_width(
            portrait, face_mesh=face_mesh(), use_vision=False, camera=camera
        )
        assert result.status == "relative-depth"
        assert "not 'absolute'" in result.message
    assert neck_width_from_edges(portrait, (400, 900), (700, 900)).status == "relative-depth"


def test_no_camera_messages_distinguish_the_cause():
    missing = measure_neck_width(make_portrait(camera=False), face_mesh=face_mesh())
    assert "no camera intrinsics" in missing.message

    class Unusable:  # intrinsics in the file, but portrait.camera is None
        def __getattr__(self, name):
            return getattr(portrait, name)

        camera = None

    portrait = make_portrait()
    result = measure_neck_width(Unusable(), face_mesh=face_mesh(), use_vision=False)
    assert result.status == STATUS_NO_CAMERA
    assert "not usable" in result.message


def test_portrait_without_depth_is_no_depth():
    portrait = IOSPortrait(Image.new("RGB", (60, 80)))
    assert portrait.depth is None
    assert measure_neck_width(portrait, face_mesh=face_mesh()).status == "no-depth"


def test_vision_joint_above_the_chin_falls_back_to_chin_offset():
    pose = BodyPose(joints={"neck_1_joint": (CX, CHIN_Y - 100.0, 0.9)})
    result = measure_neck_width(make_portrait(), face_mesh=face_mesh(), body_pose=pose)
    assert result.band_source == BAND_SOURCE_CHIN_OFFSET
    assert result.status == STATUS_OK
    assert any("Vision neck joint not found below the chin" in w for w in result.warnings)


def test_skin_span_touching_the_border_has_no_edges():
    skin = np.zeros((100, 200), dtype=np.float32)
    skin[:, 0:120] = 255  # runs off the left border
    kwargs = {"gap_px": 10, "search_px": 150, "min_span_px": 10}
    assert neck_width.skin_edges_at_row(skin, 50, 60, **kwargs) is None
    skin[:, 0] = 0
    left, right = neck_width.skin_edges_at_row(skin, 50, 60, **kwargs)
    assert 0 < left < 1.5 and right == pytest.approx(119.5, abs=0.6)


def test_beard_gap_at_the_midline_is_bridged():
    skin = np.zeros((100, 400), dtype=np.float32)
    skin[:, 100:300] = 255
    skin[:, 185:215] = 0  # dark beard / shadow in the middle
    edges = neck_width.skin_edges_at_row(skin, 50, 200, gap_px=40, search_px=150, min_span_px=10)
    assert edges is not None
    assert edges[0] == pytest.approx(99.5, abs=0.6) and edges[1] == pytest.approx(299.5, abs=0.6)


def test_clean_run_is_broken_by_a_tall_gap():
    def row(y):
        return neck_width.NeckWidthRow(y, 0, 1, 40, 40, 50, 50, 130.0, True, True)

    rows = [row(10), row(12), row(14), row(40), row(42)]
    assert [r.y for r in neck_width._topmost_clean_run(rows, 5)] == [10, 12, 14]
    assert len(neck_width._topmost_clean_run(rows, 50)) == 5


def test_neck_width_property_caches_errors(monkeypatch, caplog):
    calls = []

    def failing(portrait):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(neck_width, "measure_neck_width", failing)
    portrait = make_portrait()
    with caplog.at_level("ERROR"):
        first = portrait.neck_width
    assert first.status == "error" and "boom" in first.message
    assert portrait.neck_width is first
    assert calls == [1]
    assert "neck width measurement failed" in caplog.text


# -- Apple Vision with fake frameworks ----------------------------------------------


class _Point:
    def __init__(self, x, y, confidence):
        self._loc = type("Loc", (), {"x": x, "y": y})()
        self._confidence = confidence

    def location(self):
        return self._loc

    def confidence(self):
        return self._confidence


class _Observation:
    def __init__(self, points):
        self.points = points

    def confidence(self):
        return 0.9

    def recognizedPointsForGroupKey_error_(self, key, error):
        return self.points, None


def _fake_vision(points):
    class Request:
        def initWithCompletionHandler_(self, handler):
            return self

        def results(self):
            return [_Observation(points)]

    class Handler:
        def initWithCGImage_options_(self, image, options):
            return self

        def performRequests_error_(self, requests, error):
            return True, None

    class Vision:
        VNHumanBodyPoseObservationJointsGroupNameAll = "all"
        VNImageRequestHandler = type("H", (), {"alloc": staticmethod(Handler)})
        VNDetectHumanBodyPoseRequest = type("R", (), {"alloc": staticmethod(Request)})

    return Vision


def test_detect_body_pose_maps_vision_points_to_photo_pixels(monkeypatch):
    monkeypatch.setattr(apple_vision, "_cgimage_from_rgb", lambda quartz, image: (object(), b""))
    photo = Image.new("RGB", (300, 400))
    # Canvas 150x200 (half resolution), photo at 1/6 scale = 50x67 pasted at
    # ((150 - 50) // 2, (200 - 67) // 3) = (50, 44).
    vision = _fake_vision(
        {
            "neck_1_joint": _Point(0.5, 0.25, 0.7),  # canvas (75, 150), origin bottom-left
            "nose_joint": _Point(0.4, 0.75, 0.0),  # confidence 0: dropped
        }
    )
    pose = apple_vision._detect_body_pose(vision, None, photo)
    x, y, confidence = pose.neck()
    assert x == pytest.approx((75 - 50) * 6)
    assert y == pytest.approx((150 - 44) * 6)
    assert confidence == 0.7
    assert "nose_joint" not in pose.joints


def test_old_macos_without_body_pose_request_is_unavailable(monkeypatch):
    fake = type(sys)("Vision")
    monkeypatch.setattr(apple_vision.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "Vision", fake)
    monkeypatch.setitem(sys.modules, "Quartz", type(sys)("Quartz"))
    with pytest.raises(AppleVisionUnavailable, match="VNDetectHumanBodyPoseRequest"):
        apple_vision._import_backend()


# -- framing and head turn in 3-D (re-review #2) ----------------------------------

JAW_HALF_MM = 55.0  # jaw landmarks either side of the neck axis


def _rotate(x, z, x0, z0, yaw_deg):
    t = math.radians(yaw_deg)
    dx, dz = x - x0, z - z0
    return x0 + dx * math.cos(t) - dz * math.sin(t), z0 + dx * math.sin(t) + dz * math.cos(t)


def _ellipse_neck_depth(u, axis_x_mm, yaw_deg, a_mm, b_mm):
    """Near-surface depth (mm) along photo columns ``u`` of an elliptic
    cylinder (semi-axes a lateral, b front-back) centred at
    (axis_x_mm, NECK_AXIS_Z_MM) and turned by ``yaw_deg``; NaN outside."""
    x = (np.asarray(u, dtype=float) - CX) / FX
    t = math.radians(yaw_deg)
    c, s_ = math.cos(t), math.sin(t)
    p, q = x * c + s_, axis_x_mm * c + NECK_AXIS_Z_MM * s_
    r, s0 = -x * s_ + c, -axis_x_mm * s_ + NECK_AXIS_Z_MM * c
    qa = p * p / a_mm**2 + r * r / b_mm**2
    qb = -2 * (p * q / a_mm**2 + r * s0 / b_mm**2)
    qc = q * q / a_mm**2 + s0 * s0 / b_mm**2 - 1
    disc = qb * qb - 4 * qa * qc
    with np.errstate(invalid="ignore"):
        z = (-qb - np.sqrt(disc)) / (2 * qa)
    return np.where(disc >= 0, z, np.nan)


def make_scene(
    *, axis_x_mm=0.0, yaw_deg=0.0, a_mm=NECK_RADIUS_MM, b_mm=NECK_RADIUS_MM, face_z_mm=FACE_Z_MM
):
    """Portrait + FaceMesh landmarks of a subject whose neck axis is
    ``axis_x_mm`` off the optical axis and whose head/neck is turned by
    ``yaw_deg`` about the neck axis (jaw landmarks and elliptic neck turn
    together)."""
    depth_w, depth_h = PHOTO_W // DEPTH_SCALE, PHOTO_H // DEPTH_SCALE
    us = np.arange(depth_w) * (PHOTO_W - 1) / (depth_w - 1)
    vs = np.arange(depth_h) * (PHOTO_H - 1) / (depth_h - 1)
    depth = np.full((depth_h, depth_w), BACKGROUND_Z_MM)
    below = vs >= CHIN_Y
    neck_row = _ellipse_neck_depth(us, axis_x_mm, yaw_deg, a_mm, b_mm)
    depth[below] = np.where(np.isfinite(neck_row), neck_row, BACKGROUND_Z_MM)
    face_centre_px = CX + axis_x_mm * FX / face_z_mm
    face_cols = np.abs(us - face_centre_px) <= 250
    depth[np.ix_(~below, face_cols)] = face_z_mm

    skin = np.zeros((PHOTO_H, PHOTO_W), dtype=np.uint8)
    xs = np.arange(PHOTO_W)
    on = np.isfinite(_ellipse_neck_depth(xs + 0.5, axis_x_mm, yaw_deg, a_mm, b_mm))
    skin[int(CHIN_Y) :, on] = 255
    skin[: int(CHIN_Y), np.abs(xs - face_centre_px) <= 250] = 255
    portrait = IOSPortrait(
        photo=Image.new("RGB", (PHOTO_W, PHOTO_H), (90, 90, 90)),
        skinmap=Image.fromarray(skin, "L"),
        depth_m=(depth / 1000.0).astype(np.float32),
        depth_accuracy="absolute",
        depth_plausible=True,
        focal_length_px=(FX, FX),
        principal_point_px=(CX, CY),
    )

    def project(x_mm, z_mm):
        return CX + x_mm * FX / z_mm

    landmarks = [(face_centre_px, CHIN_Y - 300.0)] * 478
    chin_x, chin_z = _rotate(axis_x_mm, NECK_AXIS_Z_MM - 110.0, axis_x_mm, NECK_AXIS_Z_MM, yaw_deg)
    landmarks[152] = (project(chin_x, chin_z), CHIN_Y)
    landmarks[1] = (project(chin_x, chin_z), CHIN_Y - 300.0)
    for index, side in ((172, -1), (397, 1)):
        jx, jz = _rotate(
            axis_x_mm + side * JAW_HALF_MM, face_z_mm, axis_x_mm, NECK_AXIS_Z_MM, yaw_deg
        )
        landmarks[index] = (project(jx, jz), CHIN_Y - 60.0)
    return portrait, landmarks


@pytest.mark.parametrize("axis_x_mm", [-80.0, 80.0])
def test_verdict_does_not_depend_on_framing(axis_x_mm):
    """A subject off the optical axis: the jaw (at the face plane, 40 cm) and
    the neck (~45-51 cm) project with different scales. Compared in pixels
    this read as ~16 mm off-centre (rejected); in 3-D it is centred."""
    portrait0, mesh0 = make_scene()
    centred = measure_neck_width(portrait0, face_mesh=mesh0, body_pose=body_pose())
    portrait, mesh = make_scene(axis_x_mm=axis_x_mm)
    shifted = measure_neck_width(portrait, face_mesh=mesh, body_pose=body_pose())
    assert shifted.status == centred.status == STATUS_OK
    assert shifted.quality == centred.quality == "good", shifted.quality_reasons
    assert shifted.width_mm == pytest.approx(centred.width_mm, rel=0.02)
    assert not any("off the jaw centre" in r for r in shifted.quality_reasons)


@pytest.mark.parametrize("yaw_deg", [-5.0, 5.0])
def test_small_real_head_turn_measures(yaw_deg):
    """Head and (elliptic) neck turned 5 degrees together: the jaw landmarks
    move ~5 mm and the neck edges get different depths; still measured."""
    straight, mesh0 = make_scene(a_mm=67.0, b_mm=57.0, face_z_mm=460.0)
    reference = measure_neck_width(straight, face_mesh=mesh0, body_pose=body_pose())
    portrait, mesh = make_scene(yaw_deg=yaw_deg, a_mm=67.0, b_mm=57.0, face_z_mm=460.0)
    result = measure_neck_width(portrait, face_mesh=mesh, body_pose=body_pose())
    assert result.status == STATUS_OK, result.message
    assert result.left_depth_cm != pytest.approx(result.right_depth_cm, abs=0.05)
    assert result.width_mm == pytest.approx(reference.width_mm, rel=0.03)


def test_large_head_turn_is_rejected_with_face_the_camera():
    portrait, mesh = make_scene(yaw_deg=20.0, a_mm=67.0, b_mm=57.0, face_z_mm=460.0)
    result = measure_neck_width(portrait, face_mesh=mesh, body_pose=body_pose())
    assert result.status == STATUS_EDGES_OCCLUDED
    assert "off-centre" in result.reject_counts
    assert result.message.startswith(neck_width.ADVICE_TURNED)


def test_depth_step_counts_for_hand_advice_only_as_the_sole_reason():
    def row(codes):
        r = neck_width.NeckWidthRow(0, 0, 1, 40, 40, 50, 50, 130.0, False, False)
        r.reject_codes = list(codes)
        return r

    collar_v = [row(["occluded", "depth-step"])] * 6 + [row(["oblique"])] * 4
    counts = neck_width._count_rejections(collar_v)
    assert neck_width._advice(counts, False, collar_v) == neck_width.ADVICE_COLLAR
    hand = [row(["depth-step"])] * 6 + [row(["oblique"])] * 4
    counts = neck_width._count_rejections(hand)
    assert neck_width._advice(counts, False, hand) == neck_width.ADVICE_BESIDE
