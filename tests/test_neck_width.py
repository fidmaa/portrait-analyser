"""Tests for the width-based neck measurement (synthetic portraits only).

The synthetic portrait is a vertical cylinder "neck" of known radius under a
flat "face", seen by a pinhole camera: float depth (metres) from exact
ray/cylinder intersection, the skin matte from the cylinder's silhouette.
"""

import math
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
    circumference_ellipse_range,
    circumference_pi_w,
    measure_neck_width,
    neck_width_from_edges,
)

PHOTO_W, PHOTO_H = 1200, 1600
DEPTH_SCALE = 4  # photo pixels per depth pixel (real files: ~6.3)
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


def face_mesh():
    landmarks = [(CX, CHIN_Y - 300.0)] * 478
    landmarks[152] = (CX, CHIN_Y)
    return landmarks


def body_pose():
    return BodyPose(joints={"neck_1_joint": (CX, NECK_JOINT_Y, 0.7)})


# -- circumference maths ------------------------------------------------------


def test_pi_w():
    assert circumference_pi_w(130.0) == pytest.approx(math.pi * 130.0)


def test_ellipse_range_matches_ramanujan():
    def ramanujan(a, b):
        h = ((a - b) / (a + b)) ** 2
        return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

    low, high = circumference_ellipse_range(130.0)
    assert low == pytest.approx(ramanujan(65.0, 0.85 * 65.0), rel=1e-9)
    assert high == pytest.approx(ramanujan(65.0, 0.90 * 65.0), rel=1e-9)
    # 0.93-0.95 x pi W, below the circle's upper bound.
    assert 0.92 * math.pi * 130 < low < high < 0.96 * math.pi * 130


def test_ellipse_of_a_circle_is_pi_w():
    low, high = circumference_ellipse_range(100.0, (1.0, 1.0))
    assert low == pytest.approx(math.pi * 100.0) and high == pytest.approx(math.pi * 100.0)


# -- automatic measurement ----------------------------------------------------


def test_cylinder_width_within_two_percent():
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
    assert len(result.rows_used) >= neck_width.MIN_CLEAN_ROWS
    assert result.row_y == result.rows_used[-1]
    assert result.circumference_pi_w_mm == pytest.approx(math.pi * result.width_mm)
    assert result.circumference_ellipse_mm == circumference_ellipse_range(result.width_mm)
    assert result.warnings == []


def test_collar_outside_nearer_is_rejected_with_warning():
    # Collar from just below the chin to the bottom: no clean row anywhere.
    portrait = make_portrait(collar=(CHIN_Y, 20.0))
    result = measure_neck_width(portrait, face_mesh=face_mesh(), body_pose=body_pose())
    assert result.status == STATUS_EDGES_OCCLUDED
    assert result.width_mm is None and result.circumference_pi_w_mm is None
    assert "neck edges not visible" in result.message
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


def test_off_midline_edges_are_rejected():
    mesh = face_mesh()
    left, _ = _silhouette_columns()
    mesh[152] = (left + 20.0, CHIN_Y)  # "midline" near one edge
    result = measure_neck_width(make_portrait(), face_mesh=mesh, body_pose=body_pose())
    assert result.status == STATUS_EDGES_OCCLUDED


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
    assert result.circumference_pi_w_mm == pytest.approx(math.pi * result.width_mm)
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
