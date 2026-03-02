"""Tests for automatic neck circumference measurement.

Uses synthetic PIL Images — no HEIC files needed.
"""

import numpy
from PIL import Image

from portrait_analyser.face import (
    _gaussian_kernel,
    estimate_neck_search_zone,
    find_neck_measurement_point,
    find_neck_narrowest_row,
)
from portrait_analyser.neck import (
    NeckMeasurement,
    compute_neck_circumference,
    estimate_face_from_skinmap,
)


def _make_tapered_skin_image(
    width,
    height,
    face_location,
    narrowest_offset=30,
    narrowest_width=80,
    taper=2,
    value=255,
    collar_y=None,
):
    """Create a synthetic skin map with a neck that narrows then widens.

    face_location: (x, y, w, h)
    narrowest_offset: how many rows below face bottom the narrowest point is
    narrowest_width: pixel width at the narrowest row
    taper: pixels of widening per row after the narrowest point
    collar_y: if set, stop drawing skin at this y coordinate (simulates clothing)
    """
    face_x, face_y, face_w, face_h = face_location
    face_bottom = face_y + face_h
    center_x = face_x + face_w // 2

    img = Image.new("L", (width, height), 0)

    # Draw neck from face bottom downward:
    # - First narrow from face width to narrowest_width
    # - Then widen from narrowest_width outward (simulating shoulders)
    for dy in range(height - face_bottom):
        y = face_bottom + dy
        if collar_y is not None and y >= collar_y:
            break

        if dy <= narrowest_offset:
            # Narrowing phase: linearly interpolate from face_w to narrowest_width
            if narrowest_offset > 0:
                ratio = dy / narrowest_offset
            else:
                ratio = 1.0
            current_width = int(face_w * (1 - ratio) + narrowest_width * ratio)
        else:
            # Widening phase: grow by taper pixels per row
            extra = (dy - narrowest_offset) * taper
            current_width = narrowest_width + extra
            # Don't exceed image bounds
            current_width = min(current_width, width)

        half_w = current_width // 2
        x_left = max(0, center_x - half_w)
        x_right = min(width, center_x + half_w)

        for x in range(x_left, x_right):
            img.putpixel((x, y), value)

    return img


def _make_depth_image(width, height, fill_value=128):
    """Create a uniform depth map."""
    return Image.new("L", (width, height), fill_value)


def _make_curved_depth_image(
    width, height, center_value=200, edge_value=150, center_x=None,
):
    """Create a depth map with front-surface curvature.

    Center of the image (horizontally) is closer to camera (higher disparity value),
    edges are farther (lower disparity value). This simulates the cylindrical
    shape of a neck viewed from the front.
    """
    if center_x is None:
        center_x = width // 2
    img = Image.new("L", (width, height), edge_value)
    for y in range(height):
        for x in range(width):
            dist_from_center = abs(x - center_x)
            max_dist = max(center_x, width - center_x)
            if max_dist == 0:
                val = center_value
            else:
                ratio = dist_from_center / max_dist
                val = int(center_value * (1 - ratio) + edge_value * ratio)
            img.putpixel((x, y), val)
    return img


class TestNeckMeasurementDataclass:
    def test_construction(self):
        m = NeckMeasurement(
            neck_y=500,
            left_x=100,
            right_x=300,
            arc_points_3d=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            arc_points_photo=[(100, 500), (300, 500)],
            front_arc_length_mm=10.5,
            circumference_mm=31.5,
        )
        assert m.neck_y == 500
        assert m.left_x == 100
        assert m.right_x == 300
        assert len(m.arc_points_3d) == 2
        assert len(m.arc_points_photo) == 2
        assert m.front_arc_length_mm == 10.5
        assert m.circumference_mm == 31.5
        assert m.circumference_multiplier == 1.75

    def test_custom_multiplier(self):
        m = NeckMeasurement(
            neck_y=500,
            left_x=100,
            right_x=300,
            arc_points_3d=[],
            arc_points_photo=[],
            front_arc_length_mm=10.0,
            circumference_mm=30.0,
            circumference_multiplier=3.0,
        )
        assert m.circumference_multiplier == 3.0


class TestComputeNeckCircumference:
    """Test the full compute_neck_circumference pipeline with synthetic data."""

    def _face_and_skin(self):
        """Create a synthetic setup: 400x600 image, face at top, neck below.

        Face bounding box: (100, 50, 200, 200) — x=100, y=50, w=200, h=200
        So face bottom edge is at y=250.
        Neck narrows from face width (200px) down to 80px, then widens again.
        This shape ensures find_neck_measurement_point terminates via the
        'widening' break condition.
        """
        width, height = 400, 600
        face_location = (100, 50, 200, 200)  # (x, y, w, h)

        skinmap = _make_tapered_skin_image(
            width, height,
            face_location=face_location,
            narrowest_offset=30,
            narrowest_width=80,
            taper=4,
            collar_y=350,
        )

        return width, height, face_location, skinmap

    def test_basic_measurement(self):
        """Basic case: uniform depth, skin neck below face."""
        width, height, face_location, skinmap = self._face_and_skin()
        depthmap = _make_depth_image(width, height, fill_value=180)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=20,
        )

        assert result is not None
        assert isinstance(result, NeckMeasurement)

        # Neck should be found below the face
        assert result.neck_y >= 250
        assert len(result.arc_points_3d) > 0
        assert len(result.arc_points_photo) > 0
        assert len(result.arc_points_3d) == len(result.arc_points_photo)
        assert result.front_arc_length_mm > 0
        assert result.circumference_mm > 0

    def test_circumference_equals_arc_times_multiplier(self):
        """circumference_mm should equal front_arc_length_mm * multiplier."""
        width, height, face_location, skinmap = self._face_and_skin()
        depthmap = _make_depth_image(width, height, fill_value=180)

        for mult in [2.5, 3.0, 3.5]:
            result = compute_neck_circumference(
                skinmap=skinmap,
                depthmap=depthmap,
                photo_width=width,
                photo_height=height,
                float_min=0.5,
                float_max=2.0,
                face_location=face_location,
                n_samples=20,
                circumference_multiplier=mult,
            )
            assert result is not None
            expected = result.front_arc_length_mm * mult
            assert abs(result.circumference_mm - expected) < 0.001
            assert result.circumference_multiplier == mult

    def test_n_samples_affects_point_count(self):
        """More samples should produce more arc points (up to skin width)."""
        width, height, face_location, skinmap = self._face_and_skin()
        depthmap = _make_depth_image(width, height, fill_value=180)

        result_10 = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=10,
        )
        result_40 = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=40,
        )

        assert result_10 is not None
        assert result_40 is not None
        assert len(result_40.arc_points_photo) > len(result_10.arc_points_photo)

    def test_curved_depth_longer_arc(self):
        """A curved depth surface should produce a longer arc than a flat one."""
        width, height, face_location, skinmap = self._face_and_skin()

        flat_depth = _make_depth_image(width, height, fill_value=180)
        curved_depth = _make_curved_depth_image(
            width, height, center_value=200, edge_value=140,
        )

        result_flat = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=flat_depth,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=30,
        )
        result_curved = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=curved_depth,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=30,
        )

        assert result_flat is not None
        assert result_curved is not None
        # Curved surface should have a longer front arc (the depth variation
        # adds a Z component to the inter-point distances)
        assert result_curved.front_arc_length_mm > result_flat.front_arc_length_mm

    def test_arc_sag_zero_gives_straight_line(self):
        """With arc_sag=0, all arc points should stay at neck_y (straight line)."""
        width, height, face_location, skinmap = self._face_and_skin()
        depthmap = _make_depth_image(width, height, fill_value=180)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=15,
            arc_sag=0,
        )
        assert result is not None
        for _, py in result.arc_points_photo:
            assert py == result.neck_y

    def test_edge_points_at_neck_y_center_below(self):
        """Edge points should be at neck_y; center points at or below neck_y."""
        width, height, face_location, skinmap = self._face_and_skin()
        depthmap = _make_depth_image(width, height, fill_value=180)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=20,
            arc_sag=30,  # explicit sag to test arc shape
        )
        assert result is not None
        for _, py in result.arc_points_photo:
            assert py >= result.neck_y  # All points at or below neck_y

    def test_arc_curves_downward(self):
        """With explicit arc_sag, center points should curve below neck_y."""
        width, height, face_location, skinmap = self._face_and_skin()
        depthmap = _make_depth_image(width, height, fill_value=180)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=20,
            arc_sag=30,  # explicit sag to test arc shape
        )
        assert result is not None

        # Collect Y values of arc points
        ys = [py for _, py in result.arc_points_photo]
        assert len(ys) > 4  # need enough points to check

        # Edge points (first and last) should be at neck_y
        assert ys[0] == result.neck_y
        assert ys[-1] == result.neck_y

        # Center points should be below neck_y (curved downward by arc)
        mid = len(ys) // 2
        center_ys = ys[mid - 1 : mid + 2]
        assert any(cy > result.neck_y for cy in center_ys), (
            f"Center points should curve below neck_y={result.neck_y}, got {center_ys}"
        )

    def test_no_skin_returns_none(self):
        """If no skin is detected below the face, returns None."""
        width, height = 400, 600
        face_location = (100, 50, 200, 200)
        skinmap = Image.new("L", (width, height), 0)  # all black, no skin
        depthmap = _make_depth_image(width, height, fill_value=180)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
        )
        assert result is None

    def test_zero_depth_returns_none(self):
        """All-zero depth map (zero disparity) should return None."""
        width, height, face_location, skinmap = self._face_and_skin()
        depthmap = _make_depth_image(width, height, fill_value=0)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
        )
        assert result is None

    def test_different_depth_resolution(self):
        """Depth map at different resolution than photo should still work."""
        width, height, face_location, skinmap = self._face_and_skin()

        # Depth map at half resolution
        depthmap = _make_depth_image(width // 2, height // 2, fill_value=180)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=15,
        )
        assert result is not None
        assert result.front_arc_length_mm > 0

    def test_auto_sag_no_chin_straight_line(self):
        """With uniform depth and arc_sag=None, auto-sag=0 → straight line."""
        width, height, face_location, skinmap = self._face_and_skin()
        depthmap = _make_depth_image(width, height, fill_value=180)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=15,
            arc_sag=None,  # auto-detect
        )
        assert result is not None
        # Uniform depth → no chin protrusion → sag=0 → all points at neck_y
        for _, py in result.arc_points_photo:
            assert py == result.neck_y

    def test_auto_sag_detects_chin(self):
        """With chin protrusion in depth map, auto-sag should curve below neck_y."""
        width, height, face_location, skinmap = self._face_and_skin()

        # Create depth map: high depth at center near neck_y (chin protrusion),
        # lower depth at edges and further down (neck surface)
        depthmap = _make_depth_image(width, height, fill_value=120)

        # First, run with uniform depth to find where neck_y lands
        probe = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=10,
            arc_sag=0,
        )
        assert probe is not None
        neck_y = probe.neck_y
        center_x = (probe.left_x + probe.right_x) // 2

        # Paint chin protrusion: high depth at center around neck_y,
        # fading to normal depth below (simulating chin sticking out)
        chin_depth = 220
        chin_rows = 20  # chin extends 20 rows below neck_y
        for dy in range(chin_rows + 15):
            y = neck_y + dy
            if y >= height:
                break
            for dx in range(-30, 31):
                x = center_x + dx
                if 0 <= x < width:
                    if dy <= chin_rows:
                        # Chin zone: high depth
                        depthmap.putpixel((x, y), chin_depth)
                    else:
                        # Transition: fade back to normal
                        fade = (dy - chin_rows) / 15
                        val = int(chin_depth * (1 - fade) + 120 * fade)
                        depthmap.putpixel((x, y), val)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=20,
            arc_sag=None,  # auto-detect
        )
        assert result is not None

        # Auto-detected sag should cause center points to curve below neck_y
        ys = [py for _, py in result.arc_points_photo]
        assert len(ys) > 4
        mid = len(ys) // 2
        center_ys = ys[mid - 1 : mid + 2]
        assert any(cy > result.neck_y for cy in center_ys), (
            f"Auto-sag should curve below neck_y={result.neck_y}, got {center_ys}"
        )

    def test_edge_inset_avoids_extreme_edges(self):
        """Arc points should be inset from raw skin edges by 5%."""
        width, height, face_location, skinmap = self._face_and_skin()

        # Create a depth map where extreme left/right edges near the neck
        # have very low (bad) depth, but the interior has normal depth.
        depthmap = _make_depth_image(width, height, fill_value=180)

        # First, probe to find where the raw neck edges are
        probe = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=face_location,
            n_samples=10,
            arc_sag=0,
        )
        assert probe is not None

        # The returned left_x/right_x should already be inset.
        # Verify the arc points don't touch the raw skin edges.

        # Find actual raw skin edges at neck_y from the skin map
        neck_y = probe.neck_y
        raw_left = None
        raw_right = None
        for x in range(width):
            if skinmap.getpixel((x, neck_y)) >= 200:
                if raw_left is None:
                    raw_left = x
                raw_right = x

        assert raw_left is not None
        assert raw_right is not None

        # The leftmost arc point x should be > raw_left (inset inward)
        arc_xs = [px for px, _ in probe.arc_points_photo]
        assert min(arc_xs) > raw_left, (
            f"Leftmost arc point {min(arc_xs)} should be inset from raw edge {raw_left}"
        )
        assert max(arc_xs) < raw_right, (
            f"Rightmost arc point {max(arc_xs)} should be inset from raw edge {raw_right}"
        )

    def test_no_face_location_uses_fallback(self):
        """Call with face_location=None — should auto-estimate from skin map."""
        width, height, _face_location, skinmap = self._face_and_skin()
        depthmap = _make_depth_image(width, height, fill_value=180)

        result = compute_neck_circumference(
            skinmap=skinmap,
            depthmap=depthmap,
            photo_width=width,
            photo_height=height,
            float_min=0.5,
            float_max=2.0,
            face_location=None,
            n_samples=20,
        )

        assert result is not None
        assert isinstance(result, NeckMeasurement)
        assert result.neck_y >= 250  # still below face region
        assert result.front_arc_length_mm > 0
        assert result.circumference_mm > 0


class TestEstimateFaceFromSkinmap:
    """Tests for the skin-map-based face location estimator."""

    def test_returns_tuple_for_valid_skin(self):
        """Estimate should produce a bounding box for a head+neck skin shape."""
        width, height = 400, 600
        skinmap = Image.new("L", (width, height), 0)
        cx = width // 2

        # Paint a head shape: oval face (widest at jaw y=250), then neck narrows
        jaw_y = 250
        head_top = 50
        head_half_w = 100  # widest at jaw

        # Head: oval from top to jaw
        for y in range(head_top, jaw_y + 1):
            ratio = (y - head_top) / (jaw_y - head_top)
            half = int(head_half_w * (0.6 + 0.4 * ratio))  # wider toward jaw
            for x in range(cx - half, cx + half):
                if 0 <= x < width:
                    skinmap.putpixel((x, y), 255)

        # Neck: narrower below jaw, stays narrow
        for dy in range(100):
            y = jaw_y + 1 + dy
            if y >= height:
                break
            half = max(30, head_half_w - dy * 2)  # narrows quickly
            for x in range(cx - half, cx + half):
                if 0 <= x < width:
                    skinmap.putpixel((x, y), 255)

        est = estimate_face_from_skinmap(skinmap)
        assert est is not None

        # The estimated face bottom should be near the jaw (widest row)
        est_bottom = est[1] + est[3]
        assert abs(est_bottom - jaw_y) < 5, (
            f"Estimated face bottom {est_bottom} too far from jaw {jaw_y}"
        )

    def test_returns_none_for_empty_skin(self):
        """All-black skin map should return None."""
        skinmap = Image.new("L", (400, 600), 0)
        assert estimate_face_from_skinmap(skinmap) is None

    def test_widest_row_matches_jaw(self):
        """The face bottom should sit at the widest skin row."""
        width, height = 300, 500
        skinmap = Image.new("L", (width, height), 0)

        # Paint an inverted triangle: widest at row 100, narrowing below
        widest_y = 100
        widest_half = 80
        for dy in range(200):
            y = widest_y + dy
            if y >= height:
                break
            half = max(10, widest_half - dy)
            cx = width // 2
            for x in range(cx - half, cx + half):
                skinmap.putpixel((x, y), 255)

        # Also paint some narrower rows above to simulate the top of the head
        for dy in range(50):
            y = widest_y - 50 + dy
            if y < 0:
                continue
            half = 40 + dy  # narrows toward top
            cx = width // 2
            for x in range(cx - half, cx + half):
                if 0 <= x < width:
                    skinmap.putpixel((x, y), 255)

        est = estimate_face_from_skinmap(skinmap)
        assert est is not None
        est_bottom = est[1] + est[3]
        # The widest row is at y=100 — estimated face bottom should be there
        assert abs(est_bottom - widest_y) < 5, (
            f"Estimated face bottom {est_bottom} should be near widest row {widest_y}"
        )


# ---------------------------------------------------------------------------
# Helper: synthetic Face with mock eyes (no actual image / Haar cascade)
# ---------------------------------------------------------------------------


class _MockFace:
    """Lightweight stand-in for Face that bypasses Haar cascade detection."""

    def __init__(self, x, y, width, height, eyes=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.eyes = eyes or []

    def __getitem__(self, i):
        return (self.x, self.y, self.width, self.height)[i]


class _MockEye:
    """Lightweight stand-in for Eye with center coordinates."""

    def __init__(self, center_x, center_y):
        self.center_x = center_x
        self.center_y = center_y


def _make_neck_skinmap(
    width=400,
    height=600,
    neck_center_x=200,
    neck_top_y=300,
    narrowest_y=350,
    narrowest_half_w=40,
    top_half_w=100,
    bottom_half_w=150,
    bottom_y=500,
    value=255,
):
    """Create a skin map with a clear neck shape.

    Tapers from top_half_w at neck_top_y down to narrowest_half_w at narrowest_y,
    then widens to bottom_half_w at bottom_y (simulating shoulders).
    Full-width: skin extends beyond any face box.
    """
    img = Image.new("L", (width, height), 0)
    for y in range(neck_top_y, bottom_y):
        if y <= narrowest_y:
            ratio = (y - neck_top_y) / max(narrowest_y - neck_top_y, 1)
            half_w = int(top_half_w * (1 - ratio) + narrowest_half_w * ratio)
        else:
            ratio = (y - narrowest_y) / max(bottom_y - narrowest_y, 1)
            half_w = int(narrowest_half_w * (1 - ratio) + bottom_half_w * ratio)

        x_left = max(0, neck_center_x - half_w)
        x_right = min(width, neck_center_x + half_w)
        for x in range(x_left, x_right):
            img.putpixel((x, y), value)
    return img


# ---------------------------------------------------------------------------
# Tests for estimate_neck_search_zone
# ---------------------------------------------------------------------------


class TestEstimateNeckSearchZone:
    def test_returns_zone_with_two_eyes(self):
        face = _MockFace(x=100, y=50, width=200, height=200)
        face.eyes = [
            _MockEye(center_x=60, center_y=80),   # left eye (face-relative)
            _MockEye(center_x=140, center_y=82),   # right eye (face-relative)
        ]
        zone = estimate_neck_search_zone(face)
        assert zone is not None
        center_x, scan_start_y, scan_width = zone
        # IPD = 80, midpoint = (100, 81) face-relative → (200, 131) absolute
        assert abs(center_x - 200) < 5
        # chin_y ~ 131 + 1.2*80 = 227, scan_start ~ 227 + 0.3*80 = 251
        assert scan_start_y > 200
        assert scan_width > 0

    def test_returns_none_with_fewer_than_two_eyes(self):
        face = _MockFace(x=100, y=50, width=200, height=200)
        face.eyes = [_MockEye(center_x=60, center_y=80)]
        assert estimate_neck_search_zone(face) is None

    def test_returns_none_with_no_eyes(self):
        face = _MockFace(x=100, y=50, width=200, height=200)
        face.eyes = []
        assert estimate_neck_search_zone(face) is None

    def test_rejects_eyes_too_close_horizontally(self):
        """Eyes very close together horizontally should be rejected."""
        face = _MockFace(x=100, y=50, width=200, height=200)
        # X separation = 10, which is < 20% of face width (40)
        face.eyes = [
            _MockEye(center_x=95, center_y=80),
            _MockEye(center_x=105, center_y=80),
        ]
        assert estimate_neck_search_zone(face) is None

    def test_picks_best_pair_from_multiple_eyes(self):
        """With >2 eyes detected, picks the pair with smallest Y difference."""
        face = _MockFace(x=100, y=50, width=200, height=200)
        face.eyes = [
            _MockEye(center_x=60, center_y=80),    # true left eye
            _MockEye(center_x=140, center_y=82),   # true right eye (close Y)
            _MockEye(center_x=100, center_y=140),  # spurious (eyebrow etc.)
        ]
        zone = estimate_neck_search_zone(face)
        assert zone is not None
        center_x, _, _ = zone
        # Should pick the first two eyes (closest Y), midpoint at x=200 absolute
        assert abs(center_x - 200) < 5


# ---------------------------------------------------------------------------
# Tests for find_neck_narrowest_row
# ---------------------------------------------------------------------------


class TestFindNeckNarrowestRow:
    def test_finds_collar_with_face_location(self):
        """Basic test: find the collar line using face_location fallback."""
        skinmap = _make_neck_skinmap(narrowest_y=350, narrowest_half_w=40)
        face_loc = (100, 50, 200, 200)  # face bottom at y=250
        result = find_neck_narrowest_row(
            skinmap, face_location=face_loc,
        )
        assert result is not None
        x_left, neck_y, x_right, neck_y2 = result
        assert neck_y == neck_y2
        # Should find collar near y=499 (bottom_y=500, last skin row)
        assert abs(neck_y - 499) < 5
        assert x_right > x_left

    def test_finds_collar_with_search_zone(self):
        """Using eye-anchored search zone should find the collar line."""
        skinmap = _make_neck_skinmap(narrowest_y=350, narrowest_half_w=40)
        search_zone = (200, 300, 300)  # center_x=200, start_y=300, width=300
        result = find_neck_narrowest_row(
            skinmap, search_zone=search_zone,
        )
        assert result is not None
        _, neck_y, _, _ = result
        # Should find collar near y=499 (bottom_y=500, last skin row)
        assert abs(neck_y - 499) < 5

    def test_full_width_scan_finds_neck_outside_face_box(self):
        """Neck edges outside the face box should still be detected."""
        # Neck centered at x=200 with half-width up to 150 at bottom
        # Face box at x=150..250 (width=100) — neck edges extend beyond
        skinmap = _make_neck_skinmap(
            narrowest_y=350, narrowest_half_w=120, top_half_w=120,
        )
        face_loc = (150, 50, 100, 200)  # narrow face box
        result = find_neck_narrowest_row(
            skinmap, face_location=face_loc,
        )
        assert result is not None
        x_left, _, x_right, _ = result
        # Full-width scan should find edges beyond face box (150..250)
        assert x_left < 150 or x_right > 250

    def test_single_gap_row_does_not_trigger_collar(self):
        """A single no-skin row shouldn't trigger collar detection (need 3 consecutive)."""
        skinmap = _make_neck_skinmap(narrowest_y=350, narrowest_half_w=40, bottom_y=500)
        # Erase a single row at y=400 (in the middle of the widening phase)
        arr = numpy.array(skinmap)
        arr[400, :] = 0
        skinmap_gap = Image.fromarray(arr)

        result = find_neck_narrowest_row(
            skinmap_gap, face_location=(100, 50, 200, 200),
        )
        assert result is not None
        _, neck_y, _, _ = result
        # Should NOT stop at y=399 (before the gap); should continue to bottom
        assert neck_y > 400

    def test_returns_none_for_no_skin(self):
        """All-black skin map should return None."""
        skinmap = Image.new("L", (400, 600), 0)
        result = find_neck_narrowest_row(
            skinmap, face_location=(100, 50, 200, 200),
        )
        assert result is None

    def test_returns_none_with_no_params(self):
        """Neither search_zone nor face_location → None."""
        skinmap = _make_neck_skinmap()
        assert find_neck_narrowest_row(skinmap) is None

    def test_finds_collar_not_narrowest(self):
        """Should find where skin disappears (collar), not the narrowest row."""
        # Skinmap: wide → narrow at y=350 → stays narrow → ends at y=450
        skinmap = _make_neck_skinmap(
            narrowest_y=350, narrowest_half_w=40,
            bottom_y=450, bottom_half_w=60,
        )
        face_loc = (100, 50, 200, 200)
        result = find_neck_narrowest_row(
            skinmap, face_location=face_loc,
        )
        assert result is not None
        _, neck_y, _, _ = result
        # Should find collar near y=449 (where skin ends), NOT near y=350 (narrowest)
        assert neck_y > 400, (
            f"Expected collar near y=449, got {neck_y} (would be ~350 if finding narrowest)"
        )
        assert abs(neck_y - 449) < 5


# ---------------------------------------------------------------------------
# Tests for updated find_neck_measurement_point
# ---------------------------------------------------------------------------


class TestFindNeckMeasurementPoint:
    def test_backward_compatible_without_face(self):
        """Calling without face= should work as before."""
        skinmap = _make_neck_skinmap(narrowest_y=350, narrowest_half_w=40)
        face_loc = (100, 50, 200, 200)
        result = find_neck_measurement_point(skinmap, face_loc)
        x_left, neck_y, x_right, neck_y2 = result
        assert neck_y == neck_y2
        assert x_right > x_left
        assert abs(neck_y - 350) < 15

    def test_with_face_uses_eye_anchored_zone(self):
        """When face= is provided with eyes, should use eye-anchored search."""
        skinmap = _make_neck_skinmap(narrowest_y=350, narrowest_half_w=40)
        face = _MockFace(x=100, y=50, width=200, height=200)
        face.eyes = [
            _MockEye(center_x=60, center_y=80),
            _MockEye(center_x=140, center_y=82),
        ]
        face_loc = (100, 50, 200, 200)
        result = find_neck_measurement_point(
            skinmap, face_loc, face=face,
        )
        assert result is not None
        _, neck_y, _, _ = result
        assert abs(neck_y - 350) < 15

    def test_with_face_no_eyes_falls_back(self):
        """Face with no eyes → falls back to face_location-based scan."""
        skinmap = _make_neck_skinmap(narrowest_y=350, narrowest_half_w=40)
        face = _MockFace(x=100, y=50, width=200, height=200)
        face.eyes = []
        face_loc = (100, 50, 200, 200)
        result = find_neck_measurement_point(
            skinmap, face_loc, face=face,
        )
        assert result is not None
        _, neck_y, _, _ = result
        assert abs(neck_y - 350) < 15

    def test_raises_on_no_skin(self):
        """Should raise IndexError when no valid neck row exists."""
        skinmap = Image.new("L", (400, 600), 0)
        face_loc = (100, 50, 200, 200)
        try:
            find_neck_measurement_point(skinmap, face_loc)
            assert False, "Should have raised IndexError"
        except IndexError:
            pass


# ---------------------------------------------------------------------------
# Tests for Gaussian kernel
# ---------------------------------------------------------------------------


class TestGaussianKernel:
    def test_sums_to_one(self):
        kernel = _gaussian_kernel(3.0)
        assert abs(kernel.sum() - 1.0) < 1e-6

    def test_symmetric(self):
        kernel = _gaussian_kernel(2.0)
        n = len(kernel)
        for i in range(n // 2):
            assert abs(kernel[i] - kernel[n - 1 - i]) < 1e-10

    def test_peak_at_center(self):
        kernel = _gaussian_kernel(2.0)
        center = len(kernel) // 2
        assert kernel[center] == kernel.max()
