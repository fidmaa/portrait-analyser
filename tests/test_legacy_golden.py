"""Golden regression for Camera-app (legacy 8-bit) depth on the real fixtures.

The expected values were produced by portrait-analyser 0.7.0 (main 195ff6a)
with the positional pre-DepthMap API; every one must stay bit-for-bit equal.
They are pinned as repr() strings / SHA-256 digests, so any change in a float's
last bit fails the test.
"""

import hashlib
from pathlib import Path

import pytest

import portrait_analyser as pa
from portrait_analyser.depth_sampling import (
    measure_filtered_surface_length,
    median_filter_depthmap,
    sample_points_along_line,
)
from portrait_analyser.extended_neck import compute_neck_width_3d
from portrait_analyser.mouth import compute_mouth_measurement_from_facemesh
from portrait_analyser.neck import (
    compute_neck_circumference,
    find_stable_depth_x_from_edge,
)
from portrait_analyser.tmd import compute_tmd_3d

GOLDEN = {
    "heic_depth_data.heic": {
        "depth_sha256": "337ae30f161d927d1a77a835f598a316dd417b13701500c6056b839a54b52e8a",
        "fmax": "3.767578",
        "fmin": "0.508301",
        "incisor": "(None, None, None, None)",
        "mouth": "MouthMeasurement(upper_point=(1160.0, 1503.5), "
        "lower_point=(1165.0, 1663.5), upper_depth_raw=199, "
        "lower_depth_raw=195, upper_distance_cm=32.76738373378601, "
        "lower_distance_cm=33.32567626912703, "
        "distance_3d_mm=19.915201737086402)",
        "neck": "(1942, 713, 1512, 152.3111242079148, 456.9333726237444, 115.97490616180438)",
        "neck_points_sha256": "c5f3526f097e2999f62f1f756c98e8efd64bffc666f756c0a832e8168086ca5e",
        "neck_width": "(128.07410087164084, 101.03256513854083)",
        "stable_edge": "779",
        "surface": "None",
        "surface_centre": "58.78366291066226",
        "tmd": "(52.58279138924674, 32.63072118033432, 34.0508772317187)",
    },
    "heic_face_data.heic": {
        "depth_sha256": "83b39f2b159a3ce17a0a659661b027fdf1197aaaa3419440631634d1279b1a38",
        "fmax": "3.402344",
        "fmin": "0.493652",
        "incisor": "(None, None, None, None)",
        "mouth": "MouthMeasurement(upper_point=(1160.0, 1503.5), "
        "lower_point=(1165.0, 1663.5), upper_depth_raw=246, "
        "lower_depth_raw=238, upper_distance_cm=30.30592973237183, "
        "lower_distance_cm=31.167880427044846, "
        "distance_3d_mm=19.785999340514483)",
        "neck": "(1657, 1005, 1641, 99.37715818803548, 298.13147456410644, 89.70799763020064)",
        "neck_points_sha256": "3e858422824fdeaaec44fe1477def78bd9e12009c5b44d114e681f41306289de",
        "neck_width": "(87.86771694191533, 79.08010310587757)",
        "stable_edge": "779",
        "surface": "87.08301463038767",
        "surface_centre": "89.59175322793784",
        "tmd": "(58.214189342081134, 36.03630797934586, 37.581090044380176)",
    },
}


def legacy_outputs(path):
    p = pa.load_image(str(path))
    w, h = p.photo.size
    fmin, fmax = float(p.floatValueMin), float(p.floatValueMax)
    n = compute_neck_circumference(p.skinmap, p.depthmap, w, h, fmin, fmax, hairmap=p.hairmap)
    cx, cy = w / 2, h / 2
    lm = [(cx, cy - 40)] * 17 + [(cx + 5, cy + 120)]
    filt = median_filter_depthmap(p.depthmap)
    pts = list(sample_points_along_line(cx - 300, cy - 500, cx + 300, cy - 300, 5))
    return {
        "depth_sha256": hashlib.sha256(p.depthmap.tobytes()).hexdigest(),
        "fmin": repr(p.floatValueMin),
        "fmax": repr(p.floatValueMax),
        "incisor": repr(
            (p.teeth_bbox, p.incisor_distance, p.incisor_distance_3d_mm, p.incisor_measurement)
        ),
        "neck": repr(
            (
                n.neck_y,
                n.left_x,
                n.right_x,
                n.front_arc_length_mm,
                n.circumference_mm,
                n.front_chord_length_mm,
            )
        ),
        "neck_points_sha256": hashlib.sha256(
            repr((n.arc_points_3d, n.arc_points_photo)).encode()
        ).hexdigest(),
        "neck_width": repr(
            compute_neck_width_3d(p.depthmap, n.neck_y, n.left_x, n.right_x, w, h, fmin, fmax)
        ),
        "mouth": repr(compute_mouth_measurement_from_facemesh(lm, p.depthmap, w, h, fmin, fmax)),
        "surface": repr(measure_filtered_surface_length(filt, pts, w, h, fmin, fmax)),
        "surface_centre": repr(
            measure_filtered_surface_length(
                filt,
                list(sample_points_along_line(cx - 200, cy, cx + 200, cy + 50, 5)),
                w,
                h,
                fmin,
                fmax,
            )
        ),
        "stable_edge": repr(
            find_stable_depth_x_from_edge(p.depthmap, cx - 400, cy + 300, 1, w, h, 200)
        ),
        "tmd": repr(compute_tmd_3d((cx, cy + 300), (cx, cy + 700), 200, 190, fmin, fmax, w, h)),
    }


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_legacy_outputs_equal_0_7_0(name):
    assert legacy_outputs(Path(__file__).parent / name) == GOLDEN[name]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_depth_keyword_matches_golden(name):
    """The same measurements through portrait.depth give the same numbers."""
    p = pa.load_image(str(Path(__file__).parent / name))
    w, h = p.photo.size
    depth = p.depth
    assert depth.kind == "legacy"
    n = compute_neck_circumference(
        p.skinmap, None, w, h, None, None, hairmap=p.hairmap, depth=depth
    )
    golden = GOLDEN[name]
    assert (
        repr(
            (
                n.neck_y,
                n.left_x,
                n.right_x,
                n.front_arc_length_mm,
                n.circumference_mm,
                n.front_chord_length_mm,
            )
        )
        == golden["neck"]
    )
    cx, cy = w / 2, h / 2
    lm = [(cx, cy - 40)] * 17 + [(cx + 5, cy + 120)]
    assert (
        repr(compute_mouth_measurement_from_facemesh(lm, None, w, h, None, None, depth=depth))
        == golden["mouth"]
    )
    pts = list(sample_points_along_line(cx - 200, cy, cx + 200, cy + 50, 5))
    assert repr(depth.median_filtered().surface_length_mm(pts)) == golden["surface_centre"]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_neck_width_is_none_and_leaves_legacy_outputs_alone(name):
    """Camera-app files get no width-based neck measurement, and the legacy
    outputs stay golden with the neck_width module loaded and accessed."""
    path = Path(__file__).parent / name
    p = pa.load_image(str(path))
    assert p.neck_width is None
    assert pa.measure_neck_width(p) is None
    assert legacy_outputs(path) == GOLDEN[name]
