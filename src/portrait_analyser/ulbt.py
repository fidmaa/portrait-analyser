"""Colour-based upper lip bite test (ULBT) measurement.

The ULBT is an airway-assessment manoeuvre: the patient protrudes the mandible
and bites the upper lip with the lower incisors. It is graded by where the
incisal edge lands relative to the **vermilion line** of the upper lip:

    class I   - the lower incisors cover the upper lip above the vermilion line
    class II  - they reach the upper lip but stay below the vermilion line
    class III - they cannot reach the upper lip at all

Neither MediaPipe FaceMesh nor semantic face-parsing models can be trusted to
locate the vermilion in this pose. Both were trained on faces where an upper
lip is visible above the teeth, so when the vermilion is rolled under by the
bite they *fill it in*: FaceMesh places its "upper lip" ring on the skin of the
philtrum, and FaRL/LaPa parsing independently labels that same skin `u_lip`.
Two models, two architectures, same hallucination — because both are reporting
a prior rather than an observation.

What is observable is colour. Vermilion is measurably redder than the
surrounding skin, so this module ignores what the models *call* the region and
asks what the pixels actually are:

1. take the band the model proposes as upper lip (between the outer and inner
   FaceMesh lip rings) -- the models bracket the right region reliably even
   when they mislabel it;
2. build reference colours from regions we can trust: a skin strip above the
   band, the lower lip (unbitten vermilion in any ULBT capture), and
   optionally the iOS teeth matte;
3. classify every band pixel against those references in CIELAB chromaticity.

The share of the band that is still vermilion falls as the incisors cover more
of the upper lip, which makes ``bite_coverage`` a continuous stand-in for the
ULBT grade.

.. warning::
   **The class I/II/III thresholds are not established.** This measurement has
   been checked against a single capture. It deliberately returns a continuous
   value and no class label; calibrate against a graded series before reading
   a grade out of it.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy
from PIL import Image, ImageDraw

# --- FaceMesh lip contour rings (canonical 468/478-point topology) ----------
# Each ring runs left corner -> right corner. The boundary between
# UPPER_LIP_OUTER and UPPER_LIP_INNER is the upper vermilion border.
UPPER_LIP_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
UPPER_LIP_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
LOWER_LIP_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
LOWER_LIP_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)

_MOUTH_CORNER_LEFT = 61
_MOUTH_CORNER_RIGHT = 291

_REQUIRED_LANDMARKS = max(UPPER_LIP_OUTER + UPPER_LIP_INNER + LOWER_LIP_INNER + LOWER_LIP_OUTER) + 1

# Skin reference strip, as a fraction of mouth width above the outer lip ring.
# It starts clear of the ring so it cannot straddle the vermilion edge.
_SKIN_STRIP_NEAR = 0.04
_SKIN_STRIP_FAR = 0.16

# A teethmap pixel this bright is unambiguously teeth.
_TEETH_MATTE_THRESHOLD = 200

# Below this, a reference region is too small to give a trustworthy median.
_MIN_REFERENCE_PIXELS = 50


@dataclass
class ULBTMeasurement:
    """How much of the upper lip vermilion the lower incisors have covered."""

    bite_coverage: float  # 1.0 = vermilion fully covered, 0.0 = fully visible
    lip_share: float  # fraction of the band still reading as vermilion
    skin_share: float
    teeth_share: float | None  # None when no teethmap was supplied
    mouth_width_px: float


@dataclass
class ULBTDebug:
    """Intermediate state for overlay/debug visualization."""

    band_mask: numpy.ndarray  # bool (h, w): the band that was classified
    label_map: numpy.ndarray  # int8 (h, w): reference index per band pixel, -1 elsewhere
    reference_names: tuple[str, ...]
    reference_swatches: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    reference_pixels: dict[str, int] = field(default_factory=dict)
    band_pixels: int = 0


def _ring_polygon_mask(landmarks, outer_indices, inner_indices, size):
    """Boolean mask of the band between two landmark rings."""
    polygon = [landmarks[index] for index in outer_indices]
    polygon += [landmarks[index] for index in reversed(inner_indices)]
    canvas = Image.new("L", size, 0)
    ImageDraw.Draw(canvas).polygon(polygon, fill=255)
    return numpy.asarray(canvas) > 0


def _skin_strip_mask(landmarks, size, mouth_width):
    """Skin strip directly above the outer lip ring, under the same lighting."""
    ring = numpy.array([landmarks[index] for index in UPPER_LIP_OUTER], dtype=float)
    near = ring - numpy.array([0.0, _SKIN_STRIP_NEAR * mouth_width])
    far = ring - numpy.array([0.0, _SKIN_STRIP_FAR * mouth_width])
    polygon = [tuple(point) for point in near] + [tuple(point) for point in far[::-1]]
    canvas = Image.new("L", size, 0)
    ImageDraw.Draw(canvas).polygon(polygon, fill=255)
    return numpy.asarray(canvas) > 0


def _teeth_mask(teethmap, size):
    """iOS teeth matte, resized to photo dimensions and thresholded."""
    resized = teethmap.convert("L").resize(size, Image.BILINEAR)
    return numpy.asarray(resized) > _TEETH_MATTE_THRESHOLD


def compute_ulbt_from_facemesh(photo, landmarks, teethmap=None):
    """Measure how far the lower incisors have covered the upper lip vermilion.

    :param photo: PIL Image of the portrait (converted to RGB internally)
    :param landmarks: tuple of 478 (x, y) in photo-space pixels
        (``FaceMeshDebug.landmarks``)
    :param teethmap: optional PIL Image teeth matte from :class:`IOSPortrait`.
        Supplying it adds a third reference class so incisors overlapping the
        band are not forced to choose between skin and lip.
    :returns: ``(ULBTMeasurement, ULBTDebug)``, or ``(None, None)`` when the
        landmarks are insufficient or a reference region is too small to
        characterise.

    .. warning::
       Returns a continuous coverage value, not a ULBT class. The I/II/III
       thresholds have not been established -- see the module docstring.
    """
    import cv2

    if len(landmarks) < _REQUIRED_LANDMARKS:
        return None, None

    rgb = numpy.ascontiguousarray(numpy.asarray(photo.convert("RGB")))
    height, width = rgb.shape[:2]
    size = (width, height)

    corner_left = numpy.array(landmarks[_MOUTH_CORNER_LEFT], dtype=float)
    corner_right = numpy.array(landmarks[_MOUTH_CORNER_RIGHT], dtype=float)
    mouth_width = float(numpy.hypot(*(corner_right - corner_left)))
    if mouth_width <= 0:
        return None, None

    band = _ring_polygon_mask(landmarks, UPPER_LIP_OUTER, UPPER_LIP_INNER, size)
    if not band.any():
        return None, None

    references = {
        "skin": _skin_strip_mask(landmarks, size, mouth_width),
        "lip": _ring_polygon_mask(landmarks, LOWER_LIP_INNER, LOWER_LIP_OUTER, size),
    }
    if teethmap is not None:
        teeth = _teeth_mask(teethmap, size)
        if teeth.sum() >= _MIN_REFERENCE_PIXELS:
            references["teeth"] = teeth

    if any(mask.sum() < _MIN_REFERENCE_PIXELS for mask in references.values()):
        return None, None

    # Chromaticity only: dropping L* keeps shading and exposure out of the
    # decision, which matters because the band sits in the shadow of the lip.
    chroma = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(numpy.float32)[:, :, 1:]

    names = tuple(references)
    centres = numpy.array([numpy.median(chroma[mask], axis=0) for mask in references.values()])
    distances = numpy.linalg.norm(chroma[band][:, None, :] - centres[None, :, :], axis=2)
    assignment = distances.argmin(axis=1)

    shares = {name: float((assignment == index).mean()) for index, name in enumerate(names)}
    lip_share = shares["lip"]

    label_map = numpy.full((height, width), -1, dtype=numpy.int8)
    label_map[band] = assignment

    measurement = ULBTMeasurement(
        bite_coverage=1.0 - lip_share,
        lip_share=lip_share,
        skin_share=shares["skin"],
        teeth_share=shares.get("teeth"),
        mouth_width_px=mouth_width,
    )
    debug = ULBTDebug(
        band_mask=band,
        label_map=label_map,
        reference_names=names,
        reference_swatches={
            name: tuple(int(value) for value in numpy.median(rgb[mask], axis=0))
            for name, mask in references.items()
        },
        reference_pixels={name: int(mask.sum()) for name, mask in references.items()},
        band_pixels=int(band.sum()),
    )
    return measurement, debug


# --- CLI -------------------------------------------------------------------

_THRESHOLD_CAVEAT = (
    "The class I/II/III thresholds are NOT established: this is a continuous\n"
    "measurement validated against a single capture. Calibrate against a graded\n"
    "series before reading a ULBT grade out of it."
)

_REPORT_LABELS = (
    ("lip", "vermilion visible"),
    ("skin", "skin"),
    ("teeth", "teeth"),
)


def _load_portrait(path):
    """Return ``(photo, teethmap)``; the teeth matte needs an iOS HEIC."""
    from .exceptions import UnknownExtension
    from .ios import load_image

    try:
        portrait = load_image(str(path))
    except UnknownExtension:
        # Any other format still measures, just without the teeth reference.
        return Image.open(path), None
    return portrait.photo, portrait.teethmap


def _print_report(path, photo, measurement, debug):
    print(f"{path.name} — {photo.width}x{photo.height}")
    print()
    print("Upper lip bite test (ULBT), colour measurement")
    print(f"  mouth width          {measurement.mouth_width_px:8.1f} px")
    print(f"  band classified      {debug.band_pixels:8d} px")
    print()

    shares = {
        "lip": measurement.lip_share,
        "skin": measurement.skin_share,
        "teeth": measurement.teeth_share,
    }
    for name, label in _REPORT_LABELS:
        share = shares[name]
        if share is None:
            continue
        swatch = debug.reference_swatches[name]
        count = debug.reference_pixels[name]
        print(f"  {label:<18s} {share:7.1%}   reference RGB {str(swatch):<16s} n={count}")

    print()
    print(f"  BITE COVERAGE        {measurement.bite_coverage:7.1%}   (1.0 = vermilion fully covered)")
    print()
    print(_THRESHOLD_CAVEAT)


def main(argv=None) -> int:
    """Report how far the lower incisors cover the upper lip vermilion."""
    from .pose import detect_face_mesh

    parser = argparse.ArgumentParser(
        prog="measure-ulbt",
        description=(
            "Measure how far the lower incisors cover the upper lip vermilion "
            "in an upper lip bite test (ULBT) photo. Reports a continuous "
            "coverage value, not a ULBT class."
        ),
    )
    parser.add_argument(
        "image", type=Path, help="iOS Portrait Mode HEIC (other formats work, without the teeth matte)"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a human report")
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"no such file: {args.image}", file=sys.stderr)
        return 2

    photo, teethmap = _load_portrait(args.image)

    face_mesh = detect_face_mesh(photo)
    if face_mesh is None:
        print("Face Mesh found no face in this image.", file=sys.stderr)
        return 1

    measurement, debug = compute_ulbt_from_facemesh(photo, face_mesh.landmarks, teethmap=teethmap)
    if measurement is None:
        print(
            "Could not characterise the lip region: degenerate landmarks or a "
            "reference region too small to sample.",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "image": str(args.image),
                    "bite_coverage": measurement.bite_coverage,
                    "lip_share": measurement.lip_share,
                    "skin_share": measurement.skin_share,
                    "teeth_share": measurement.teeth_share,
                    "mouth_width_px": measurement.mouth_width_px,
                    "band_pixels": debug.band_pixels,
                    "reference_swatches": debug.reference_swatches,
                    "reference_pixels": debug.reference_pixels,
                },
                indent=2,
            )
        )
        return 0

    _print_report(args.image, photo, measurement, debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
