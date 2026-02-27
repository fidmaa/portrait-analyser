"""CLI diagnostic tool for inspecting iOS Portrait Mode HEIC files.

Usage:
    python -m portrait_analyser <path>
    analyse-portrait <path>
"""

import argparse
import xml.etree.ElementTree as ET

import piexif
import pyheif

from .exceptions import ExifValidationFailed, NoDepthMapFound, UnknownExtension
from .ios import load_image


def _print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_section(title):
    print(f"\n--- {title} ---")


def _inspect_raw_container(path):
    """Inspect the raw HEIF container before any processing."""
    _print_header("RAW HEIF CONTAINER INSPECTION")

    with open(path, "rb") as f:
        container = pyheif.open_container(f)

    primary = container.primary_image

    # Primary image info
    _print_section("Primary Image")
    loaded = primary.image.load()
    print(f"  Mode: {loaded.mode}")
    print(f"  Size: {loaded.size[0]} x {loaded.size[1]}")
    print(f"  Data length: {len(loaded.data)} bytes")
    metadata_count = len(loaded.metadata) if loaded.metadata else 0
    print(f"  Metadata entries: {metadata_count}")

    # EXIF data
    _print_section("EXIF Data")
    exif_found = False
    for metadata in loaded.metadata or []:
        if metadata.get("type", "") == "Exif":
            exif_found = True
            try:
                exif = piexif.load(metadata["data"])
                ifd_0th = exif.get("0th", {})
                ifd_exif = exif.get("Exif", {})

                model = ifd_0th.get(piexif.ImageIFD.Model, b"")
                if isinstance(model, bytes):
                    model = model.decode("utf-8", errors="replace").strip("\x00")
                print(f"  Camera model: {model or '(not found)'}")

                make = ifd_0th.get(piexif.ImageIFD.Make, b"")
                if isinstance(make, bytes):
                    make = make.decode("utf-8", errors="replace").strip("\x00")
                print(f"  Make: {make or '(not found)'}")

                lens_info = ifd_exif.get(42036, b"")
                if isinstance(lens_info, bytes):
                    lens_info = lens_info.decode("utf-8", errors="replace").strip(
                        "\x00"
                    )
                print(f"  Lens info (tag 42036): {lens_info or '(not found)'}")

                if lens_info and "front TrueDepth" in str(lens_info):
                    print("  -> TrueDepth camera: YES")
                else:
                    print("  -> TrueDepth camera: NO (tag 42036 does not match)")
            except Exception as e:
                print(f"  Error parsing EXIF: {e}")
    if not exif_found:
        print("  No EXIF metadata found")

    # Depth image
    _print_section("Depth Image")
    if primary.depth_image is not None:
        depth_loaded = primary.depth_image.image.load()
        print("  Present: YES")
        print(f"  Mode: {depth_loaded.mode}")
        print(f"  Size: {depth_loaded.size[0]} x {depth_loaded.size[1]}")
        print(f"  Data length: {len(depth_loaded.data)} bytes")

        # Parse depth metadata for float min/max
        if depth_loaded.metadata:
            for meta in depth_loaded.metadata:
                if meta.get("type", "") == "mime":
                    try:
                        root = ET.fromstring(meta.get("data"))
                        for elem in root[0][0]:
                            tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                            print(f"  {tag_name}: {elem.text}")
                    except Exception as e:
                        print(f"  Error parsing depth metadata: {e}")
    else:
        print("  Present: NO")

    # Auxiliary images
    _print_section("Auxiliary Images")
    aux_images = primary.auxiliary_images
    if aux_images:
        print(f"  Count: {len(aux_images)}")
        for i, aux in enumerate(aux_images):
            aux_type = getattr(aux, "type", "(no type)")
            print(f"  [{i}] Type: {aux_type}")
            try:
                aux_loaded = aux.image.load()
                print(f"       Mode: {aux_loaded.mode}")
                print(f"       Size: {aux_loaded.size[0]} x {aux_loaded.size[1]}")
                print(f"       Data length: {len(aux_loaded.data)} bytes")
            except Exception as e:
                print(f"       Error loading: {e}")
    else:
        print("  Count: 0 (no auxiliary images)")


def _inspect_processed(path, skip_exif):
    """Run load_image() and inspect the processed result."""
    _print_header("PROCESSED INSPECTION (via load_image)")

    try:
        result = load_image(path, use_exif=not skip_exif)
    except (ExifValidationFailed, NoDepthMapFound, UnknownExtension) as e:
        print(f"\n  load_image() raised {type(e).__name__}: {e}")
        print("  Try --skip-exif if this is an EXIF validation issue.")
        return
    except Exception as e:
        print(f"\n  load_image() raised unexpected {type(e).__name__}: {e}")
        return

    if result is None:
        print("\n  load_image() returned None")
        return

    _print_section("Photo")
    print(f"  Size: {result.photo.size[0]} x {result.photo.size[1]}")
    print(f"  Mode: {result.photo.mode}")

    _print_section("Depth Map")
    if result.depthmap is not None:
        print("  Present: YES")
        print(f"  Size: {result.depthmap.size[0]} x {result.depthmap.size[1]}")
        print(f"  Mode: {result.depthmap.mode}")
        print(f"  FloatValueMin: {result.floatValueMin}")
        print(f"  FloatValueMax: {result.floatValueMax}")
    else:
        print("  Present: NO")

    _print_section("Teeth Map")
    if result.teethmap is not None:
        print("  Present: YES")
        print(f"  Size: {result.teethmap.size[0]} x {result.teethmap.size[1]}")
        print(f"  Mode: {result.teethmap.mode}")
        print(f"  Teeth bbox: {result.teeth_bbox}")
        print(f"  Incisor distance (legacy): {result.incisor_distance}")
        print(f"  Incisor distance 3D (legacy, mm): {result.incisor_distance_3d_mm}")

        m = result.incisor_measurement
        if m is not None:
            print(f"  Upper centroid: ({m.upper_centroid[0]:.1f}, {m.upper_centroid[1]:.1f})")
            print(f"  Lower centroid: ({m.lower_centroid[0]:.1f}, {m.lower_centroid[1]:.1f})")
            print(f"  Pixel distance (Y): {m.pixel_distance_y:.1f}")
            print(f"  Upper depth raw: {m.upper_depth_raw}")
            print(f"  Lower depth raw: {m.lower_depth_raw}")
            print(f"  Upper distance (cm): {m.upper_distance_cm}")
            print(f"  Lower distance (cm): {m.lower_distance_cm}")
            print(f"  3D distance (mm): {m.distance_3d_mm}")
        else:
            print("  Incisor measurement: not available (centroids not found)")
    else:
        print("  Present: NO")
        print("  (teeth matte may exist in raw container but failed to decode)")

    _print_section("Skin Map")
    if result.skinmap is not None:
        print("  Present: YES")
        print(f"  Size: {result.skinmap.size[0]} x {result.skinmap.size[1]}")
        print(f"  Mode: {result.skinmap.mode}")
    else:
        print("  Present: NO")


def main():
    parser = argparse.ArgumentParser(
        prog="analyse-portrait",
        description="Diagnostic tool for inspecting iOS Portrait Mode HEIC files. "
        "Shows both raw HEIF container contents and processed library output.",
    )
    parser.add_argument("path", help="Path to a .heic or .heif file")
    parser.add_argument(
        "--skip-exif",
        action="store_true",
        help="Skip EXIF TrueDepth validation (useful for non-TrueDepth files)",
    )

    args = parser.parse_args()

    print(f"File: {args.path}")

    _inspect_raw_container(args.path)
    _inspect_processed(args.path, skip_exif=args.skip_exif)

    print()


if __name__ == "__main__":
    main()
