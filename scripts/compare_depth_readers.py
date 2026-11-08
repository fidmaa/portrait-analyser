#!/usr/bin/env python
"""Check that Linux and macOS read the same depth map from a 16-bit HEIC.

Two steps, on two machines:

1. Linux -- compute the depth with the cross-platform reader
   (``heif_depth.read_heif_depth``: raw item bytes + imagecodecs + XMP) and
   save it::

       uv run python scripts/compare_depth_readers.py dump PHOTOS_DIR DUMP_DIR

2. macOS -- read the same photos with Apple's ImageIO/AVFoundation
   (``apple_depth.read_apple_depth``, the reader the Mac app uses) and compare
   with the Linux dump, pixel by pixel::

       uv run python scripts/compare_depth_readers.py compare PHOTOS_DIR DUMP_DIR

   It prints one line per photo plus a summary, writes
   ``DUMP_DIR/comparison.csv`` and exits with 1 if any photo differs beyond
   the tolerance. The dump records which file it came from by name and
   SHA-256, so a different copy of a photo is reported, not compared.

What counts as "the same": identical shape and invalid (NaN) mask, every
valid pixel within ``--tolerance-mm`` (default 0.1 mm), and identical
accuracy / quality / filtered / source type / intrinsics / orientation.
Photos whose depth libheif can decode (8-bit HEVC) are skipped: they never
reach either 16-bit reader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from portrait_analyser.exceptions import AppleDepthDecodeError
from portrait_analyser.heif_depth import read_heif_depth

META_FIELDS = (
    "accuracy",
    "quality",
    "filtered",
    "source_type",
    "intrinsics",
    "intrinsics_reference_size",
    "lens_distortion_center",
    "depth_orientation",
)


def _photos(folder: Path, every: int):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".heic", ".heif"))
    return files[::every]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta(data) -> dict:
    meta = {}
    for field in META_FIELDS:
        value = getattr(data, field)
        meta[field] = list(value) if isinstance(value, tuple) else value
    return meta


def _versions() -> dict:
    import imagecodecs
    import pyheif

    import portrait_analyser

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "imagecodecs": imagecodecs.__version__,
        "libheif (pyheif)": pyheif.libheif_version(),
        "portrait_analyser": getattr(portrait_analyser, "__version__", "?"),
    }


def dump(photos_dir: Path, dump_dir: Path, every: int) -> int:
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "versions-dump.json").write_text(json.dumps(_versions(), indent=2))
    written = skipped = failed = 0
    for path in _photos(photos_dir, every):
        try:
            data = read_heif_depth(path)
        except AppleDepthDecodeError as exc:
            if "not JPEG-coded" in str(exc):
                skipped += 1
                continue
            print(f"FAIL  {path.name}: {exc}")
            failed += 1
            continue
        if data is None:
            skipped += 1
            continue
        meta = _meta(data)
        meta["file"] = path.name
        meta["sha256"] = _sha256(path)
        np.savez_compressed(
            dump_dir / f"{path.name}.npz", depth_m=data.depth_m, meta=json.dumps(meta)
        )
        written += 1
        print(f"dump  {path.name}  {data.depth_m.shape}  valid {np.isfinite(data.depth_m).mean():.1%}")
    print(f"\n{written} dumped, {skipped} skipped (no 16-bit depth), {failed} failed -> {dump_dir}")
    return 1 if failed else 0


def _compare_one(path: Path, dump_file: Path, tolerance_m: float) -> dict:
    from portrait_analyser.apple_depth import read_apple_depth

    saved = np.load(dump_file)
    linux_depth = saved["depth_m"]
    linux_meta = json.loads(str(saved["meta"]))
    row = {"file": path.name, "ok": False}
    if _sha256(path) != linux_meta["sha256"]:
        row["problem"] = "photo differs from the one dumped on Linux (SHA-256)"
        return row

    apple = read_apple_depth(path)
    if apple is None:
        row["problem"] = "ImageIO finds no depth"
        return row
    apple_depth = apple.depth_m
    if apple_depth.shape != linux_depth.shape:
        row["problem"] = f"shape {apple_depth.shape} (Apple) vs {linux_depth.shape} (Linux)"
        return row

    apple_nan, linux_nan = np.isnan(apple_depth), np.isnan(linux_depth)
    row["nan_mismatch_px"] = int((apple_nan != linux_nan).sum())
    both = ~apple_nan & ~linux_nan
    diff = np.abs(apple_depth[both].astype(np.float64) - linux_depth[both].astype(np.float64))
    row["valid_px"] = int(both.sum())
    row["max_diff_mm"] = float(diff.max() * 1000) if diff.size else 0.0
    row["median_diff_mm"] = float(np.median(diff) * 1000) if diff.size else 0.0
    row["median_depth_cm"] = float(np.median(apple_depth[both]) * 100) if diff.size else None

    apple_meta = _meta(apple)
    mismatched = []
    for field in META_FIELDS:
        a, b = apple_meta[field], linux_meta[field]
        if isinstance(a, list) and isinstance(b, list):
            same = len(a) == len(b) and np.allclose(a, b, rtol=1e-5)
        else:
            same = a == b
        if not same:
            mismatched.append(f"{field}: Apple {a!r} vs Linux {b!r}")
    row["meta_mismatch"] = "; ".join(mismatched)

    problems = []
    if row["nan_mismatch_px"]:
        problems.append(f"{row['nan_mismatch_px']} px valid on one side only")
    if row["max_diff_mm"] > tolerance_m * 1000:
        problems.append(f"max diff {row['max_diff_mm']:.3f} mm")
    if mismatched:
        problems.append("metadata differs")
    row["problem"] = ", ".join(problems)
    row["ok"] = not problems
    return row


def compare(photos_dir: Path, dump_dir: Path, tolerance_mm: float) -> int:
    if sys.platform != "darwin":
        print("compare needs macOS (it runs Apple's ImageIO/AVFoundation reader)")
        return 2
    (dump_dir / "versions-compare.json").write_text(json.dumps(_versions(), indent=2))
    rows = []
    for dump_file in sorted(dump_dir.glob("*.npz")):
        path = photos_dir / dump_file.name[: -len(".npz")]
        if not path.exists():
            rows.append({"file": path.name, "ok": False, "problem": "photo not found"})
        else:
            try:
                rows.append(_compare_one(path, dump_file, tolerance_mm / 1000))
            except Exception as exc:  # report and keep going
                rows.append({"file": path.name, "ok": False, "problem": f"{type(exc).__name__}: {exc}"})
        row = rows[-1]
        status = "OK  " if row["ok"] else "DIFF"
        detail = (
            f"max {row['max_diff_mm']:.4f} mm, median {row['median_diff_mm']:.4f} mm"
            if "max_diff_mm" in row
            else ""
        )
        print(f"{status}  {row['file']}  {detail}  {row.get('problem', '')}  {row.get('meta_mismatch', '')}")

    fields = [
        "file", "ok", "problem", "valid_px", "nan_mismatch_px", "max_diff_mm",
        "median_diff_mm", "median_depth_cm", "meta_mismatch",
    ]  # fmt: skip
    with open(dump_dir / "comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    bad = [r for r in rows if not r["ok"]]
    worst = max((r.get("max_diff_mm", 0.0) for r in rows), default=0.0)
    print(
        f"\n{len(rows) - len(bad)}/{len(rows)} identical within {tolerance_mm} mm "
        f"(worst pixel difference {worst:.4f} mm) -> {dump_dir / 'comparison.csv'}"
    )
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump", help="Linux: save depth from the imagecodecs reader")
    d.add_argument("photos_dir", type=Path)
    d.add_argument("dump_dir", type=Path)
    d.add_argument("--every", type=int, default=1, help="take every N-th photo (default: all)")
    c = sub.add_parser("compare", help="macOS: compare Apple's reader with a Linux dump")
    c.add_argument("photos_dir", type=Path)
    c.add_argument("dump_dir", type=Path)
    c.add_argument("--tolerance-mm", type=float, default=0.1)
    args = parser.parse_args()
    if args.command == "dump":
        return dump(args.photos_dir, args.dump_dir, args.every)
    return compare(args.photos_dir, args.dump_dir, args.tolerance_mm)


if __name__ == "__main__":
    sys.exit(main())
