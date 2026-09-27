"""Tests for the macOS ImageIO/AVFoundation Apple depth reader.

Most of this module needs macOS + the pyobjc Quartz/AVFoundation frameworks
to run at all, so almost every test is skipped elsewhere (notably in CI,
which runs on ubuntu-latest -- see .github/workflows/build.yml). The one
exception is the "unavailable" test below, which simulates a non-macOS
platform and so exercises the early-exit path on any OS.
"""

import sys

import numpy as np
import pytest

from portrait_analyser.apple_depth import MAX_PLAUSIBLE_DEPTH_M, read_apple_depth
from portrait_analyser.exceptions import AppleDepthUnavailable
from portrait_analyser.incisor import depth_raw_to_distance_cm
from portrait_analyser.ios import load_image

requires_macos = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Apple depth reading needs macOS ImageIO/AVFoundation via pyobjc",
)


def _build_synthetic_disparity_heic(path, width, height, disparities):
    """Write a tiny HEIC with a real AVDepthData disparity aux image.

    Builds a DisparityFloat32 CVPixelBuffer from ``disparities`` (an
    ``(height, width)`` array), wraps it in an AVDepthData, converts that to
    the CGImageDestination-writable aux-data dictionary via
    ``dictionaryRepresentationForAuxiliaryDataType_``, and writes it
    alongside a flat grey base image -- mirroring what a real capture app
    does, just with a hand-built pixel buffer instead of camera data.
    """
    import Quartz
    import AVFoundation
    from Foundation import NSURL

    err, disparity_buf = Quartz.CVPixelBufferCreate(
        None, width, height, Quartz.kCVPixelFormatType_DisparityFloat32, None, None
    )
    assert err == 0, f"CVPixelBufferCreate failed: {err}"
    Quartz.CVPixelBufferLockBaseAddress(disparity_buf, 0)
    try:
        bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(disparity_buf)
        base_address = Quartz.CVPixelBufferGetBaseAddress(disparity_buf)
        raw = base_address.as_buffer(bytes_per_row * height)
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row)
        rows[:, : width * 4] = np.asarray(disparities, dtype=np.float32).view(np.uint8).reshape(
            height, width * 4
        )
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(disparity_buf, 0)

    depth_data = AVFoundation.AVDepthData.alloc().initWithPixelBuffer_depthMetadataDictionary_(
        disparity_buf, {}
    )
    aux_dict, aux_type = depth_data.dictionaryRepresentationForAuxiliaryDataType_(None)

    err, rgb_buf = Quartz.CVPixelBufferCreate(
        None, width, height, Quartz.kCVPixelFormatType_32ARGB, None, None
    )
    assert err == 0, f"CVPixelBufferCreate failed: {err}"
    Quartz.CVPixelBufferLockBaseAddress(rgb_buf, 0)
    try:
        bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(rgb_buf)
        base_address = Quartz.CVPixelBufferGetBaseAddress(rgb_buf)
        raw = base_address.as_buffer(bytes_per_row * height)
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row)
        rows[:, : width * 4] = 128
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(rgb_buf, 0)

    ci_image = Quartz.CIImage.imageWithCVPixelBuffer_(rgb_buf)
    cg_image = Quartz.CIContext.context().createCGImage_fromRect_(
        ci_image, ((0, 0), (width, height))
    )

    url = NSURL.fileURLWithPath_(str(path))
    destination = Quartz.CGImageDestinationCreateWithURL(url, "public.heic", 1, None)
    assert destination is not None, "CGImageDestinationCreateWithURL failed"
    Quartz.CGImageDestinationAddImage(destination, cg_image, None)
    Quartz.CGImageDestinationAddAuxiliaryDataInfo(destination, aux_type, aux_dict)
    ok = Quartz.CGImageDestinationFinalize(destination)
    assert ok, "CGImageDestinationFinalize failed"


def test_read_apple_depth_raises_when_platform_unsupported(monkeypatch, tmp_path):
    """Simulates a non-macOS platform, so this runs (and is meaningful) on any OS."""
    monkeypatch.setattr(sys, "platform", "not-darwin")
    dummy = tmp_path / "whatever.heic"
    dummy.write_bytes(b"not a real image, never read")
    with pytest.raises(AppleDepthUnavailable):
        read_apple_depth(dummy)


@requires_macos
def test_read_apple_depth_returns_none_without_depth_aux_png(tmp_path):
    from PIL import Image

    path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)
    assert read_apple_depth(path) is None


@requires_macos
def test_read_apple_depth_returns_none_without_depth_aux_jpeg(tmp_path):
    from PIL import Image

    path = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)
    assert read_apple_depth(path) is None


@requires_macos
def test_read_apple_depth_plausible_on_existing_fixture(heic_image_path):
    result = read_apple_depth(heic_image_path)
    assert result is not None
    assert result.source_type == "disparity"
    assert result.accuracy in ("absolute", "relative")
    assert isinstance(result.filtered, bool)
    assert result.quality in ("high", "low")

    depth_m = result.depth_m
    assert depth_m.dtype == np.float32
    valid = np.isfinite(depth_m)
    # The fixture has no gaps -- a sanity floor well under 100% keeps this
    # robust without being able to assert an exact count.
    assert valid.sum() / valid.size > 0.9
    # A TrueDepth portrait capture: tens of cm to a couple of metres.
    assert 0.1 < np.nanmin(depth_m) < 5.0
    assert 0.1 < np.nanmax(depth_m) < 5.0

    assert result.intrinsics is not None
    fx, fy, cx, cy = result.intrinsics
    assert fx > 0 and fy > 0 and cx > 0 and cy > 0
    assert result.intrinsics_reference_size is not None


@requires_macos
def test_read_apple_depth_agrees_with_pyheif_path(heic_image_path):
    """Cross-checks the AVFoundation metres against the existing pyheif-based
    raw-disparity-byte -> cm conversion (`depth_raw_to_distance_cm`), at a
    sample of pixels. This is the crux of the module: it proves the new
    macOS reader's physical units agree with the library's already-trusted
    conversion, not just that *a* number comes out.
    """
    apple = read_apple_depth(heic_image_path)
    assert apple is not None

    portrait = load_image(str(heic_image_path))
    depthmap = portrait.depthmap
    float_min = portrait.floatValueMin
    float_max = portrait.floatValueMax
    width, height = depthmap.size

    rng = np.random.default_rng(0)
    compared = 0
    for _ in range(300):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        pixel = depthmap.getpixel((x, y))
        raw = pixel[0] if isinstance(pixel, tuple) else pixel
        expected_cm = depth_raw_to_distance_cm(raw, float_min, float_max)
        if expected_cm is None:
            continue
        actual_m = apple.depth_m[y, x]
        if not np.isfinite(actual_m):
            continue
        actual_cm = actual_m * 100.0
        # 8-bit disparity quantisation is coarse (~256 steps over the
        # FloatMin..FloatMax range); half a centimetre is generous slack
        # around that, well under the ~1 cm one quantisation step is worth
        # here, and still tight enough to catch a wrong conversion.
        assert abs(actual_cm - expected_cm) < 0.5, (x, y, raw, expected_cm, actual_cm)
        compared += 1

    # Guards against the loop silently comparing nothing.
    assert compared > 100


@requires_macos
def test_read_apple_depth_synthetic_round_trip(tmp_path):
    width, height = 4, 3
    disparities = np.linspace(0.2, 1.0, width * height, dtype=np.float32).reshape(height, width)
    path = tmp_path / "synthetic_depth.heic"
    _build_synthetic_disparity_heic(path, width, height, disparities)

    result = read_apple_depth(path)
    assert result is not None
    assert result.source_type == "disparity"
    # No metadata dict keys were recognised by AVDepthData's raw-pixel-buffer
    # initializer -- it defaults to relative/unfiltered/low, which is itself
    # a legitimate round trip to assert on.
    assert result.accuracy == "relative"
    assert result.filtered is False
    assert result.quality == "low"

    expected_depth_m = 1.0 / disparities
    actual = result.depth_m
    assert actual.shape == (height, width)
    assert np.all(np.isfinite(actual))
    # The HEIC container re-encodes the auxiliary disparity data (it is not
    # stored as a raw blob), which introduces a little quantisation error --
    # even for a tiny synthetic image. 5% is generous slack around the ~1-2%
    # observed in practice while still proving the disparity->depth
    # (1/disparity) conversion, not just "some numbers came back".
    np.testing.assert_allclose(actual, expected_depth_m, rtol=0.05)


@requires_macos
def test_pixel_buffer_to_depth_m_caps_implausible_values():
    """Directly exercises the sentinel/outlier cap documented on
    MAX_PLAUSIBLE_DEPTH_M: a near-zero-disparity pixel can convert to a huge
    fake distance (observed up to ~9999.975 m on a real capture) instead of
    a clean non-finite value, so anything past the cap must become NaN too.
    """
    import Quartz

    from portrait_analyser.apple_depth import _pixel_buffer_to_depth_m

    width, height = 4, 1
    err, pixel_buffer = Quartz.CVPixelBufferCreate(
        None, width, height, Quartz.kCVPixelFormatType_DepthFloat32, None, None
    )
    assert err == 0

    values = np.array([1.5, 9999.975, 0.0, -3.0], dtype=np.float32)
    Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
    try:
        bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
        base_address = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
        raw = base_address.as_buffer(bytes_per_row * height)
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row)
        rows[:, : width * 4] = values.view(np.uint8).reshape(height, width * 4)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)

    depth_m = _pixel_buffer_to_depth_m(Quartz, pixel_buffer)

    assert depth_m.shape == (height, width)
    assert depth_m[0, 0] == pytest.approx(1.5)
    assert np.isnan(depth_m[0, 1]), "value above MAX_PLAUSIBLE_DEPTH_M must become NaN"
    assert np.isnan(depth_m[0, 2]), "zero depth must become NaN"
    assert np.isnan(depth_m[0, 3]), "negative depth must become NaN"
    assert 9999.975 > MAX_PLAUSIBLE_DEPTH_M
