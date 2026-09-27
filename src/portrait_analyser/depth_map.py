"""One depth abstraction for every measurement: :class:`DepthMap`.

A portrait's depth comes in two very different shapes:

* **Camera-app files** (iPhone 12/14 ...): an 8-bit disparity image plus
  Apple's ``FloatMinValue``/``FloatMaxValue``. :class:`LegacyDepthMap` wraps
  exactly that and samples it exactly as the library always has
  (``sample_depth_at_point``, ``median_filter_depthmap``,
  ``sample_filtered_depth``, ``depth_raw_to_distance_cm``), so every number
  stays bit-for-bit identical.
* **TrueDepth capture-app files**: full-precision float depth in metres
  (``IOSPortrait.depth_m``). :class:`FloatDepthMap` samples that float map
  directly -- NaN is invalid, medians and bilinear interpolation run on the
  float metres, there is no 8-bit quantisation and no 3 m far cap. The 8-bit
  image (``to_display_image()`` / ``IOSPortrait.depthmap``) exists for
  display only and is never read by a measurement.

``load_image`` attaches the right one as ``portrait.depth``. The API a GUI
needs::

    depth = portrait.depth
    depth.distance_cm(x, y, radius=1)      # cm at a photo pixel (median), or None
    depth.shape                            # (rows, cols) of the native map
    depth.valid_mask                       # bool array, native resolution
    depth.photo_to_depth(x, y)             # photo pixel -> native depth pixel
    smooth = depth.median_filtered()       # NaN-aware 3x3 median variant
    smooth.profile(points, camera=portrait.camera)       # cm along photo points
    smooth.surface_length_mm(points, camera=portrait.camera)
    depth.integration_map(camera)          # the map profiles/lengths integrate over
    depth.distance_3d_mm(p1, p2, camera=portrait.camera)
    depth.to_cm_array()                    # float cm, NaN = invalid
    depth.to_display_image()               # 8-bit, display only

Integration along paths (``profile``, ``surface_length_mm``, the neck arc):
legacy maps use the 3x3 median exactly as before; float maps -- unfiltered
TrueDepth depth with ~1 mm per-pixel jitter -- always go through
:meth:`DepthMap.integration_map` (3x3 median + edge-preserving 6 mm bilateral,
see :data:`INTEGRATION_SIGMA_MM`), whichever variant they are called on.

All coordinates are photo-space pixels of the full-resolution upright photo
(``portrait.photo.size``); the map knows that size (``photo_size``) and maps
into its own resolution with the library's endpoint-matching convention
(``x_depth = x * (depth_w - 1) / (photo_w - 1)``).

"Code" values (:meth:`DepthMap.code_array`, :meth:`DepthMap.bilinear_code`,
:attr:`DepthSample.code`) are a *detector scale*, not a measurement: the
0-255 disparity-code scale (higher = nearer) on which the neck/chin
detectors' thresholds were tuned. Legacy maps return their stored codes;
float maps return ``255 * (1/Z - float_min) / (float_max - float_min)``
unquantised on the fixed :data:`FLOAT_DETECTOR_CODE_RANGE` (1/3 m .. 1/0.25
m, saturating at 1..255, "invalid" beyond 3 m) -- independent of the file's
own depth range and of the display encoding. Metric values never use this
scale.

Other float sources (e.g. a multi-frame median depth exported by the capture
app) only need a ``(rows, cols)`` metres array aligned with the photo:
``FloatDepthMap(depth_m, photo.size)``.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .apple_depth import MAX_PLAUSIBLE_DEPTH_M, encode_depth_as_disparity_8bit
from .depth_sampling import median_filter_depthmap, sample_filtered_depth
from .face import sample_depth_at_point, teeth_threshold
from .incisor import (
    depth_raw_to_distance_cm,
    distance_3d_from_cm,
    point_to_mm,
    raw_depth_to_distance_cm,
    vector_length_3d,
)

# Detector scale of float maps: a FIXED reference disparity range, 1/3 m ..
# 1/0.25 m, i.e. the 8-bit code scale a typical capture-app portrait gets
# (Z_near ~0.25-0.30 m, far cap 3 m) -- but independent of the file's own
# near/far ends and of the display encoding's constants, so a float file's
# neck/chin detector results cannot move when the display encoding changes.
# Depths nearer than 0.25 m saturate at 255, farther than 3 m are "invalid"
# for the detectors only. Metric values never use this scale.
FLOAT_DETECTOR_NEAR_M = 0.25
FLOAT_DETECTOR_FAR_M = 3.0
FLOAT_DETECTOR_CODE_RANGE = (1.0 / FLOAT_DETECTOR_FAR_M, 1.0 / FLOAT_DETECTOR_NEAR_M)

# Smoothing of float (unfiltered TrueDepth) depth before anything integrates
# along it (surface lengths, profiles, the neck arc). Unfiltered TrueDepth
# depth has ~1 mm per-pixel jitter (residual vs a local quadratic on 11x11
# cheek/forehead patches: 0.8-1.1 mm; Apple-filtered Camera-app depth
# 0.2-0.4 mm) *plus* mid-frequency relief of +-2-4 mm correlated over
# 5-10 mm, which a walked path integrates as fake surface. Calibrated on a
# flat ChArUco board (IMG_2376, PnP-verified, 6 lines of 50-100 mm): with the
# 3x3 median only, surface/linear = 1.09-1.21; with this bilateral filter
# (spatial sigma 6 mm, range sigma 20 mm) 1.010-1.022. Curvature on
# synthetic noise-free ground truth: a r = 40 mm sphere's arc over +-0.8 r
# comes out 2.0 % short (apex moved back 0.7 mm), a r = 60 mm cylinder's
# 0.4 % short; with realistic noise both are within +-1 % on average.
# The spatial sigma is converted to depth pixels with the camera's focal
# length at the subject's distance (``FloatDepthMap.subject_depth_m``, set by
# load_image to the skin-matte median); without a camera
# DEFAULT_INTEGRATION_SIGMA_PX is used (~6 mm at 38 cm).
INTEGRATION_SIGMA_MM = 6.0
# Range sigma: neighbours differing from the centre by several times this
# weigh ~nothing, so a silhouette (a jump of tens of cm) is never blended
# with the background, while the few-mm noise relief is averaged out.
INTEGRATION_RANGE_SIGMA_MM = 20.0
DEFAULT_INTEGRATION_SIGMA_PX = 7.0
# Pixels farther than this are left unsmoothed (background; keeps the
# filter's internal range table fine-grained for the subject).
INTEGRATION_FAR_M = 3.0


@dataclass(frozen=True)
class DepthSample:
    """One neighbourhood-median depth sample.

    :ivar distance_cm: camera distance in cm, or None when the sample holds no
        usable depth (e.g. code 0 of a capture-app 8-bit map)
    :ivar raw: the 8-bit code the legacy map's median picked; always None for
        float maps (they have no codes)
    :ivar code: detector-scale value (see the module docstring); the raw code
        for legacy maps
    """

    distance_cm: float | None
    raw: int | None = None
    code: float | None = None


class DepthMap(ABC):
    """Camera distance at photo pixels, whatever the file's depth format.

    Subclasses: :class:`LegacyDepthMap` (8-bit Camera-app disparity) and
    :class:`FloatDepthMap` (capture-app float metres).
    """

    #: "legacy" or "float"
    kind: str = ""

    def __init__(self, photo_size):
        self.photo_size = (int(photo_size[0]), int(photo_size[1]))

    # -- geometry ---------------------------------------------------------

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of the native depth map."""

    @property
    def is_float(self) -> bool:
        return self.kind == "float"

    def photo_to_depth(self, x, y) -> tuple[float, float]:
        """Continuous native-depth coordinates of photo pixel ``(x, y)``."""
        photo_w, photo_h = self.photo_size
        rows, cols = self.shape
        depth_x = 0.0 if photo_w <= 1 else x * (cols - 1) / (photo_w - 1)
        depth_y = 0.0 if photo_h <= 1 else y * (rows - 1) / (photo_h - 1)
        return depth_x, depth_y

    # -- sampling ---------------------------------------------------------

    @abstractmethod
    def sample(
        self,
        x,
        y,
        radius=1,
        *,
        support_mask=None,
        support_threshold=None,
        inward_y=0,
    ) -> DepthSample | None:
        """Median of the ``(2 * radius + 1)``-square of depth pixels at ``(x, y)``.

        ``support_mask`` / ``support_threshold`` / ``inward_y`` behave as in
        :func:`portrait_analyser.face.sample_depth_at_point`. Returns None
        when the point is outside the photo or no pixel in the window is
        usable.
        """

    def distance_cm(self, x, y, radius=1, **kwargs) -> float | None:
        """Camera distance in cm at photo pixel ``(x, y)`` (neighbourhood median).

        ``radius=0`` reads the single nearest pixel. Keyword arguments are
        passed to :meth:`sample`. None for invalid depth.
        """
        sample = self.sample(x, y, radius, **kwargs)
        return None if sample is None else sample.distance_cm

    @abstractmethod
    def bilinear_cm(self, x, y) -> float | None:
        """Bilinearly interpolated camera distance in cm at a fractional photo
        point; None if any contributing pixel is invalid."""

    @abstractmethod
    def bilinear_code(self, x, y) -> float | None:
        """Bilinearly interpolated detector-scale value (module docstring)."""

    def integration_map(self, camera=None) -> DepthMap:
        """The map every along-path integration samples (see
        :meth:`profile`, :meth:`surface_length_mm`, the neck arc).

        Legacy maps: ``median_filtered(3)`` -- exactly the historical
        pipeline. Float maps: 3x3 NaN-aware median plus a NaN-aware Gaussian
        of :data:`INTEGRATION_SIGMA_MM` (see there), always derived from the
        original unfiltered map, so ``depth.integration_map()`` and
        ``depth.median_filtered().integration_map()`` are the same map.
        """
        return self.median_filtered(3)

    def profile(self, points, *, camera=None) -> list[float | None]:
        """Camera distances (cm, bilinear) at each photo point, None where invalid.

        Samples :meth:`integration_map` for float maps (smoothed for
        integration; ``camera`` sets the physical smoothing scale). Legacy
        maps are sampled as they are -- call it on :meth:`median_filtered`
        for a noise-robust line profile. Points typically come from
        :func:`portrait_analyser.depth_sampling.sample_points_along_line`;
        a step of about one depth pixel is enough.
        """
        return [self.bilinear_cm(x, y) for x, y in points]

    @abstractmethod
    def median_filtered(self, size=3) -> DepthMap:
        """Same-size median-filtered variant of this map (invalid-aware)."""

    # -- whole-map views --------------------------------------------------

    @property
    @abstractmethod
    def valid_mask(self) -> np.ndarray:
        """Boolean ``shape`` array of pixels holding usable depth."""

    @abstractmethod
    def to_cm_array(self) -> np.ndarray:
        """Float ``shape`` array of camera distances in cm, NaN = invalid."""

    @abstractmethod
    def code_array(self) -> np.ndarray:
        """Detector-scale array (higher = nearer, 0 = invalid/background)."""

    @abstractmethod
    def to_display_image(self) -> Image.Image:
        """8-bit image for display only -- never measure with it."""

    # -- metric helpers ---------------------------------------------------

    def point_3d_mm(self, x, y, *, camera=None, radius=1):
        """``(x_mm, y_mm, z_mm)`` of photo pixel ``(x, y)``, or None."""
        z_cm = self.distance_cm(x, y, radius)
        if z_cm is None:
            return None
        photo_w, photo_h = self.photo_size
        xy = point_to_mm(x, y, z_cm, photo_w, photo_h, camera)
        if xy is None:
            return None
        return xy[0], xy[1], z_cm * 10.0

    def distance_3d_mm(self, point_a, point_b, *, camera=None, radius=1):
        """3-D distance between two photo points, each depth a neighbourhood median.

        :returns: ``(distance_mm, z_a_cm, z_b_cm)`` or None
        """
        photo_w, photo_h = self.photo_size
        return distance_3d_from_cm(
            point_a,
            point_b,
            self.distance_cm(point_a[0], point_a[1], radius),
            self.distance_cm(point_b[0], point_b[1], radius),
            photo_w,
            photo_h,
            camera,
        )

    def surface_length_mm(self, points, *, camera=None):
        """3-D polyline length (mm) through photo points, depth read bilinearly.

        Legacy maps: call it on :meth:`median_filtered` (the historical
        pipeline). Float maps always integrate over :meth:`integration_map`,
        whichever variant it is called on. None if fewer than two points,
        any point has invalid depth or is outside the working range.
        """
        photo_w, photo_h = self.photo_size
        return surface_length_mm(self, points, photo_w, photo_h, camera=camera)


def surface_length_mm(depth, points, photo_width, photo_height, *, camera=None):
    """Sum of 3-D segment lengths through ``points`` using ``depth.bilinear_cm``."""
    points_3d = []
    for x, y in points:
        z_cm = depth.bilinear_cm(x, y)
        if z_cm is None:
            return None

        point_mm = point_to_mm(x, y, z_cm, photo_width, photo_height, camera)
        if point_mm is None:
            # Beyond the calibrated range the conversion is meaningless; a
            # partial walk would silently report a shorter surface, so fail
            # the whole measurement instead.
            return None

        points_3d.append((point_mm[0], point_mm[1], z_cm * 10.0))

    if len(points_3d) < 2:
        return None

    return sum(
        vector_length_3d(*point_1, *point_2)
        for point_1, point_2 in itertools.pairwise(points_3d)
    )


class LegacyDepthMap(DepthMap):
    """8-bit disparity image + ``FloatMinValue``/``FloatMaxValue`` (Camera app).

    Samples exactly like the library's historical functions, so results are
    bit-for-bit what they always were:

    * :meth:`sample` = ``sample_depth_at_point`` then
      ``raw_depth_to_distance_cm`` (code 0 excluded/None only when
      ``zero_is_invalid``);
    * :meth:`bilinear_cm` = ``sample_filtered_depth`` (code 0 never
      interpolated) then ``depth_raw_to_distance_cm``;
    * :meth:`median_filtered` = ``median_filter_depthmap``.

    :param image: PIL depth image (``"RGB"`` Camera-app maps use channel 0)
    :param float_min: FloatMinValue (may be None when only codes are needed)
    :param float_max: FloatMaxValue
    :param photo_size: ``(width, height)`` of the full-resolution photo
    :param zero_is_invalid: code 0 means "no depth" (False for Camera-app
        maps, where it is the farthest depth)
    """

    kind = "legacy"

    def __init__(self, image, float_min, float_max, photo_size, *, zero_is_invalid=False):
        super().__init__(photo_size)
        self.image = image
        self.float_min = None if float_min is None else float(float_min)
        self.float_max = None if float_max is None else float(float_max)
        self.zero_is_invalid = bool(zero_is_invalid)

    @property
    def shape(self):
        return self.image.height, self.image.width

    def _to_cm(self, raw):
        if self.float_min is None or self.float_max is None:
            return None
        return raw_depth_to_distance_cm(
            raw, self.float_min, self.float_max, zero_is_invalid=self.zero_is_invalid
        )

    def sample(
        self,
        x,
        y,
        radius=1,
        *,
        support_mask=None,
        support_threshold=None,
        inward_y=0,
    ):
        photo_w, photo_h = self.photo_size
        raw = sample_depth_at_point(
            self.image,
            x,
            y,
            photo_w,
            photo_h,
            kernel_size=2 * int(radius) + 1,
            support_mask=support_mask,
            support_threshold=support_threshold,
            inward_y=inward_y,
            invalid_value=0 if self.zero_is_invalid else None,
        )
        if raw is None:
            return None
        return DepthSample(distance_cm=self._to_cm(raw), raw=raw, code=raw)

    def bilinear_raw(self, x, y):
        """``sample_filtered_depth`` on this image (code 0 = hole, None)."""
        photo_w, photo_h = self.photo_size
        return sample_filtered_depth(self.image, x, y, photo_w, photo_h)

    def bilinear_cm(self, x, y):
        raw = self.bilinear_raw(x, y)
        if raw is None or self.float_min is None or self.float_max is None:
            return None
        return depth_raw_to_distance_cm(raw, self.float_min, self.float_max)

    def bilinear_code(self, x, y):
        return self.bilinear_raw(x, y)

    def median_filtered(self, size=3):
        return LegacyDepthMap(
            median_filter_depthmap(self.image, size=size),
            self.float_min,
            self.float_max,
            self.photo_size,
            zero_is_invalid=self.zero_is_invalid,
        )

    def code_array(self):
        codes = np.array(self.image)
        if codes.ndim == 3:
            codes = codes[:, :, 0]
        return codes

    @property
    def valid_mask(self):
        codes = self.code_array()
        if self.zero_is_invalid:
            return codes > 0
        return np.ones(codes.shape, dtype=bool)

    def to_cm_array(self):
        codes = self.code_array().astype(np.float64)
        if self.float_min is None or self.float_max is None:
            return np.full(codes.shape, np.nan)
        disparity = self.float_max * codes / 255 + self.float_min * (1 - codes / 255)
        with np.errstate(divide="ignore"):
            cm = np.where(disparity != 0, 100.0 / np.where(disparity != 0, disparity, 1), np.nan)
        if self.zero_is_invalid:
            cm[codes == 0] = np.nan
        return cm

    def to_display_image(self):
        return self.image


class FloatDepthMap(DepthMap):
    """Full-precision depth in metres (capture-app files), NaN = invalid.

    Values that are non-finite, ``<= 0`` or above
    :data:`~portrait_analyser.apple_depth.MAX_PLAUSIBLE_DEPTH_M` (Apple's
    near-zero-disparity sentinels) are invalid. There is no far cap: a wall
    at 5 m is a valid 500 cm. Samples never read the 8-bit display image.

    :param depth_m: ``(rows, cols)`` metres, upright (aligned with the photo)
    :param photo_size: ``(width, height)`` of the full-resolution photo

    The detector scale (``code_array`` etc.) uses the fixed
    :data:`FLOAT_DETECTOR_CODE_RANGE`, never the file's own range.
    """

    kind = "float"
    code_range = FLOAT_DETECTOR_CODE_RANGE

    def __init__(self, depth_m, photo_size):
        super().__init__(photo_size)
        depth = np.array(depth_m, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError(f"depth_m must be a 2-D array, got shape {depth.shape}")
        with np.errstate(invalid="ignore"):
            invalid = ~np.isfinite(depth) | (depth <= 0) | (depth > MAX_PLAUSIBLE_DEPTH_M)
        depth[invalid] = np.nan
        depth.setflags(write=False)
        self.depth_m = depth
        self._display = None
        # The unfiltered map this one was derived from (None = this is it),
        # and the smoothing applied: None, "median", or ("integration", px).
        self._source = None
        self.smoothing = None
        self._integration_cache = {}
        # Subject (face) distance in metres for the smoothing scale; set by
        # load_image, None = estimate from the map centre.
        self.subject_depth_m = None

    @property
    def shape(self):
        return self.depth_m.shape

    def _codes(self, z_m):
        """Detector-scale value(s) for depth(s) in metres: the unquantised
        8-bit code on :data:`FLOAT_DETECTOR_CODE_RANGE`, saturating at 1..255
        like an 8-bit encoding; NaN beyond its far end or invalid."""
        z_m = np.atleast_1d(np.asarray(z_m, dtype=np.float64))
        float_min, float_max = self.code_range
        far_limit = 1.0 / float_min
        with np.errstate(divide="ignore", invalid="ignore"):
            codes = 255.0 * (1.0 / z_m - float_min) / (float_max - float_min)
            codes = np.clip(codes, 1.0, 255.0)
            codes[~np.isfinite(z_m) | (z_m > far_limit * (1 + 1e-9))] = np.nan
        return codes

    def _code(self, z_m):
        if z_m is None or not math.isfinite(z_m):
            return None
        code = float(self._codes(z_m)[0])
        return None if math.isnan(code) else code

    def _window_values(self, depth_x, depth_y, radius, support_mask, support_threshold):
        rows, cols = self.shape
        values = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sx = depth_x + dx
                sy = depth_y + dy
                if not (0 <= sx < cols and 0 <= sy < rows):
                    continue
                if support_mask is not None:
                    mask_x = 0 if cols == 1 else round(sx * (support_mask.width - 1) / (cols - 1))
                    mask_y = 0 if rows == 1 else round(sy * (support_mask.height - 1) / (rows - 1))
                    mask_value = support_mask.getpixel((mask_x, mask_y))
                    if isinstance(mask_value, tuple):
                        mask_value = mask_value[0]
                    if mask_value < support_threshold:
                        continue
                value = self.depth_m[sy, sx]
                if np.isfinite(value):
                    values.append(float(value))
        return values

    def sample(
        self,
        x,
        y,
        radius=1,
        *,
        support_mask=None,
        support_threshold=None,
        inward_y=0,
    ):
        radius = int(radius)
        if radius < 0:
            raise ValueError("radius must be >= 0")
        photo_w, photo_h = self.photo_size
        rows, cols = self.shape
        if photo_w < 1 or photo_h < 1 or rows < 1 or cols < 1:
            return None
        if not 0 <= x <= photo_w - 1 or not 0 <= y <= photo_h - 1:
            return None
        # Same nearest-pixel mapping as sample_depth_at_point.
        depth_x = 0 if photo_w == 1 else round(x * (cols - 1) / (photo_w - 1))
        depth_y = 0 if photo_h == 1 else round(y * (rows - 1) / (photo_h - 1))
        depth_y += int(inward_y)
        if support_mask is not None and support_threshold is None:
            support_threshold = teeth_threshold(support_mask)
        values = self._window_values(depth_x, depth_y, radius, support_mask, support_threshold)
        if not values:
            return None
        # An even count gives the mean of the two middle values; the legacy
        # 8-bit sampler picks the upper-middle code instead. Float maps have
        # no legacy numbers to reproduce, so the textbook median is used.
        z_m = float(np.median(values))
        return DepthSample(distance_cm=z_m * 100.0, raw=None, code=self._code(z_m))

    def _bilinear_m(self, x, y):
        rows, cols = self.shape
        if rows == 0 or cols == 0:
            return None
        depth_x, depth_y = self.photo_to_depth(x, y)
        depth_x = min(max(float(depth_x), 0.0), cols - 1.0)
        depth_y = min(max(float(depth_y), 0.0), rows - 1.0)
        x0 = math.floor(depth_x)
        y0 = math.floor(depth_y)
        x1 = min(x0 + 1, cols - 1)
        y1 = min(y0 + 1, rows - 1)
        fx = depth_x - x0
        fy = depth_y - y0
        total = 0.0
        for value, weight in (
            (self.depth_m[y0, x0], (1.0 - fx) * (1.0 - fy)),
            (self.depth_m[y0, x1], fx * (1.0 - fy)),
            (self.depth_m[y1, x0], (1.0 - fx) * fy),
            (self.depth_m[y1, x1], fx * fy),
        ):
            if weight == 0:
                continue
            if not np.isfinite(value):
                # Never interpolate across a hole.
                return None
            total += float(value) * weight
        return total

    def bilinear_cm(self, x, y):
        z_m = self._bilinear_m(x, y)
        return None if z_m is None else z_m * 100.0

    def bilinear_code(self, x, y):
        return self._code(self._bilinear_m(x, y))

    def median_filtered(self, size=3):
        """NaN-aware ``size`` x ``size`` median.

        Each output pixel is the median of the valid pixels in its window,
        provided at least half of the window's in-image pixels are valid;
        otherwise NaN (mirroring the 8-bit filter, where a majority of
        code-0 holes yields 0).
        """
        if size < 3 or size % 2 == 0:
            raise ValueError("median filter size must be an odd number >= 3")
        half = size // 2
        rows, cols = self.shape
        padded = np.pad(self.depth_m, half, constant_values=np.nan)
        inside = np.pad(np.ones(self.shape, dtype=np.int32), half, constant_values=0)
        windows = np.stack(
            [padded[dy : dy + rows, dx : dx + cols] for dy in range(size) for dx in range(size)]
        )
        in_image = sum(
            inside[dy : dy + rows, dx : dx + cols] for dy in range(size) for dx in range(size)
        )
        valid_count = np.isfinite(windows).sum(axis=0)
        keep = valid_count * 2 >= in_image
        # nanmedian warns on all-NaN windows; those become NaN below anyway.
        windows[:, valid_count == 0] = 0.0
        filtered = np.nanmedian(windows, axis=0)
        filtered[~keep | (valid_count == 0)] = np.nan
        filtered_map = FloatDepthMap(filtered, self.photo_size)
        filtered_map._source = self._root()
        filtered_map.smoothing = "median"
        return filtered_map

    def _root(self):
        return self if self._source is None else self._source

    def subject_distance_m(self):
        """Distance of the measured subject in metres: :attr:`subject_depth_m`
        when set (load_image: the skin-matte median), else the median valid
        depth in the central third of the map. Never a whole-map percentile,
        so a near foreground cannot shrink the smoothing."""
        root = self._root()
        if root.subject_depth_m is not None:
            return float(root.subject_depth_m)
        rows, cols = root.shape
        centre = root.depth_m[rows // 3 : 2 * rows // 3, cols // 3 : 2 * cols // 3]
        valid = centre[np.isfinite(centre)]
        if valid.size == 0:
            valid = root.depth_m[np.isfinite(root.depth_m)]
        return float(np.median(valid)) if valid.size else None

    def integration_sigma_px(self, camera=None):
        """Spatial sigma in depth pixels for :data:`INTEGRATION_SIGMA_MM`:
        ``sigma_mm * f_depth / Z_subject`` (``f_depth`` = the camera's focal
        length in depth pixels, ``Z_subject`` = :meth:`subject_distance_m`);
        :data:`DEFAULT_INTEGRATION_SIGMA_PX` without a camera."""
        z_subject = self.subject_distance_m()
        if camera is None or z_subject is None:
            return DEFAULT_INTEGRATION_SIGMA_PX
        focal_depth_px = camera.fx * self.shape[1] / self.photo_size[0]
        return INTEGRATION_SIGMA_MM * focal_depth_px / (z_subject * 1000.0)

    def integration_map(self, camera=None):
        if isinstance(self.smoothing, tuple):
            return self
        root = self._root()
        sigma_px = round(root.integration_sigma_px(camera), 3)
        cached = root._integration_cache.get(sigma_px)
        if cached is None:
            median = root.median_filtered(3)
            cached = FloatDepthMap(_nan_bilateral(median.depth_m, sigma_px), root.photo_size)
            cached.subject_depth_m = root.subject_depth_m
            cached._source = root
            cached.smoothing = ("integration", sigma_px)
            root._integration_cache[sigma_px] = cached
        return cached

    def profile(self, points, *, camera=None):
        smooth = self.integration_map(camera)
        return [smooth.bilinear_cm(x, y) for x, y in points]

    def surface_length_mm(self, points, *, camera=None):
        smooth = self.integration_map(camera)
        photo_w, photo_h = self.photo_size
        return surface_length_mm(smooth, points, photo_w, photo_h, camera=camera)

    @property
    def valid_mask(self):
        return np.isfinite(self.depth_m)

    def to_cm_array(self):
        return self.depth_m * 100.0

    def code_array(self):
        codes = self._codes(self.depth_m.ravel()).reshape(self.shape)
        return np.where(np.isnan(codes), 0.0, codes).astype(np.float32)

    def display_encoding(self):
        """``(image, float_min, float_max)`` of the display-only 8-bit encoding."""
        if self._display is None:
            self._display = encode_depth_as_disparity_8bit(self.depth_m)
        return self._display

    def to_display_image(self):
        return self.display_encoding()[0]


def _nan_bilateral(depth, sigma_px, sigma_range_m=None, far_m=None):
    """NaN-aware bilateral smoothing of a metres map (OpenCV).

    Spatial Gaussian of ``sigma_px`` (window radius 2 sigma) times a range
    Gaussian of ``sigma_range_m`` on the depth difference to the centre
    pixel. Invalid pixels and pixels beyond ``far_m`` are replaced by a
    sentinel 50 range-sigmas beyond ``far_m`` (zero weight for the subject),
    and get their original value (NaN / unsmoothed) back afterwards.
    """
    import cv2

    if sigma_range_m is None:
        sigma_range_m = INTEGRATION_RANGE_SIGMA_MM / 1000.0
    if far_m is None:
        far_m = INTEGRATION_FAR_M
    if sigma_px <= 0:
        return depth.copy()
    valid = np.isfinite(depth)
    with np.errstate(invalid="ignore"):
        smoothable = valid & (depth <= far_m)
    sentinel = far_m + 50.0 * sigma_range_m
    work = np.where(smoothable, depth, sentinel).astype(np.float32)
    # OpenCV tabulates the range kernel over the image's own min..max; an
    # extra row pinned to 0 and the sentinel makes that table (and so the
    # result) independent of the file's nearest/farthest pixel. The row is far
    # outside every subject pixel's range kernel.
    anchor = np.full((1, work.shape[1]), sentinel, dtype=np.float32)
    anchor[0, 0] = 0.0
    work = np.vstack([work, anchor])
    radius = max(1, math.ceil(2.0 * sigma_px))
    filtered = cv2.bilateralFilter(work, 2 * radius + 1, sigma_range_m, sigma_px)[:-1]
    result = np.where(smoothable, filtered.astype(np.float64), depth)
    result[~valid] = np.nan
    return result
