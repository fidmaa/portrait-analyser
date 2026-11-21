TRUEDEPTH_EXIF_ID = "front TrueDepth"

# EXIF ``Model`` values whose 16-bit depth, as read by ``heif_depth`` (the
# reader used where Apple's ImageIO/AVFoundation is unavailable), has been
# compared pixel by pixel with Apple's reader -- see
# ``scripts/compare_depth_readers.py``. 97/97 photos from an iPhone 17 Pro
# (iOS 26.3.1) agree. ``load_image`` refuses other models on that path unless
# called with ``allow_unverified_device=True``.
HEIF_DEPTH_VERIFIED_MODELS = frozenset({"iPhone 17 Pro"})
