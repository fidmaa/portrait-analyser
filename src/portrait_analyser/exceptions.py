class ExifValidationFailed(Exception):
    pass


class NoDepthMapFound(Exception):
    pass


class UnverifiedDepthDevice(NoDepthMapFound):
    """The photo's 16-bit depth can only be read by the imagecodecs reader
    here, and that reader has not been checked against Apple's for the
    device that took the photo.

    Raised by ``load_image`` where the macOS ImageIO/AVFoundation reader is
    unavailable (Linux, Windows, macOS without pyobjc) and the EXIF model is
    not in ``const.HEIF_DEPTH_VERIFIED_MODELS``. Another model may store its
    depth differently, so the map could be decoded into wrong distances.
    Pass ``allow_unverified_device=True`` to ``load_image`` to load anyway.

    A subclass of :class:`NoDepthMapFound`, so callers that only handle that
    still report the photo as unreadable.

    :ivar model: EXIF ``Model`` of the photo, or ``None`` when absent.
    :ivar software: EXIF ``Software`` (the iOS version), or ``None``.
    """

    def __init__(self, message, model=None, software=None):
        super().__init__(message)
        self.model = model
        self.software = software


class UnknownExtension(Exception):
    pass


class NoFacesDetected(Exception):
    pass


class MultipleFacesDetected(Exception):
    pass


class AppleDepthUnavailable(Exception):
    """The macOS ImageIO/AVFoundation depth reader cannot run here.

    Raised when not running on macOS, or when the pyobjc frameworks
    (pyobjc-framework-Quartz / pyobjc-framework-AVFoundation) required to
    call into ImageIO/AVFoundation are not importable.
    """


class AppleDepthDecodeError(Exception):
    """macOS failed to decode an embedded Apple depth/disparity image.

    Distinct from :func:`portrait_analyser.apple_depth.read_apple_depth`
    returning ``None`` (which means the file simply has no depth/disparity
    auxiliary image at all): this means aux data was present but
    ImageIO/AVFoundation could not turn it into usable depth.
    """


class AppleVisionUnavailable(Exception):
    """Apple Vision cannot run here: not macOS, or pyobjc-framework-Vision /
    pyobjc-framework-Quartz are not importable."""


class AppleVisionError(Exception):
    """An Apple Vision request failed or returned data of an unexpected shape."""
