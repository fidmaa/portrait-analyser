class ExifValidationFailed(Exception):
    pass


class NoDepthMapFound(Exception):
    pass


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
