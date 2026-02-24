import xml.etree.ElementTree as ET
from typing import Union

import piexif
import pyheif
from PIL import Image

from . import const
from .exceptions import ExifValidationFailed, NoDepthMapFound, UnknownExtension
from .face import find_bounding_box_teeth, find_incisor_distance_teeth


class IOSPortrait:
    def __init__(
        self,
        photo,
        depthmap=None,
        teethmap=None,
        skinmap=None,
        floatValueMin=None,
        floatValueMax=None,
        teeth_bbox=None,
        incisor_distance=None,
    ):
        self.photo = photo
        self.depthmap = depthmap
        self.teethmap = teethmap
        self.skinmap = skinmap
        self.teeth_bbox = teeth_bbox
        self.incisor_distance = incisor_distance
        self.floatValueMin = float(floatValueMin) if floatValueMin is not None else None
        self.floatValueMax = float(floatValueMax) if floatValueMax is not None else None

    def teeth_bbox_translated(self, max_wi, max_he):
        if self.teeth_bbox is None:
            return
        x, y, wi, he = self.teeth_bbox
        return (
            x * max_wi / self.teethmap.size[0],
            y * max_he / self.teethmap.size[1],
            wi * max_wi / self.teethmap.size[0],
            he * max_he / self.teethmap.size[1],
        )


def _validate_exif(primary_image, use_exif):
    """Extract and validate TrueDepth EXIF metadata."""
    for exif_metadata in [
        metadata
        for metadata in primary_image.image.load().metadata
        if metadata.get("type", "") == "Exif"
    ]:
        exif = piexif.load(exif_metadata["data"])
        if use_exif:
            check_exif_data(exif)


def _parse_depth_metadata(depth_image):
    """Parse XML metadata from depth image for float min/max values."""
    ret = {}
    if depth_image.metadata:
        for metadata in depth_image.metadata:
            if metadata.get("type", "") == "mime":
                root = ET.fromstring(metadata.get("data"))
                for elem in root[0][0]:
                    ret[elem.tag] = elem.text

    float_min = ret.get(
        "{http://ns.apple.com/pixeldatainfo/1.0/}FloatMinValue", 0.0
    )
    float_max = ret.get(
        "{http://ns.apple.com/pixeldatainfo/1.0/}FloatMaxValue", 0.0
    )
    return float_min, float_max


def _decode_picture(raw_image):
    """Decode primary picture with device-specific dimension correction."""
    try:
        # iPhone 14
        return Image.frombytes(
            raw_image.mode,
            (raw_image.size[0] + 4, raw_image.size[1] - 1),
            raw_image.data,
        )
    except ValueError:
        # iPhone 12
        return Image.frombytes(
            raw_image.mode,
            (raw_image.size[0], raw_image.size[1]),
            raw_image.data,
        )


def _decode_semantic_map(raw_image):
    """Decode a semantic segmentation map (teeth/skin) using the Apple format."""
    loaded = raw_image.load()
    try:
        return Image.frombytes(
            "L",
            (loaded.size[0] * 3 + 14, loaded.size[1] - 1),
            loaded.data,
        )
    except ValueError:
        return None


def load_image(fileName: str, use_exif=True) -> Union[IOSPortrait, None]:
    """Load HEIC/HEIF with depth data, return an IOSPortrait instance."""
    if not (fileName.lower().endswith("heic") or fileName.lower().endswith("heif")):
        raise UnknownExtension(
            "only supported extensions for filenames are: HEIF, HEIC"
        )

    with open(fileName, "rb") as f:
        heif_container = pyheif.open_container(f)

        primary_image = heif_container.primary_image
        _validate_exif(primary_image, use_exif)

        if primary_image.depth_image is None:
            raise NoDepthMapFound(f"{fileName} has no depth data")

        # Extract auxiliary semantic maps
        teeth_raw = skin_raw = None
        for aux in primary_image.auxiliary_images:
            aux_type = getattr(aux, "type", "")
            if aux_type == "urn:com:apple:photo:2019:aux:semanticteethmatte":
                teeth_raw = aux.image
            elif aux_type == "urn:com:apple:photo:2019:aux:semanticskinmatte":
                skin_raw = aux.image

        # Decode depth map
        depth_loaded = primary_image.depth_image.image.load()
        float_min, float_max = _parse_depth_metadata(depth_loaded)
        depth_image = Image.frombytes(
            depth_loaded.mode, depth_loaded.size, depth_loaded.data
        )

        # Decode primary picture
        picture_image = _decode_picture(primary_image.image.load())

        # Decode semantic maps
        teeth_image = _decode_semantic_map(teeth_raw) if teeth_raw else None
        skin_image = _decode_semantic_map(skin_raw) if skin_raw else None

    # Process teeth map: resize and analyze
    teeth_bbox = None
    incisor_distance = None
    if teeth_image is not None:
        teeth_image = teeth_image.resize(picture_image.size)
        teeth_bbox = find_bounding_box_teeth(teeth_image)
        if teeth_bbox is not None:
            incisor_distance = find_incisor_distance_teeth(teeth_image, teeth_bbox)

    # Process skin map: resize
    if skin_image is not None:
        skin_image = skin_image.resize(picture_image.size)

    return IOSPortrait(
        photo=picture_image,
        depthmap=depth_image,
        teethmap=teeth_image,
        skinmap=skin_image,
        floatValueMin=float_min,
        floatValueMax=float_max,
        teeth_bbox=teeth_bbox,
        incisor_distance=incisor_distance,
    )


def check_exif_data(exif):
    data = exif.get("Exif", {})
    data = data.get(42036, "default")

    reason = ""

    if isinstance(data, str):
        ret = data.find(const.TRUEDEPTH_EXIF_ID)
    elif isinstance(data, bytes):
        ret = data.find(const.TRUEDEPTH_EXIF_ID.encode("ascii"))
        try:
            reason = data.decode("ascii")
        except Exception:
            reason = "cannot encode"
    else:
        ret = -1

    if ret == -1:
        raise ExifValidationFailed(reason)
