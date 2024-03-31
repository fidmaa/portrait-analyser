from typing import Union

import piexif
import pyheif
from PIL import Image, ImageFilter, ImageOps
from pyheif import UndecodedHeifImage

from . import const
from .exceptions import ExifValidationFailed, NoDepthMapFound, UnknownExtension
from .face import find_bounding_box_teeth


class IOSPortrait:
    def __init__(
        self,
        photo,
        depthmap=None,
        teethmap=None,
        floatValueMin=None,
        floatValueMax=None,
    ):
        self.photo = photo
        self.depthmap = depthmap
        self.teethmap = teethmap

        self.floatValueMin = floatValueMin
        self.floatValueMax = floatValueMax

        if self.floatValueMin is not None:
            self.floatValueMin = float(self.floatValueMin)

        if self.floatValueMax is not None:
            self.floatValueMax = float(self.floatValueMax)

        self.teeth_bbox = None
        if self.teethmap:
            self.teethmap = ImageOps.mirror(self.teethmap)
            self.teethmap = self.teethmap.resize(self.photo.size)
            ret = find_bounding_box_teeth(
                self.teethmap.filter(ImageFilter.MaxFilter(size=13))
            )
            if ret is not None:
                self.teeth_bbox = ret

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


def load_image(fileName: str) -> Union[IOSPortrait, None]:
    """
    Load HEIF or JPEG with depth data,
    return tuple of (image, depth)
    """
    if fileName.lower().endswith("heic") or fileName.lower().endswith("heif"):
        #
        # Get depth map from HEIC/HEIF container, then proceed normally:
        #
        heif_container = pyheif.open_container(open(fileName, "rb"))

        primary_image = heif_container.primary_image

        for exif_metadata in [
            metadata
            for metadata in primary_image.image.load().metadata
            if metadata.get("type", "") == "Exif"
        ]:
            exif = piexif.load(exif_metadata["data"])
            check_exif_data(exif)

        if primary_image.depth_image is None:
            raise NoDepthMapFound(f"{fileName} has no depth data")

        teeth_image = None
        for looking_for in primary_image.auxiliary_images:
            if (
                getattr(looking_for, "type", "")
                == "urn:com:apple:photo:2019:aux:semanticteethmatte"
            ):
                teeth_image = looking_for.image  # .load()
                break

        depth_image = primary_image.depth_image.image.load()
        ret = {}
        if depth_image.metadata:
            for metadata in depth_image.metadata:
                if metadata.get("type", "") == "mime":
                    import xml.etree.ElementTree as ET

                    root = ET.fromstring(metadata.get("data"))
                    for elem in root[0][0]:
                        ret[elem.tag] = elem.text

        float_min_value = ret.get(
            "{http://ns.apple.com/pixeldatainfo/1.0/}FloatMinValue", 0.0
        )

        float_max_value = ret.get(
            "{http://ns.apple.com/pixeldatainfo/1.0/}FloatMaxValue", 0.0
        )

        depth_image = Image.frombytes(
            depth_image.mode, depth_image.size, depth_image.data
        )

        picture_image = primary_image.image.load()
        try:
            # iPhone 14
            picture_image = Image.frombytes(
                picture_image.mode,
                (picture_image.size[0] + 4, picture_image.size[1] - 1),
                picture_image.data,
            )
        except ValueError:
            # iPhone 12; probably flagged somewhere in the image; no idea, will just
            # leave this as-is.
            picture_image = Image.frombytes(
                picture_image.mode,
                (picture_image.size[0], picture_image.size[1]),
                picture_image.data,
            )
        #     int aux_bit_depth = heif_image_handle_get_luma_bits_per_pixel(aux_handle);
        #
        #             struct heif_image* aux_image;
        #             err = heif_decode_image(aux_handle,
        #                                     &aux_image,
        #                                     encoder->colorspace(false),
        #                                     encoder->chroma(false, aux_bit_depth),
        #                                     nullptr);

        if teeth_image:
            teeth_image: UndecodedHeifImage

            teeth_image = teeth_image.load()
            teeth_image = ImageOps.mirror(
                Image.frombytes(
                    "L",
                    (teeth_image.size[0] * 3 + 14, teeth_image.size[1] - 1),
                    teeth_image.data,
                )
            )
            # teeth_image = teeth_image.scale(self.photo.size)

        return IOSPortrait(
            picture_image,
            depth_image,
            teeth_image,
            float_min_value,
            float_max_value,
        )

    else:
        raise UnknownExtension(
            "only supported extensions for filenames are: HEIF, HEIC"
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
        except BaseException:
            reason = "cannot encode"
    else:
        ret = -1

    if ret == -1:
        raise ExifValidationFailed(reason)
