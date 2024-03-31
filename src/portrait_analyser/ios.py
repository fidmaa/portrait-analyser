from io import BytesIO
from typing import List, Tuple

import piexif
import pyheif
from PIL import Image

from . import const
from .exceptions import ExifValidationFailed, NoDepthMapFound, UnknownExtension


def load_image(fileName: str) -> Tuple[Image]:
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

        depth_image = primary_image.depth_image.image.load()
        depth_image = Image.frombytes(depth_image.mode, depth_image.size, depth_image.data)

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
