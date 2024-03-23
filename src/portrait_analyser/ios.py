from io import BytesIO
from typing import List, Tuple

import piexif
import pyheif
from PIL import Image

from . import const
from .exceptions import ExifValidationFailed, NoDepthMapFound, UnknownExtension


class IOSImage:
    # Adapted from https://github.com/lyangas/depthmap_from_jpeg/tree/master
    # Class for opening JPEGs with depth information, for iPhone X/Xs

    def __init__(self, file_name):
        # every layer starts with 0xff, 0xd8 bytes and ends with 0xff, 0xd9
        # bytes
        self.start_bytes = [0xFF, 0xD8]
        self.end_bytes = [0xFF, 0xD9]

        with open(file_name, "rb") as f:
            self.byte_str = f.read()

        self.layers = self.extract_layers()
        self.depthmap = self.layers[2]
        self.image = self.layers[0]

    def extract_layers(self):
        """
        extract all layers from image file
        """

        # get all layers as byte-string
        image_cnt, old_byte = 0, 0

        layers_bytes = {}
        stack_of_images = []
        for byte in self.byte_str:
            if (old_byte == self.start_bytes[0]) and (byte == self.start_bytes[1]):
                stack_of_images.append(image_cnt)
                layers_bytes[image_cnt] = [old_byte]
                image_cnt += 1

            for image_ind in stack_of_images:
                layers_bytes[image_ind] += [byte]

            if (old_byte == self.end_bytes[0]) and (byte == self.end_bytes[1]):
                stack_of_images.pop()

            old_byte = byte

        # transform all layers to np.ayrray's
        layers_arrays = []
        for _, layer_bytes in layers_bytes.items():
            layers_arrays.append(self.bytes_to_arr(layer_bytes))

        return layers_arrays

    def bytes_to_arr(self, byte_str):
        """
        transform byte-string with image to numpy-array
        """

        depthmap_bytes = bytes(byte_str)

        stream = BytesIO()
        stream.write(depthmap_bytes)

        depthmap = Image.open(stream)
        # depthmap_arr = np.array(depthmap)

        stream.close()
        return depthmap


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
        picture_image = Image.frombytes(
            picture_image.mode,
            (picture_image.size[0] + 4, picture_image.size[1] - 1),
            picture_image.data,
        )

        return (picture_image, depth_image)

    elif fileName.lower().endswith("jpg") or fileName.lower().endswith("jpeg"):
        # If there's no EXIF data, that's no problem. But if there is EXIF,
        # check it:

        exif = None
        try:
            exif = piexif.load(fileName)
        except piexif.InvalidImageDataError:
            pass

        if exif is not None:
            check_exif_data(exif)

        # Let's load the magin image

        img = IOSImage(fileName)
        ret = img.extract_layers()
        return ret[0], ret[1]

    else:
        raise UnknownExtension(
            "only supported extensions for filenames are: HEIF, HEIC, JPG, JPEG"
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
