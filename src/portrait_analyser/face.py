import os
from dataclasses import dataclass

import cv2
import numpy
from PIL import Image

from .exceptions import MultipleFacesDetected, NoFacesDetected


@dataclass
class IncisorMeasurement:
    upper_centroid: tuple[float, float]  # (x, y) in photo/teethmap coordinates
    lower_centroid: tuple[float, float]  # (x, y) in photo/teethmap coordinates
    upper_depth_raw: int | None = None  # raw pixel value from depth map
    lower_depth_raw: int | None = None
    upper_distance_cm: float | None = None  # physical distance from camera (cm)
    lower_distance_cm: float | None = None
    distance_3d_mm: float | None = None  # 3D Euclidean distance between centroids
    pixel_distance_y: float = 0.0  # legacy-style vertical pixel gap

face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.__path__[0], "data/haarcascade_frontalface_default.xml")
)

eye_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.__path__[0], "data/haarcascade_eye_tree_eyeglasses.xml")
)


def Image_to_cv2(pil_image):
    return cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)


def translate_coordinates(
    rect, new_max_width, new_max_height, image_width, image_height
):
    mul_x = new_max_width / image_width
    mul_y = new_max_height / image_height

    new_x1 = rect.x * mul_x
    new_y1 = rect.y * mul_y
    new_x2 = (rect.x + rect.width) * mul_x
    new_y2 = (rect.y + rect.height) * mul_y

    new_width = new_x2 - new_x1
    new_height = new_y2 - new_y1

    return (new_x1, new_y1, new_width, new_height)


class Rectangle:
    def __init__(self, x, y, wi, he):
        self.x = x
        self.y = y
        self.width = wi
        self.height = he

        self.center_x = x + wi / 2
        self.center_y = y + he / 2

    def __str__(self):
        return f"{id(self)} at {self.x}:{self.y}, {self.width}x{self.height}"

    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        elif i == 2:
            return self.width
        elif i == 3:
            return self.height
        else:
            raise IndexError


class Eye(Rectangle):
    def __init__(self, face, *args, **kw):
        super().__init__(*args, **kw)
        self.face = face

    def translate_coordinates(self, max_wi, max_he):
        rect = self.face.translate_coordinates(max_wi, max_he)
        self_rect = translate_coordinates(
            self, max_wi, max_he, self.face.image.size[0], self.face.image.size[1]
        )
        return (
            rect[0] + self_rect[0],
            rect[1] + self_rect[1],
            self_rect[2],
            self_rect[3],
        )

    def get_image_for_analysis(self, mode):
        ret = Image_to_cv2(self.face.image)
        if mode:
            ret = cv2.cvtColor(ret, mode)
        return ret[
            self.face.y + self.y : self.y + self.face.y + self.height,
            self.x + self.face.x : self.x + self.face.x + self.width,
        ].copy()


class Face(Rectangle):
    def __init__(self, image, *args, **kw):
        super().__init__(*args, **kw)
        self.image = image
        self.find_eyes()

    def translate_coordinates(self, new_max_width, new_max_height):
        return translate_coordinates(
            self, new_max_width, new_max_height, self.image.size[0], self.image.size[1]
        )

    def calculate_percentage_of_image(self):
        """How much of the image is the face?"""

        img_wi, img_he = self.image.size  # [:2]
        percent_width = float(self.width) / float(img_wi)
        percent_height = float(self.height) / float(img_he)

        return percent_width, percent_height

    def get_image_for_analysis(self, mode=cv2.COLOR_BGR2GRAY):
        ret = Image_to_cv2(self.image)
        if mode:
            ret = cv2.cvtColor(ret, mode)
        return ret[self.y : self.y + self.height - 1, self.x : self.x + self.width - 1]

    def find_eyes(self):
        gray = self.get_image_for_analysis()

        # detects eyes of within the detected face area (roi)
        eyes = eye_cascade.detectMultiScale(gray)

        self.eyes = []
        # draw a rectangle around eyes
        for ex, ey, ew, eh in eyes:
            self.eyes.append(Eye(self, ex, ey, ew, eh))


def get_face_parameters(input_image: Image, raise_opencv_exceptions=False):
    """Get face position and size or return an exception in
    case there's none."""

    # image = numpy.asarray(input_image.convert("RGB"))
    image = Image_to_cv2(input_image)

    try:
        face = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    except Exception as e:
        if raise_opencv_exceptions:
            raise e
        face = []

    #
    # Update midline point to match detected face coords
    #

    if len(face) == 1:
        x, y, wi, he = face[0]
        return Face(input_image, x, y, wi, he)

    elif len(face) == 0:
        raise NoFacesDetected()

    else:
        raise MultipleFacesDetected()


def find_neck_measurement_point(skinmap, face_location, threshold=200):
    """Find the most narrow line horizontal from the bottom to the top, to the lower
    boundary of the face"""

    # WARNING: this asserts that skinmap and the photo that the face_location
    # bounding box was taken from has the SAME size

    y = face_location[1] + face_location[3]
    results = []

    while y < skinmap.size[1]:
        start_x, end_x = None, None
        for x in range(face_location[0], face_location[0] + face_location[2]):
            if skinmap.getpixel((x, y)) >= threshold:
                if start_x is None:
                    start_x = x
                    continue

                end_x = x

        if end_x is None or start_x is None:
            continue

        distance = end_x - start_x
        if results:
            if distance > results[-1][0]:
                break

        results.append((distance, y, start_x, end_x))
        y += 1

    results.sort()

    res = results[0]
    return (res[2], res[1], res[3], res[1])


def find_bounding_box_teeth(teethmap, margin_x=100, margin_y=100, min_value=200):
    min_teeth_x = None
    min_teeth_y = None
    max_teeth_x = None
    max_teeth_y = None

    for y in range(margin_y, teethmap.size[1] - margin_y):
        for x in range(margin_x, teethmap.size[0] - margin_x):
            if teethmap.getpixel((x, y)) > min_value:
                if min_teeth_x is None or min_teeth_x > x:
                    min_teeth_x = x
                if max_teeth_x is None or max_teeth_x < x:
                    max_teeth_x = x

                if min_teeth_y is None or min_teeth_y > y:
                    min_teeth_y = y
                if max_teeth_y is None or max_teeth_y < y:
                    max_teeth_y = y

    if max_teeth_y == teethmap.size[1] - margin_y - 1:
        # bottom not found!
        return

    if min_teeth_x is None:
        return

    if max_teeth_y - min_teeth_y < 200:
        return

    return (
        min_teeth_x,
        min_teeth_y,
        max_teeth_x - min_teeth_x,
        max_teeth_y - min_teeth_y,
    )


def find_incisor_distance_teeth(
    teethmap, bounding_box_teeth, threshold=200, margin_x=0.25
):
    """Iterate from teethmap x1 to x1 + width, starting from the half of it,
    try finding points with MAXIMAL distance as long as they are within bounding_box_teeth
    and their value is above 200 (well-detected teeth, to avoid diasthemes which would probably
    be the highest distance points, but that's not what we're looking for...)"""

    y_mid = bounding_box_teeth[1] + bounding_box_teeth[3] / 2
    min_he = bounding_box_teeth[1]
    max_he = bounding_box_teeth[1] + bounding_box_teeth[3]

    x_start = int(
        bounding_box_teeth[0]
        + bounding_box_teeth[2] / 2
        - margin_x * bounding_box_teeth[2] / 2
    )
    x_end = int(
        bounding_box_teeth[0]
        + bounding_box_teeth[2] / 2
        + margin_x * bounding_box_teeth[2] / 2
    )

    found_values = []
    for x in range(x_start, x_end):
        upper_y, lower_y = y_mid, y_mid

        while upper_y > min_he:
            upper_y -= 1
            value = teethmap.getpixel((x, upper_y))
            if value >= threshold:
                break

        if upper_y <= min_he:
            # No upper teeth found!
            continue

        while lower_y < max_he:
            lower_y += 1
            value = teethmap.getpixel((x, lower_y))
            if value >= threshold:
                break

        if lower_y >= max_he:
            # No lower teeth found!
            continue

        distance = lower_y - upper_y
        found_values.append((distance, x, upper_y, lower_y))

    if not found_values:
        return

    found_values.sort()
    _, x, y1, y2 = found_values.pop()
    return (x, y1, x, y2)


def _pick_side(xs, ys, x_midline, use_left):
    """Keep only the left or right half of pixel coordinates.

    Used to avoid the septum (dark gap between central incisors).
    The caller decides which side to use so that upper and lower
    centroids are on the same side (left-to-left or right-to-right).
    """
    if use_left:
        mask = xs < x_midline
    else:
        mask = xs >= x_midline
    return xs[mask], ys[mask]


def find_incisor_centroids(
    teethmap,
    bounding_box_teeth,
    threshold=200,
    margin_x=0.25,
    min_pixels=50,
    centroid_margin_x=0.35,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Find centroids of upper and lower incisor surfaces.

    Returns ((upper_cx, upper_cy), (lower_cx, lower_cy)) in teethmap coordinates,
    or None if centroids cannot be determined.
    """
    bb_x, bb_y, bb_w, bb_h = bounding_box_teeth
    y_mid = bb_y + bb_h / 2

    # Determine the gap midline using median of per-column midpoints
    x_start = int(bb_x + bb_w / 2 - margin_x * bb_w / 2)
    x_end = int(bb_x + bb_w / 2 + margin_x * bb_w / 2)
    min_he = bb_y
    max_he = bb_y + bb_h

    gap_midpoints = []
    for x in range(x_start, x_end):
        upper_y = y_mid
        while upper_y > min_he:
            upper_y -= 1
            if teethmap.getpixel((x, int(upper_y))) >= threshold:
                break
        else:
            continue

        lower_y = y_mid
        while lower_y < max_he:
            lower_y += 1
            if teethmap.getpixel((x, int(lower_y))) >= threshold:
                break
        else:
            continue

        if upper_y > min_he and lower_y < max_he:
            gap_midpoints.append((upper_y + lower_y) / 2)

    if not gap_midpoints:
        return None

    gap_midline = sorted(gap_midpoints)[len(gap_midpoints) // 2]  # median

    # Extract ROI restricted to central incisor strip (excludes lateral teeth
    # that form an arch and would pull centroids into the mouth cavity)
    arr = numpy.array(teethmap)
    cx_start = int(bb_x + bb_w / 2 - centroid_margin_x * bb_w / 2)
    cx_end = int(bb_x + bb_w / 2 + centroid_margin_x * bb_w / 2)
    roi = arr[bb_y : bb_y + bb_h, cx_start:cx_end]

    ys, xs = numpy.where(roi >= threshold)
    if len(ys) == 0:
        return None

    # Convert to teethmap coordinates
    abs_xs = xs + cx_start
    abs_ys = ys + bb_y

    # Split into upper and lower groups
    upper_mask = abs_ys < gap_midline
    lower_mask = abs_ys > gap_midline

    upper_xs = abs_xs[upper_mask]
    upper_ys = abs_ys[upper_mask]
    lower_xs = abs_xs[lower_mask]
    lower_ys = abs_ys[lower_mask]

    if len(upper_xs) < min_pixels or len(lower_xs) < min_pixels:
        return None

    # Avoid the septum (dark gap between central incisors): split pixels
    # into left/right halves at the X midline, then keep the same side for
    # both upper and lower groups. This ensures the centroid pair measures
    # left-upper to left-lower (or right to right), not across the septum.
    # The side is chosen by total pixel count across both groups.
    x_midline = (cx_start + cx_end) / 2

    upper_left = numpy.sum(upper_xs < x_midline)
    upper_right = numpy.sum(upper_xs >= x_midline)
    lower_left = numpy.sum(lower_xs < x_midline)
    lower_right = numpy.sum(lower_xs >= x_midline)

    use_left = (upper_left + lower_left) >= (upper_right + lower_right)

    upper_xs, upper_ys = _pick_side(upper_xs, upper_ys, x_midline, use_left)
    lower_xs, lower_ys = _pick_side(lower_xs, lower_ys, x_midline, use_left)

    if len(upper_xs) == 0 or len(lower_xs) == 0:
        return None

    upper_centroid = (float(numpy.mean(upper_xs)), float(numpy.mean(upper_ys)))
    lower_centroid = (float(numpy.mean(lower_xs)), float(numpy.mean(lower_ys)))

    return (upper_centroid, lower_centroid)


def sample_depth_at_point(
    depthmap, point_x, point_y, photo_width, photo_height, kernel_size=3
) -> int | None:
    """Sample depth map at a photo-space coordinate using median filtering.

    Translates from photo-space to depth-map-space and returns the median
    value of a kernel_size x kernel_size region.
    """
    depth_x = round(point_x * depthmap.width / photo_width)
    depth_y = round(point_y * depthmap.height / photo_height)

    half = kernel_size // 2
    values = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            sx = depth_x + dx
            sy = depth_y + dy
            if 0 <= sx < depthmap.width and 0 <= sy < depthmap.height:
                px = depthmap.getpixel((sx, sy))
                # Multi-channel depth maps (e.g. RGB): take first channel
                if isinstance(px, tuple):
                    px = px[0]
                values.append(px)

    if not values:
        return None

    values.sort()
    return values[len(values) // 2]


