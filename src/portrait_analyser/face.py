import os

import cv2
import numpy
from PIL import Image

from .exceptions import MultipleFacesDetected, NoFacesDetected

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
