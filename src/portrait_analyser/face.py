import cv2
import numpy
from PIL import Image

from .exceptions import MultipleFacesDetected, NoFacesDetected


def get_face_parameters(input_image: Image):
    """Get face position and size or return an exception in
    case there's none."""

    image = numpy.asarray(input_image.convert("RGB"))

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    )
    try:
        face = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=4)
    except BaseException:
        face = []

    #
    # Update midline point to match detected face coords
    #

    if len(face) == 1:
        x, y, wi, he = face[0]

        center_x = x + wi / 2
        center_y = y + he / 2

        # calculate percentage of the face
        img_wi, img_he = input_image.getbbox()[2:4]
        percent_width = float(wi) / float(img_wi)
        percent_height = float(he) / float(img_he)

        return (center_x, center_y, wi, he, percent_width, percent_height)

    elif len(face) == 0:
        raise NoFacesDetected()

    else:
        raise MultipleFacesDetected()
