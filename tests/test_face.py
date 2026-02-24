import pytest
from portrait_analyser.exceptions import NoFacesDetected
from portrait_analyser.face import get_face_parameters
from portrait_analyser.ios import load_image


def test_get_face_parameters_no_face(heic_image_path):
    result = load_image(str(heic_image_path))
    with pytest.raises(NoFacesDetected):
        get_face_parameters(result.photo)


def test_get_face_parameters_face(heic_face_image_path):
    result = load_image(str(heic_face_image_path))

    res = get_face_parameters(result.photo)
    assert res
