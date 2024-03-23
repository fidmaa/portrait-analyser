from pathlib import Path

import pytest


@pytest.fixture
def heic_image_path():
    return Path(__file__).parent / "heic_depth_data.heic"

@pytest.fixture
def heic_face_image_path():
    return Path(__file__).parent / "heic_face_data.heic"


@pytest.fixture
def jpeg_no_depth_data_path():
    return Path(__file__).parent / "jpeg_no_depth_data.jpg"


@pytest.fixture
def jpeg_depth_data_path():
    return Path(__file__).parent / "jpeg_depth_data.jpg"

