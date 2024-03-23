from pathlib import Path

import pytest
from portrait_analyser.ios import load_image


def test_load_image_heic(heic_image_path: Path):
    res = load_image(str(heic_image_path))
    assert len(res) == 2


def test_load_image_jpeg_(jpeg_depth_data_path: Path):
    load_image(str(jpeg_depth_data_path))
