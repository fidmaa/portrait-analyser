from pathlib import Path

import pytest
from portrait_analyser.exceptions import UnknownExtension
from portrait_analyser.ios import IOSPortrait, load_image


def test_load_image_heic(heic_image_path: Path):
    res = load_image(str(heic_image_path))
    assert isinstance(res, IOSPortrait)
    assert res.photo is not None
    assert res.depthmap is not None


def test_load_image_jpeg_(jpeg_depth_data_path: Path):
    with pytest.raises(UnknownExtension):
        load_image(str(jpeg_depth_data_path))
