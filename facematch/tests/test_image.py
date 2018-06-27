from .. import consts

from ..image import Image

import pytest

test_image = '.data/George_W_Bush/George_W_Bush_0104.jpg'

class TestImage(object):
    def test_image_creation(self):
        image = Image(test_image)
        assert image.image.shape == consts.norm_shape
        assert len(image.landmark_points) == 9
        assert len(image.feature_points) == 5
        image.assert_valid_state()
