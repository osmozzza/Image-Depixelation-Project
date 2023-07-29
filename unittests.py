import unittest
import numpy as np
from datasets import to_grayscale

class TestToGrayscale(unittest.TestCase):
    def test_to_grayscale_single_channel(self):
        # Test with a single-channel image (H, W, 1)
        img_array = np.ones((5, 5, 1))
        expected_result = np.ones((1, 5, 5))
        result = to_grayscale(img_array)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_to_grayscale_three_channels(self):
        # Test with a three-channel image (H, W, 3)
        img_array = np.array([[[100, 50, 200], [30, 140, 70]], [[80, 90, 100], [10, 20, 30]]])
        expected_result = np.array([[[82., 99.],
        [88., 18.]]])
        result = to_grayscale(img_array)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_to_grayscale_invalid_shape(self):
        # Test with an image of invalid shape (4-dimensional array)
        img_array = np.zeros((10, 10, 3, 1))
        with self.assertRaises(ValueError):
            to_grayscale(img_array)

    def test_to_grayscale_invalid_channels(self):
        # Test with an image that does not have 3 channels
        img_array = np.ones((5, 5, 2))
        with self.assertRaises(ValueError):
            to_grayscale(img_array)


if __name__ == '__main__':
    unittest.main()
