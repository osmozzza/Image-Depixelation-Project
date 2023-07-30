import unittest
import numpy as np
import torch
from datasets import to_grayscale, prepare_image

class TestToGrayscale(unittest.TestCase):
    def test_to_grayscale_single_channel(self):
        # Test with a single-channel image (H, W, 1)
        img_array = np.ones((5, 5, 1))
        expected_result = np.ones((1, 5, 5), dtype=np.uint8)
        result = to_grayscale(img_array)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_to_grayscale_three_channels(self):
        # Test with a three-channel image (H, W, 3)
        img_array = np.array([[[100, 50, 200], [30, 140, 70]], [[80, 90, 100], [10, 20, 30]]])
        expected_result = np.array([[[82., 99.], [88., 18.]]], dtype=np.uint8)
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


class TestPrepareImage(unittest.TestCase):
    def test_prepare_image(self):
        # Test pixelating a small area (3x3) in a 6x6 image with block size 2
        image = np.arange(36).reshape(1, 6, 6).astype(np.uint8)
        x, y, width, height, size = 1, 1, 3, 3, 2
        expected_original_image = torch.tensor(image/255)
        expected_pixelated_image = torch.tensor(np.array([[[ 0.0000,  1.0000,  2.0000,  3.0000,  4.0000,  5.0000],
         [ 6.0000, 10.5000, 10.5000, 12.0000, 10.0000, 11.0000],
         [12.0000, 10.5000, 10.5000, 12.0000, 16.0000, 17.0000],
         [18.0000, 19.5000, 19.5000, 21.0000, 22.0000, 23.0000],
         [24.0000, 25.0000, 26.0000, 27.0000, 28.0000, 29.0000],
         [30.0000, 31.0000, 32.0000, 33.0000, 34.0000, 35.0000]]])/255)
        expected_known_array = np.array([[[ True,  True,  True,  True,  True,  True],
        [ True, False, False, False,  True,  True],
        [ True, False, False, False,  True,  True],
        [ True, False, False, False,  True,  True],
        [ True,  True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True,  True]]])

        pixelated_image, known_array, original_image = prepare_image(image, x, y, width, height, size)

        self.assertTrue(np.allclose(pixelated_image, expected_pixelated_image))
        self.assertTrue(np.array_equal(known_array, expected_known_array))
        self.assertTrue(np.allclose(original_image, expected_original_image))


if __name__ == '__main__':
    unittest.main()
