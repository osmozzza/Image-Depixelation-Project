import unittest
from unittest.mock import patch
import os
from PIL import Image
import numpy as np
import torch
import dill as pickle
from datasets import to_grayscale, prepare_image, TrainingDataset, TestDataset

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

class TestTrainingDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and populate it with some test images
        self.test_dir = "./test_images"
        os.makedirs(self.test_dir, exist_ok=True)
        test_image = np.ones((64, 64, 1))
        test_image[0:3,0:3,:] = 3

        self.test_images = [
            test_image,
            test_image,
        ]
        for i, image in enumerate(self.test_images):
            image_path = os.path.join(self.test_dir, f"test_image_{i}.jpg")
            pil_image = Image.fromarray(image.squeeze(), 'L')
            pil_image.save(image_path)

    def tearDown(self):
        # Remove the temporary directory and test images after each test
        for i in range(len(self.test_images)):
            image_path = os.path.join(self.test_dir, f"test_image_{i}.jpg")
            os.remove(image_path)
        os.rmdir(self.test_dir)

    def test_getitem(self):
        np_array = np.ones((1, 64, 64), dtype=np.uint8)
        gs_image = np_array
        gs_image[:, 0:3, 0:3] = 3

        original_image = torch.from_numpy(gs_image.copy()).float()/255

        pixelated_image = np_array * 0.0039
        pixelated_image[:, 0:2, 0] = 0.0118
        pixelated_image[:, 0:2, 1:3] = 0.0078
        pixelated_image_tensor = torch.from_numpy(pixelated_image.copy()).float()
        known_array = np.ones((1, 64, 64), dtype=bool)
        known_array[:, 0:2, 1:3] = False

        expected_inputs = np.concatenate((pixelated_image_tensor, known_array), axis=0)

        with patch("datasets.to_grayscale") as mock_to_grayscale, \
                patch("datasets.prepare_image") as mock_prepare_image:
            mock_to_grayscale.return_value = gs_image
            mock_prepare_image.return_value = (pixelated_image_tensor, known_array, original_image)

            dataset = TrainingDataset(self.test_dir)
            sample = dataset[0]

            self.assertTrue(np.array_equal(sample[0], expected_inputs))  # Array comparison of concatenated image and mask
            self.assertTrue(np.array_equal(sample[1], original_image))  # Array comparison of original image
            self.assertEqual(sample[2], 0)  # Index

    def test_len(self):
        dataset = TrainingDataset(self.test_dir)
        self.assertEqual(len(dataset), 2)

class TestTestDataset(unittest.TestCase):
    def setUp(self):
        # Create a dictionary with pixelated images and known arrays
        pixelated_image = np.ones((1, 64, 64))
        pixelated_image[:, 0:2, 0] = 3
        pixelated_image[:, 0:2, 1:3] = 2

        known_array = np.ones((1, 64, 64), dtype=bool)
        known_array[:, 0:2, 1:3] = False

        self.expected_inputs = np.concatenate((pixelated_image/255, known_array), axis=0)

        test_set = {'pixelated_images': (pixelated_image, pixelated_image),
                    'known_arrays': (known_array, known_array)}

        with open("test_dataset.pkl", 'wb') as file:
            pickle.dump(test_set, file)

    def tearDown(self):
        # Remove the test pickle file after the test
        os.remove("test_dataset.pkl")

    def test_getitem(self):
        dataset = TestDataset("test_dataset.pkl")
        sample = dataset[0]

        self.assertTrue(np.allclose(sample[0].numpy(), self.expected_inputs)) # Concatenated image and mask as tensor
        self.assertTrue(np.array_equal(sample[1], self.expected_inputs)) # Concatenated image and mask as NumPy arrray
        self.assertEqual(sample[2], 0)  # Index

    def test_len(self):
        dataset = TestDataset("test_dataset.pkl")
        self.assertEqual(len(dataset), 2)

if __name__ == '__main__':
    unittest.main()
