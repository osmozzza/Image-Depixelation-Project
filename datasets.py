"""
Author: Angelika Vižintin

################################################################################

Datasets file of Image Depixelation Project.
"""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
import dill as pickle
import torch
from torchvision import transforms

def to_grayscale(img_array: np.ndarray) -> np.ndarray:
    """Converts image as numpy array of shape (H, W, 1) or (H, W, 3) to
    grayscale image as numpy array of shape (1, H, W) with uint8 data type."""
    if img_array.ndim == 2:
        gs_image = np.reshape(img_array.copy(), (1, img_array.shape[0], img_array.shape[1]))
    elif img_array.ndim == 3:
        if img_array.shape[2] != 3:
            raise ValueError("Image does not have 3 channels!")
        else:
            gs_image = (img_array[..., 0]*0.2989 +
                        img_array[..., 1]*0.5870 +
                        img_array[..., 2]*0.1140)
            gs_image = np.round(gs_image)
            gs_image = np.asarray(gs_image, dtype=np.uint8)
            gs_image = np.reshape(gs_image, (1, img_array.shape[0], img_array.shape[1]))
    else:
        raise ValueError("Image does not have required shape!")
    return gs_image

def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ The function takes an input grayscale image of shape (1, H, W) along with specified coordinates
    and dimensions to pixelate a specific area of the image.

    :param image: A NumPy array representing the input image.
    :param x: The starting x-coordinate of the pixelated area.
    :param y: The starting y-coordinate of the pixelated area.
    :param width: The width of the pixelated area.
    :param height: The height of the pixelated area.
    :param size: The size of each pixelation block.
    :return: a tuple containing the pixelated image normalized to range [0,1], a binary mask indicating
    the pixelated area, and the original image normilized to range [0,1].
    """
    original_image = torch.from_numpy(image.copy()).float()

    pixelated_image = torch.from_numpy(image.copy()).float()
    curr_x = x
    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (..., slice(curr_y, min(curr_y + size, y + height)), slice(curr_x, min(curr_x + size, x + width)))
            pixelated_image[block] = pixelated_image[block].mean()
            curr_y += size
        curr_x += size

    pixelated_area = (..., slice(y, y + height), slice(x, x + width))
    known_array = np.ones_like(image, dtype=bool).astype(np.bool)
    known_array[pixelated_area] = False

    return pixelated_image / 255, known_array, original_image / 255


class TrainingDataset(Dataset):

    def __init__(self, image_dir):
        """
        Dataset which resizes, crops, converts to grayscale and pixelates images located in a specified directrory.

        :param image_dir: the directory where the images are located
        """
        self.image_files = sorted(glob.glob(os.path.join(os.path.abspath(image_dir), "**", "*.jpg"), recursive=True))
        self.width_range = (4, 32)
        self.height_range = (4, 32)

    def __getitem__(self, index: int):
        transforms_chain = transforms.Compose([
            transforms.Resize(size=64),
            transforms.CenterCrop(size=(64, 64)),
        ])
        image_file = self.image_files[index]
        with Image.open(image_file) as im:
            image_width = im.width
            image_height = im.height
            transformed_im = transforms_chain(im.copy())
            gs_image = to_grayscale(np.array(transformed_im))

        rng = np.random.default_rng(seed=index)
        width = rng.integers(low=min(self.width_range[0], image_width), high=min(self.width_range[1], image_width), endpoint=True) #self.width?
        height = rng.integers(low=min(self.height_range[0], image_height), high=min(self.height_range[1], image_height), endpoint=True)  # self.height?
        x = rng.integers(low=0, high=(64 - width), endpoint=True)
        y = rng.integers(low=0, high=(64 - height), endpoint=True)
        size = rng.integers(low=4, high=16, endpoint=True)

        prepared_image = prepare_image(gs_image, x, y, width, height, size)

        inputs = np.concatenate((prepared_image[0], prepared_image[1]), axis=0)

        return inputs, prepared_image[2], index

    def __len__(self):
        return len(self.image_files)


class TestDataset(Dataset):

    def __init__(self, pkl_file):
        """"""
        with open(pkl_file, 'rb') as file:
            test_set = pickle.load(file)
        self.pixelated_images = test_set['pixelated_images']
        self.known_arrays = test_set['known_arrays']

    def __getitem__(self, index: int):
        norm_pixelated_image = self.pixelated_images[index] / 255
        inputs = np.concatenate((norm_pixelated_image, self.known_arrays[index]), axis=0)
        return inputs, index

    def __len__(self):
        return len(self.pixelated_images)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = TrainingDataset("two_images")

    inputs, _, index = ds[0]
    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(inputs[0] * 255, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("pixelated_image")
    axes[1].imshow(inputs[1], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("known_array")
    fig.suptitle(index)
    fig.tight_layout()
    plt.show()