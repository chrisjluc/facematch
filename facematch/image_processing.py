import consts

import copy
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def _reflection(image):
    """
    Given an Image object, return the reflected image as an Image object.
    """
    im, feature_points, landmark_points = image.image, image.feature_points, image.landmark_points
    feature_points = [(im.shape[1] - p[0], p[1]) for p in feature_points]
    landmark_points = [(im.shape[1] - p[0], p[1]) for p in landmark_points]
    landmark_points[:3], landmark_points[6:] = landmark_points[6:], landmark_points[:3]
    im = np.array([list(reversed(row)) for row in im])
    image.im, image.feature_points, image.landmark_points = im, feature_points, landmark_points
    return image

def apply_reflection(images):
    """
    images: list of list of images

    Apply reflection to all the images and return it as a list of list of images
    """
    return [
            [_reflection(image) for image in _images]
            for _images in images
            ]

def get_random_noise_image(image, coords, width):
    """
    Apply random gaussian generated values
    and distribute them on gaussian distributed square
    centered on the coordinates passed in for the image
    """
    noise = np.zeros((image.shape[0], image.shape[1]))
    for coord in coords:
        # Convert coordinates to rows / columns
        apply_noise_at_point(coord[1], coord[0], noise, width)
    return np.clip(image + noise, 0, 1)

def apply_noise_at_point(row, col, noise, width):
    """
    Generate a block with a single random value placed at the center
    Apply the Gaussian filter with std of 4
    Place it on the noise array at the appropriate coordinates
    """
    block = np.zeros((width, width))
    block[width / 2, width / 2] = np.random.normal()
    block = gaussian_filter(block, sigma=4)

    row -= width / 2
    col -= width / 2

    row_end = min(noise.shape[0] - row, block.shape[0])
    row_start =  max(0, -row)

    col_end = min(noise.shape[1] - col, block.shape[1])
    col_start = max(0, -col)

    noise[max(0, row):row + block.shape[0], max(0, col):col + block.shape[1]] = block[row_start:row_end,col_start:col_end]

def _apply_noise(image):
    image.image = get_random_noise_image(image.image, image.feature_points, consts.noise_width)
    return image

def apply_noise(images):
    """
    images: list of list of images
    """
    return [
            [_apply_noise(image) for image in _images]
            for _images in images
            ]

def get_image_window(image, size, point):
    """
    Assume image is grey image
    params:
    image: np array
    size: (row, cols)
    point: (x, y)
    """
    top = int(point[1] - size[0] / 2)
    left = int(point[0] - size[1] / 2)
    return image[top:top + size[0], left:left + size[1]]
