import copy
import gc
import numpy as np

import consts
import image_processing as ip


def merge(a, b):
    """
    args:
    a - list of list of images
    b - list of list of images

    Returns:
    Merged list of list of images
    """
    if len(a) != len(b):
        raise Exception('a and b should have the same length')

    merged = []
    for x, y in zip(a, b):
        merged.append(x + y)
    assert(len(merged) == len(a))
    return merged


def clone(images, factor):
    """
    images: list of list
    factor: int
    """
    cloned_images = []
    for _images in images:
        _cloned_images = []
        for i in range(factor):
            _cloned_images += copy.deepcopy(_images)
        cloned_images.append(_cloned_images)
    assert(len(images) == len(cloned_images))
    return cloned_images

def _get_images_window(data, rows, cols, landmark_index):
    temp_images_by_class = [None for _ in range(len(data))]
    for i in range(len(data)):
        temp_images_by_class[i] = [ip.get_image_window(
                    image.image,
                    (rows, cols),
                    image.landmark_points[landmark_index]) for image in data[i]]
    images = [image
            for images in temp_images_by_class
            for image in images
            ]
    return images

def create_training_data_for_mmdfr(data):
    """
    Get image window and correct sizes for
    each CNN. Shuffle the data and return
    it as a tuple.

    This function is fairly memory intensive
    so we're going to garbage collect and
    free memory of unused variables often.
    """
    # Explicitly run garbage collection
    gc.collect()

    data_h1 = _get_images_window(data, consts.nn2_input_shape[1], consts.nn2_input_shape[2], 4)
    data_p1 = _get_images_window(data, consts.nn1_input_shape[1], consts.nn1_input_shape[2], 0)
    data_p2 = _get_images_window(data, consts.nn1_input_shape[1], consts.nn1_input_shape[2], 1)
    data_p3 = _get_images_window(data, consts.nn1_input_shape[1], consts.nn1_input_shape[2], 2)
    data_p4 = _get_images_window(data, consts.nn1_input_shape[1], consts.nn1_input_shape[2], 3)
    data_p5 = _get_images_window(data, consts.nn1_input_shape[1], consts.nn1_input_shape[2], 4)
    data_p6 = _get_images_window(data, consts.nn1_input_shape[1], consts.nn1_input_shape[2], 5)
    data_y = [images[0] for images in enumerate(data) for image in images[1]]

    # Don't reference this anymore
    del data
    gc.collect()

    zipped_data = np.array(zip(data_h1, data_p1, data_p2, data_p3, data_p4, data_p5, data_p6, data_y))

    zipped_data = [images
            for images in zipped_data
            if images[0].shape == (consts.nn2_input_shape[1], consts.nn2_input_shape[2])
            and images[1].shape == (consts.nn1_input_shape[1], consts.nn1_input_shape[2])
            and images[2].shape == (consts.nn1_input_shape[1], consts.nn1_input_shape[2])
            and images[3].shape == (consts.nn1_input_shape[1], consts.nn1_input_shape[2])
            and images[4].shape == (consts.nn1_input_shape[1], consts.nn1_input_shape[2])
            and images[5].shape == (consts.nn1_input_shape[1], consts.nn1_input_shape[2])
            and images[6].shape == (consts.nn1_input_shape[1], consts.nn1_input_shape[2])
            ]

    np.random.shuffle(zipped_data)

    ret_data = (np.array([x[0] for x in zipped_data]),
            np.array([x[1] for x in zipped_data]),
            np.array([x[2] for x in zipped_data]),
            np.array([x[3] for x in zipped_data]),
            np.array([x[4] for x in zipped_data]),
            np.array([x[5] for x in zipped_data]),
            np.array([x[6] for x in zipped_data]),
            np.array([x[7] for x in zipped_data]))

    del zipped_data
    gc.collect()

    return ret_data

