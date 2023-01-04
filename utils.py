import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import helpers.consts as consts

from helpers.helpers import create_algs_wrapper


def delete_description_by_size(image, size):
    height = image.shape[0] - size
    image_with_deleted_description = image[:height]
    return image_with_deleted_description


def delete_description_auto(image):
    white_pixel = 255
    black_pixel = 0
    description_colors = set([white_pixel, black_pixel])
    return np.array(list(filter(lambda row: set(row) != description_colors and set(row) != set([black_pixel]), image)), np.uint8)


def plot_intensity_hist(images, bins=256, figsize=(5, 5)):
    images_count = len(images)
    fig, ax = plt.subplots(
        math.ceil(images_count / 3), images_count if images_count < 3 else 3
    )
    fig.set_size_inches(figsize)

    for i in range(len(images)):
        image = images[i]
        ax[i].hist(np.ravel(image), bins, label='total')
        ax[i].legend()

    return ax


filter_image = create_algs_wrapper({
    consts.AVERAGE: {
        consts.ALG: cv2.blur,
        consts.PARAMS: {'ksize': (10, 10)}
    },
    consts.MEDIAN: {
        consts.ALG: cv2.medianBlur,
        consts.PARAMS: {'ksize': 5}
    },
    consts.GAUSSIAN: {
        consts.ALG: cv2.GaussianBlur,
        consts.PARAMS: {'ksize': (3, 3), 'sigmaX': 0}
    },
})

morph_operations_kernel = np.ones((5, 5), np.uint8)
morph_transfrom = create_algs_wrapper({
    consts.EROSION: {
        consts.ALG: cv2.erode,
        consts.PARAMS: {'kernel': morph_operations_kernel, 'iterations': 1}
    },
    consts.DILATION: {
        consts.ALG: cv2.dilate,
        consts.PARAMS: {'kernel': morph_operations_kernel, 'iterations': 1}
    },
    consts.OPENING: {
        consts.ALG: cv2.morphologyEx,
        consts.PARAMS: {'op': cv2.MORPH_OPEN,
                        'kernel': morph_operations_kernel}
    },
    consts.CLOSING: {
        consts.ALG: cv2.morphologyEx,
        consts.PARAMS: {'op': cv2.MORPH_CLOSE,
                        'kernel': morph_operations_kernel}
    }
})

detect_edges = create_algs_wrapper({
    consts.SOBEL: {
        consts.ALG: cv2.Sobel,
        consts.PARAMS: {'ddepth': cv2.CV_64F, 'dx': 1, 'dy': 0, 'ksize': 5}
    },
    consts.CANNY: {
        consts.ALG: cv2.Canny,
        consts.PARAMS: {'threshold1': 100, 'threshold2': 100}
    }
})
