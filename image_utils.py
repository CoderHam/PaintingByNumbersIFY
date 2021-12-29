#! /usr/bin/env python3
import numpy as np
from PIL import Image
import cv2


def load_image(image_path, resize=False):
    image = np.asarray(Image.open(image_path))
    if resize:
        return cv2.resize(image, (200, 200), cv2.INTER_AREA)
    return np.asarray(image)


def bar_colors(centroid_size_tuples):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    x_start = 0
    for (color, percent) in centroid_size_tuples:
        x_end = x_start + (percent * 300)
        cv2.rectangle(bar, (int(x_start), 0), (int(x_end), 50),
                      color.astype("uint8").tolist(), -1)
        x_start = x_end
    return bar


def save_image(image, image_path):
    if image.shape[-1] == 3:
        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
    else:
        PIL_image = Image.fromarray(image.astype('uint8'), 'L')
    PIL_image.save(image_path)
