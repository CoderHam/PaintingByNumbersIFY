#! /usr/bin/env python3
import numpy as np
from time import time

import dominant_cluster
import image_utils
import process


def simple_matrix_to_image(mat, palette):
    simple_mat_flat = np.array(
        [[col for col in palette[index]] for index in mat.flatten()])
    return simple_mat_flat.reshape(mat.shape + (3,))


def PBNify(image_path, clusters=20, pre_blur=True):
    image = image_utils.load_image(image_path, resize=False)
    if pre_blur:
        image = process.blur_image(image)

    dominant_colors, quantized_labels, bar_image = dominant_cluster.get_dominant_colors(
        image, n_clusters=clusters, use_gpu=True, plot=True)

    # Create final PBN image
    smooth_labels = process.smoothen(quantized_labels.reshape(image.shape[:-1]))
    pbn_image = dominant_colors[smooth_labels].reshape(image.shape)

    # Create outline image
    outline_image = process.outline(pbn_image)

    return pbn_image, outline_image


t0 = time()
pbn_image, outline_image = PBNify("images/dancing.jpg", clusters=15)
print("PBNify: ", time() - t0, "secs")
image_utils.save_image(pbn_image, "images/PBNImage.jpg")
image_utils.save_image(outline_image, "images/PBNImageOutline.jpg")
