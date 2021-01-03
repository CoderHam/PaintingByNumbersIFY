#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from time import time

import dominant_cluster
import image_utils
import process

def simple_matrix_to_image(mat, palette):
    simple_mat_flat = np.array([[col for col in palette[index]] for index in mat.flatten()])
    return simple_mat_flat.reshape(mat.shape + (3,))

def PBNify(image_path, clusters=20, pre_blur=True):
    image = image_utils.load_image(image_path, resize=False)
    if pre_blur:
        image = process.blur_image(image)

    # Must pass FP32 data to get_dominant_colors since faiss does not support uint8
    dominant_colors, quantized_labels, _ = dominant_cluster.get_dominant_colors(image,
                                                   n_clusters=clusters,
                                                   use_gpu=True,
                                                   plot=True)
    # quantized_image = dominant_colors[quantized_labels].reshape(image.shape)
    # image_utils.save_plot(quantized_image, "quantized.jpg")

    smooth_labels = process.smoothen(quantized_labels.reshape(image.shape[:-1]))
    smooth_image = dominant_colors[smooth_labels].reshape(image.shape)
    # image_utils.save_plot(smooth_image, "smooth.jpg")

    edge_image = process.edge_mask(smooth_image)
    # image_utils.save_plot(edge_image, "edge.jpg")

    pbn_image = process.merge_mask(smooth_image, edge_image)
    outline_image = process.outline(smooth_image)

    return pbn_image, outline_image

t0 = time()
pbn_image, outline_image = PBNify("images/picasso.jpg", clusters=15)
print("PBNify: ", time() - t0, "secs")
image_utils.save_plot(pbn_image, "images/picasso_PBN.jpg")
image_utils.save_plot(outline_image, "images/picasso_PBNOutline.jpg")
