#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from time import time

import dominant_cluster
import image_utils
import process

def get_nearest(palette, color):
    return np.argmin(np.sum(((palette.astype(np.int32) - color))**2, axis=1))

def image_to_simple_matrix(image, palette):
    return np.apply_along_axis(lambda col: get_nearest(palette, col), 2, image)

def simple_matrix_to_image(mat, palette):
    simple_mat_flat = np.array([[col for col in palette[index]] for index in mat.flatten()])
    return simple_mat_flat.reshape(mat.shape + (3,))

def PBNify(image_path, clusters=20):
    image = image_utils.load_image(image_path, resize=False)

    t0 = time()
    # Must pass FP32 data to get_dominant_colors since faiss does not support uint8
    palette = dominant_cluster.get_dominant_colors(image,
                                                   n_clusters=clusters,
                                                   use_gpu=True,
                                                   plot=False)
    print("dominant_cluster", time() - t0)

    t0 = time()
    mat = image_to_simple_matrix(image, palette).astype(np.uint8)
    print("image_to_simple_matrix", time() - t0)

    t0 = time()
    smooth_mat, outline_image = process.img_process(mat)
    print("process.img_process", time() - t0)
    t0 = time()
    pbn_image = np.array(simple_matrix_to_image(smooth_mat, palette))
    print("simple_matrix_to_image", time() - t0)

    return pbn_image, outline_image


pbn_image, outline_image = PBNify("images/picasso.jpg")
image_utils.save_plot(pbn_image, "images/picasso_PBN.jpg")
image_utils.save_plot(outline_image, "images/picasso_PBNOutline.jpg")
