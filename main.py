#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import dominant_cluster
import image_utils
import process


def get_nearest(palette, color):
    distances = np.linalg.norm(palette - color, axis=1)
    return np.argmin(distances)

def image_to_simple_matrix(image, palette):
    return np.apply_along_axis(lambda col: get_nearest(palette, col), 2, image)

def simple_matrix_to_image(mat, palette):
    width = len(mat[0])
    height = len(mat)

    imgFlat = [[c for c in palette[xyVal]] for yVal in mat for xyVal in yVal]
    imgData = [imgFlat[i:i+height] for i in range(0, len(imgFlat), height)]

    return imgData

# Must pass FP32 data to get_dominant_colors since faiss does not support uint8
image = image_utils.load_image("images/picasso.jpg", resize=False)
palette = dominant_cluster.get_dominant_colors(image, n_clusters=20, use_gpu=True, plot=False)
# palette = [[25, 43, 49], [100, 111, 111], [58, 68, 65], [154, 155, 149], [85, 97, 97], [39, 62, 69], [18, 34, 40], [71, 84, 85], [148, 141, 122], [53, 74, 79], [123, 117, 101], [30, 51, 58], [43, 55, 53], [183, 181, 174], [121, 130, 130]]

image = (image*255).astype("uint8")
mat = image_to_simple_matrix(image, palette)

# matSmooth, labelLocs, matLine = process.img_process(mat)
smooth_mat, line_mat = process.img_process(mat)

height = len(line_mat)
borders = np.abs(line_mat - 1)
# borderFlat = [[abs(xyVal-1.0) for ii in range(0,3)] for yVal in line_mat for xyVal in yVal]
# borders = np.array([borderFlat[i:i+height] for i in range(0, len(borderFlat), height)])

PBNImage = np.array(simple_matrix_to_image(smooth_mat, palette))

image_utils.save_plot(PBNImage, "images/PBNImage2.jpg")
image_utils.save_plot(borders, "images/PBNImageOutline2.jpg")
