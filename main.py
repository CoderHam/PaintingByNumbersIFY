#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import dominant_cluster
import image_utils
import process

# def norm(x, y): 
#     return np.linalg.norm(x - y) 

def get_nearest(palette, color):
    distances = np.apply_along_axis(lambda pcol: np.linalg.norm(pcol - color), 1, palette)
    return np.argmin(distances)

def image_to_simple_matrix(image, palette):
    return np.apply_along_axis(lambda col: get_nearest(palette, col), 2, image)

def matrix_to_image_numpy(mat, palette):
    width = len(mat[0])
    height = len(mat)
    # for y in range(0,height):
    #     for x in range(0,width):
    #         col = palette[mat[y][x]]
    #         imgData[y][x] = [float(c/255.0) for c in col]

    imgFlat = [[c for c in palette[xyVal]] for yVal in mat for xyVal in yVal]
    imgData = [imgFlat[i:i+height] for i in range(0, len(imgFlat), height)]

    return imgData

# Must pass FP32 data to get_dominant_colors since faiss does not support uint8
image = image_utils.load_image("images/picasso.jpg", resize=False)
import time
t0 = time.time()
palette = dominant_cluster.get_dominant_colors(image, n_clusters=20, use_gpu=True, plot=False)
t1 = time.time()
print("k-Means runtime: %.3f s" % (t1 - t0))
# palette = [[25, 43, 49], [100, 111, 111], [58, 68, 65], [154, 155, 149], [85, 97, 97], [39, 62, 69], [18, 34, 40], [71, 84, 85], [148, 141, 122], [53, 74, 79], [123, 117, 101], [30, 51, 58], [43, 55, 53], [183, 181, 174], [121, 130, 130]]

image = (image*255).astype("uint8")
t0 = time.time()
mat = image_to_simple_matrix(image, palette)
t1 = time.time()
print("image_to_simple_matrix runtime: %.3f s" % (t1 - t0))

# matSmooth, labelLocs, matLine = process.img_process(mat)
t0 = time.time()
matSmooth, matLine = process.img_process(mat)
t1 = time.time()
print("process.img_process runtime: %.3f s" % (t1 - t0))

# height = len(matLine)
# borderFlat = [[abs(xyVal-1.0) for ii in range(0,3)] for yVal in matLine for xyVal in yVal]
# borders = np.array([borderFlat[i:i+height] for i in range(0, len(borderFlat), height)])

t0 = time.time()
PBNImage = np.array(matrix_to_image_numpy(matSmooth, palette))
t1 = time.time()
print("matrix_to_image_numpy runtime: %.3f s" % (t1 - t0))

# print(PBNImage.dtype)
# print(borders.dtype)
image_utils.save_plot(PBNImage, "images/PBNImage2.jpg")
# image_utils.save_plot(borders, "images/PBNImageOutline2.jpg")
