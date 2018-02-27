import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter

import dominant_cluster
import process

def getNearest(palette, col):
    nearest = 0
    nearestDistsq = 1000000
    for i in range(0,len(palette)):
        pcol = palette[i]
        distsq = pow(pcol[0] - col[0], 2) + pow(pcol[1] - col[1], 2) + pow(pcol[2] - col[2], 2)
        if distsq < nearestDistsq:
            nearest = i
            nearestDistsq = distsq

    return nearest

def imageDataToSimpMat(imgData, palette):
    width = len(imgData[0])
    height = len(imgData)
    mat = [[0]*width]*height
    for y in range(0,height):
        for x in range(0,width):
            nearestI = getNearest(palette, [imgData[y][x][0],imgData[y][x][1],imgData[y][x][2]])
            mat[y][x] = nearestI

    return mat

def matToImageData(mat, palette):
    width = len(mat[0])
    height = len(mat)
    imgData = [[[0]*3]*width]*height
    for y in range(0,height):
        for x in range(0,width):
            col = palette[mat[y][x]]
            imgData[y][x] = [float(c/255.0) for c in col]
    return imgData

img_path = "/home/hemant/Projects/Pytorch-Tutorials/images/picasso.jpg"
domColors = dominant_cluster.get_dom_colors(img_path,15,True)
palette = domColors
# palette = [[25, 43, 49], [100, 111, 111], [58, 68, 65], [154, 155, 149], [85, 97, 97], [39, 62, 69], [18, 34, 40], [71, 84, 85], [148, 141, 122], [53, 74, 79], [123, 117, 101], [30, 51, 58], [43, 55, 53], [183, 181, 174], [121, 130, 130]]

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mat = imageDataToSimpMat(image, palette)
smoothened = matToImageData(mat,palette)

plt.imshow(smoothened)
plt.show()

# matSmooth, labelLocs, matLine = process.img_process(mat)
matSmooth = process.img_process(mat)

processed = [palette[m] for m in matSmooth]

plt.imshow(processed)
plt.show()
