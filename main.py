import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.misc import imsave
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
    # for y in range(0,height):
    #     for x in range(0,width):
    #         mat[y][x] = getNearest(palette, imgData[y][x])

    flatMat = [getNearest(palette,xyVal) for yVal in imgData for xyVal in yVal]
    mat = [flatMat[i:i+height] for i in range(0, len(flatMat), height)]

    return mat

def matToImageData(mat, palette):
    width = len(mat[0])
    height = len(mat)
    # for y in range(0,height):
    #     for x in range(0,width):
    #         col = palette[mat[y][x]]
    #         imgData[y][x] = [float(c/255.0) for c in col]

    imgFlat = [[float(c/255.0) for c in palette[xyVal]] for yVal in mat for xyVal in yVal]
    imgData = [imgFlat[i:i+height] for i in range(0, len(imgFlat), height)]

    return imgData

img_path = "images/picasso.jpg"
domColors = dominant_cluster.get_dom_colors(img_path,20,True)
palette = domColors
# palette = [[25, 43, 49], [100, 111, 111], [58, 68, 65], [154, 155, 149], [85, 97, 97], [39, 62, 69], [18, 34, 40], [71, 84, 85], [148, 141, 122], [53, 74, 79], [123, 117, 101], [30, 51, 58], [43, 55, 53], [183, 181, 174], [121, 130, 130]]

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mat = imageDataToSimpMat(image, palette)
# smoothened = matToImageData(mat,palette)

# plt.imshow(smoothened)
# plt.show()
# imsave("images/simpImage.jpg",smoothened)

# matSmooth, labelLocs, matLine = process.img_process(mat)
matSmooth, matLine = process.img_process(mat)

height = len(matLine)
borderFlat = [[abs(xyVal-1.0) for ii in range(0,3)] for yVal in matLine for xyVal in yVal]
borders = [borderFlat[i:i+height] for i in range(0, len(borderFlat), height)]

PBNImage = matToImageData(matSmooth,palette)
# plt.imshow(PBNImage)
# plt.show()
imsave("images/PBNImage2.jpg",PBNImage)
imsave("images/PBNImageOutline2.jpg",borders)
