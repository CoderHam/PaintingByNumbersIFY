#! /usr/bin/env python3
import numpy as np
from skimage import io, transform
import cv2


def load_image(img_path, resize=False):
	tmp_img = io.imread(img_path)
	if resize:
		return transform.resize(image=tmp_img, output_shape=(200,200), anti_aliasing=True, mode='constant')
	return tmp_img


def bar_colors(centroid_size_tuples):
	bar = np.zeros((50, 300, 3), dtype="uint8")
	x_start = 0
	for (color, percent) in centroid_size_tuples:
		x_end = x_start + (percent * 300)
		cv2.rectangle(bar, (int(x_start), 0), (int(x_end), 50),
			color.astype("uint8").tolist(), -1)
		x_start = x_end
	return bar


def save_plot(plot, image_path):
	io.imsave(image_path, plot)


