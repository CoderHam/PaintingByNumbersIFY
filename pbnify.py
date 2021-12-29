#! /usr/bin/env python3
import numpy as np
import argparse
import os

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input-image',
                        type=str,
                        required=True,
                        help='Path of input image.')
    parser.add_argument('-o',
                        '--output-image',
                        type=str,
                        required=True,
                        help='Path of output image.')
    parser.add_argument(
        '-k',
        '--num-of-clusters',
        type=int,
        required=False,
        default=15,
        help=
        'Number of kmeans clusters for dominant color calculation. Defaults to 15.'
    )
    parser.add_argument('--outline',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Save outline image containing edges.')
    FLAGS = parser.parse_args()

    pbn_image, outline_image = PBNify(FLAGS.input_image,
                                      clusters=FLAGS.num_of_clusters)
    image_utils.save_image(pbn_image, FLAGS.output_image)

    if FLAGS.outline:
        outline_image_path = os.path.splitext(
            FLAGS.output_image)[0] + "_outline.jpg"
        image_utils.save_image(outline_image, outline_image_path)
