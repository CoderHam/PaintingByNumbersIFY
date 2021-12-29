#! /usr/bin/env python3
import faiss
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

import image_utils


def kmeans_faiss(dataset, k):
    """
    Runs KMeans on GPU"
    """
    dims = dataset.shape[1]
    cluster = faiss.Clustering(dims, k)
    cluster.verbose = False
    cluster.niter = 20
    cluster.max_points_per_centroid = 10**7

    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.useFloat16 = False
    config.device = 0
    index = faiss.GpuIndexFlatL2(resources, dims, config)

    # perform kmeans
    cluster.train(dataset, index)
    centroids = faiss.vector_float_to_array(cluster.centroids)

    return centroids.reshape(k, dims)


def compute_cluster_assignment(centroids, data):
    dims = centroids.shape[1]

    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.useFloat16 = False
    config.device = 0

    index = faiss.GpuIndexFlatL2(resources, dims, config)
    index.add(centroids)
    _, labels = index.search(data, 1)

    return labels.ravel()


def get_dominant_colors(image, n_clusters=10, use_gpu=True, plot=True):
    # Must pass FP32 data to kmeans_faiss since faiss does not support uint8
    flat_image = image.reshape(
        (image.shape[0] * image.shape[1], 3)).astype(np.float32)

    if use_gpu:
        centroids = kmeans_faiss(flat_image, n_clusters)
        labels = compute_cluster_assignment(centroids,
                                            flat_image).astype(np.uint8)
        centroids = centroids.astype(np.uint8)
    else:
        clt = KMeans(n_clusters=n_clusters).fit(flat_image)
        centroids = clt.cluster_centers_.astype(np.uint8)
        labels = clt.labels_.astype(np.uint8)

    if plot:
        counts = Counter(labels).most_common()
        centroid_size_tuples = [
            (centroids[k], val / len(labels)) for k, val in counts
        ]
        bar_image = image_utils.bar_colors(centroid_size_tuples)
        return centroids, labels, bar_image

    return centroids, labels
