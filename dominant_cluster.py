#! /usr/bin/env python3
import faiss
from sklearn.cluster import KMeans
import image_utils

import numpy as np
from collections import Counter

def kmeans_faiss(dataset, k):
    "Runs KMeans on GPU/s"
    dims = dataset.shape[1]
    cluster = faiss.Clustering(dims, k)
    cluster.verbose = False
    cluster.niter = 20
    cluster.max_points_per_centroid = 10 ** 7

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


def centroid_histogram(clt):
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)

    (hist, _) = np.histogram(clt.labels_, bins=num_labels)
    hist = hist.astype(np.float32)
    hist /= hist.sum()

    return hist


def get_dominant_colors(image, n_clusters=10, use_gpu=True, plot=True):
    image = image.reshape((image.shape[0] * image.shape[1], 3)).astype(np.float32)

    if use_gpu:
        centroids = kmeans_faiss(image, n_clusters)
        labels = compute_cluster_assignment(centroids, image)
        centroids = centroids.astype("uint8")

        if plot:
            counts = Counter(labels).most_common()
            total = sum(n for _, n in counts)
            centroid_size_tuples = [(centroids[k], val/total) for k, val in counts]
    else:
        clt = KMeans(n_clusters=n_clusters)
        clt.fit(image)
        centroids = clt.cluster_centers_.astype(np.uint8)

        if plot:
            hist = centroid_histogram(clt)
            centroid_size_tuples = list(zip(centroids, hist))
            centroid_size_tuples.sort(key=lambda x: x[1], reverse=True)
    if plot:
        bar_image = image_utils.bar_colors(centroid_size_tuples)
        return centroids, bar_image

    return centroids


# import time
# t0 = time.time()
# dominant_colors, bar = get_dominant_colors("images/dancing.jpg", n_clusters=10)
# t1 = time.time()
# print(dominant_colors)
# image_utils.save_plot(bar, "test1.png")
# print("k-Means runtime: %.3f s" % (t1 - t0))

# t0 = time.time()
# dominant_colors, bar = get_dominant_colors("images/dancing.jpg", n_clusters=10, use_gpu=False)
# t1 = time.time()
# print(dominant_colors)
# image_utils.save_plot(bar, "test.png")
# print("k-Means runtime: %.3f s" % (t1 - t0))
