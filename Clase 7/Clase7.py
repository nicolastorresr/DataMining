#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
import warnings
warnings.filterwarnings("ignore")
from numpy import genfromtxt
from ipywidgets import interact, fixed
import pandas as pd

from sklearn.neighbors import kneighbors_graph

from scipy.spatial import distance_matrix

def RBF(dist_matrix, sigma):
    return np.round(np.exp(- dist_matrix ** 2 / (2. * sigma ** 2)),2)

from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

def plot_spectral_embedding(forma, k=2, kernel='rbf'):
    data = genfromtxt(f"{forma}.csv", delimiter=',')
    spectral = SpectralClustering(n_clusters=k, assign_labels='kmeans', affinity=kernel).fit(data)
    embedding = SpectralEmbedding(n_components=k, affinity=kernel).fit_transform(data)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1 = axes[0]
    ax2 = axes[1]
    ax1.scatter(data[:, 0], data[:, 1], c=spectral.labels_, cmap=plt.cm.brg)
    if k == 2:
        ax2.scatter(embedding[:, 0], embedding[:, 1], c=spectral.labels_, cmap=plt.cm.brg)
    else:
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax2.set_xticks([])
        ax2 = fig.add_subplot(1,2,2,projection='3d')
        ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=spectral.labels_, cmap=plt.cm.brg)
    plt.show()


from sklearn.metrics import silhouette_score, silhouette_samples

def silhouette_kmeans(data, k):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (k + 1) * 10])
    clusterer = KMeans(n_clusters=k)
    cluster_labels = clusterer.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)
    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.coolwarm(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10 
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = plt.cm.coolwarm(cluster_labels.astype(float) / k)
    ax2.scatter(data[:, 0], data[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    centers = clusterer.cluster_centers_   
    ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
    plt.suptitle(f"Silhouette analysis for KMeans with {k} clusters", fontsize=14, fontweight="bold")
    plt.show()