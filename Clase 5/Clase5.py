#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib import offsetbox
import numpy as np
import math
from matplotlib.colors import ListedColormap
import seaborn as sns
from matplotlib import gridspec


from sklearn.decomposition import PCA

def draw_vector(text, v0, v1, ax=None, color='black'):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0, color=color)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    ax.text(v1[0], v1[1], text, fontsize=8, ha='right', va='bottom', color=color, weight='bold')


def demo_pca(X, pca):
    # Crear subplots con 1 fila y 2 columnas
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[1].scatter(X[:, 0], X[:, 1], alpha=0.2, c='k')
    colors = plt.cm.bwr(np.linspace(0, 1, len(pca.components_)))  # Change colormap as needed
    for n, (length, vector, color) in enumerate(zip(pca.explained_variance_, pca.components_, colors)):
        v = vector * 3 * np.sqrt(length)
        draw_vector(f'PC{n + 1}', pca.mean_, pca.mean_ + v, ax=axs[1], color=color)
    axs[1].axis('equal')

    # Crear datos para la tabla
    table_data = []
    for i, (pc, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        x, y = pc
        table_data.append(["PC{}".format(i + 1), round(x, 2), round(y, 2), round(var, 2)])

    # Crear la tabla y mostrarla en el subplot[0]
    table = axs[0].table(cellText=table_data, 
                         colLabels=["", "x", "y", "var"], 
                         cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 1.8)  # Ajustar el tamaño de la tabla según sea necesario

    axs[0].axis('off')  # Desactivar ejes en la primera columna

    plt.show()


def plot_projection(X, X_pca, pca, axis_equal=False):
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # First subplot: Original data
    axes[0].scatter(X[:, 0], X[:, 1], alpha=0.2, c='k')
    colors = plt.cm.bwr(np.linspace(0, 1, len(pca.components_)))
    for n, (length, vector, color) in enumerate(zip(pca.explained_variance_, pca.components_, colors)):
        v = vector * 3 * np.sqrt(length)
        draw_vector(f'PC{n + 1}', pca.mean_, pca.mean_ + v, ax=axes[0], color=color)
    axes[0].axis('equal')
    axes[0].set_title('Original Data')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    # Second subplot: Projection onto principal components
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2, c='k')
    axes[1].set_title('Projection onto Principal Components')
    axes[1].set_xlabel('component 1')
    axes[1].set_ylabel('component 2')
    axes[1].arrow(0, 0, 2, 0, color='b', alpha=0.8, width=0.02, head_width=0.1, head_length=0.3)
    axes[1].arrow(0, 0, 0, .3, color='r', alpha=0.8, width=0.05, head_width=0.2, head_length=0.08)
    if axis_equal:
        axes[1].axis('equal')

    # Show the plots
    plt.tight_layout()
    plt.show()


from sklearn.datasets import load_digits


def plot_digits(data, labels, n=10): 
    fig = plt.figure(figsize=(10,math.ceil(n/10)))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(n):
        ax = fig.add_subplot(math.ceil(n/10), 10, i + 1, xticks=[], yticks=[])
        ax.imshow(data[i].reshape(8,8), cmap=plt.cm.binary)
        ax.set_title(labels[i])

def plot_embedding(X, digits, name=None, save=False, fsize=(6,6)):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    palette = {0:'black',1:'magenta',2:'b',3:'c',4:'g',5:'lime',6:'gold',7:'orange',8:'r',9:'gray'}
    plt.figure(figsize = fsize)
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=palette[digits.target[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    shown_images = np.array([[1., 1.]])
    for i in range(digits.data.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:          
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
            X[i])
        ax.add_artist(imagebox)
    if name:
        plt.title(name, fontsize=16)
    if save:
        if name is None:
            name = "figura"
        plt.savefig(f"{name}.pdf")
    plt.show()    

def naive_dimensionality_reduction(X, indx=0):
    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(2, 10, 
                           height_ratios=[1, 1], 
                           width_ratios=[2] + [1] * 8 + [2], 
                           hspace=0)
    
    # Plot X_digit at the beginning
    digit = X[indx].reshape(8, 8)
    ax0 = plt.subplot(gs[:, 0])
    ax0.imshow(digit, cmap='gray_r')
    ax0.set_title("True")
    ax0.set_xticks([])
    ax0.set_yticks([])

    cumulative = np.zeros(64).reshape(8, 8)
    # Plot individual pixels and cumulative contribution
    for i in range(8):
        # Individual pixel
        pixel_value = np.zeros(64).reshape(8, 8)
        pixel_value[0, i] = 1
        value = digit[0, i]
        cumulative[0, i] = value

        ax1 = plt.subplot(gs[0, i + 1])
        ax1.imshow(pixel_value, cmap='gray_r')
        ax1.set_title(f"$c_{i+1}$")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Cumulative contribution
        ax2 = plt.subplot(gs[1, i+1])
        ax2.imshow(cumulative, cmap='gray_r')
        ax2.set_title(f"${value} \cdot c_{i+1}$")
        ax2.set_xticks([])
        ax2.set_yticks([])

    # Plot cumulative at the end
    ax3 = plt.subplot(gs[:, 9])
    ax3.imshow(cumulative, cmap='gray_r')
    ax3.set_title("Approx")
    ax3.set_xticks([])
    ax3.set_yticks([])

    fig.set_size_inches(20,4)
    # Adjust layout
    plt.tight_layout()
    plt.show()


def PCA_basis_functions(X, indx=0):
    # Compute the mean of the data
    data_mean = np.mean(X, axis=0)

    # Perform PCA with 64 components (original dimensionality)
    pca = PCA(n_components=64)
    X_pca = pca.fit_transform(X)

    eigenvalues = pca.explained_variance_

    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(2, 11, 
                           height_ratios=[1, 1], width_ratios=[2] + [1] * 9 + [2], 
                           hspace=0)

    # Plot X_digit at the beginning
    ax0 = plt.subplot(gs[:, 0])
    ax0.imshow(X[indx].reshape(8, 8), cmap='gray_r')
    ax0.set_title("True")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(data_mean.reshape(8, 8), cmap='gray_r')
    ax1.set_title('$\mu$')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1 = plt.subplot(gs[1, 1])
    ax1.imshow(data_mean.reshape(8, 8), cmap='gray_r')
    ax1.set_title('$1 \cdot \mu$')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot individual pixels and cumulative contribution
    for i in range(8):
        # Individual pixel
        pixel_value = np.zeros(64).reshape(8, 8)
        pixel_value[0, i] = 1

        basis_function = pca.components_[i, :].reshape(8, 8)
        ax1 = plt.subplot(gs[0, i + 2])    
        ax1.imshow(basis_function, cmap='gray_r')
        ax1.set_title(f"$c_{i+1}$")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Cumulative contribution
        ax2 = plt.subplot(gs[1, i+2])
        cumulative = data_mean + np.dot(pca.components_[:(i+1), :].T, X_pca[0, :(i+1)])

        value = eigenvalues[i]

        ax2.imshow(cumulative.reshape(8,8), cmap='gray_r')
        ax2.set_title("${:.2f} \cdot c_{}$".format(value,i+1))
        ax2.set_xticks([])
        ax2.set_yticks([])

    # Take the mean plus the weighted sum of the first eight principal components
    mean_plus_eight_components = data_mean + np.dot(pca.components_[:8, :].T, X_pca[0, :8])

    approx_image = mean_plus_eight_components.reshape(8, 8)
    # Plot cumulative at the end
    ax3 = plt.subplot(gs[:, 10])
    ax3.imshow(approx_image, cmap='gray_r')
    ax3.set_title("Approx")
    ax3.set_xticks([])
    ax3.set_yticks([])

    fig.set_size_inches(20,4)

    # Adjust layout
    plt.tight_layout()
    plt.show()


from sklearn.datasets import fetch_lfw_people

def plot_reconstruction(data, projected):
    fig, ax = plt.subplots(2, 10, figsize=(10, 2.5), 
                           subplot_kw={'xticks':[], 'yticks':[]}, 
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i in range(10):
        ax[0, i].imshow(data[i].reshape(62, 47), cmap='binary_r')
        ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
    ax[0, 0].set_ylabel('full-dim\ninput')
    ax[1, 0].set_ylabel('150-dim\nreconstruction')
    plt.show()


from sklearn.manifold import TSNE
from ipywidgets import interact, fixed, widgets

def tsne_vis(X_digits, digits, perplexity_values):
    def tsne_func(X_digits, perplexity):
        tsne = TSNE(n_components=2, 
                    perplexity=perplexity, 
                    learning_rate='auto', 
                    n_iter=300, 
                    init='random', 
                    verbose=False)
        projected_tsne = tsne.fit_transform(X_digits)
        plot_embedding(projected_tsne, digits, "t-SNE perplexity {}".format(perplexity))
    interact(
        tsne_func,
        X_digits=widgets.fixed(X_digits),
        digits=widgets.fixed(digits),
        perplexity = widgets.Dropdown(
            options=perplexity_values,
            value=30,  
            description='Perplexity',
            disabled=False,
        )
    )
