#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact,fixed, widgets
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import Normalize
from sklearn.datasets import make_blobs
import pandas as pd
from random import sample

def plot_rating_matrix(m = 6, n = 4):
    # Generate random ratings for the matrix
    ratings = np.random.randint(0, 6, size=(n, m))

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(4, 2))

    # Create a table
    table = ax.table(cellText=ratings,
                     colLabels=[f"$i_{i}$" for i in range(1, m + 1)],
                     rowLabels=[f"$u_{j}$" for j in range(1, n + 1)],
                     loc='center', cellLoc='center')

    # Modify the appearance of the table
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2)

    # Remove axis
    ax.axis('off')

    # Show the plot
    plt.show()


def plot_knn(X, y, new, nn, weights='distance', data_color='k', knn_color='gray', c_map='bwr'): 
    if nn < 1:
        nn = 1
    if nn > len(X):
        nn = len(X)
    figure, axes = plt.subplots()
    plt.scatter(X[:,0], X[:,1], c=y, zorder=5, edgecolor='k', cmap=c_map)
    
    knn = KNeighborsClassifier(n_neighbors=nn, weights=weights).fit(X,y)
    neigh_dist, neigh_ind = knn.kneighbors([new], nn)
    for i in neigh_ind[0]:
        x_values = [new[0],X[i][0]]
        y_values = [new[1],X[i][1]]
        plt.plot(x_values, y_values, c=knn_color, linestyle=":")
    c = plt.Circle((new[0], new[1]), max(neigh_dist[0]), color=knn_color, fill = False, ls="--")   
        
    pred = knn.predict(np.array((new[0], new[1])).reshape(1,-1))
    
    # Define colormap and normalization
    cmap = plt.colormaps.get_cmap(c_map)
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())

    # Map predicted label to a value between 0 and 1
    pred_color = norm(pred)
    
    plt.scatter(new[0], new[1], marker='*', s=125, 
                c=cmap(norm(pred)), 
                edgecolor='k',
                zorder=5)    
    
    axes.text(0.02, 0.98,
                "Vecindario de {} con {} vecinos".format(new,nn),
                ha='left', va='top', transform=axes.transAxes,
                color=knn_color)
    plt.gca().add_patch(c)    


from sklearn.metrics.pairwise import cosine_similarity


def get_title(movie_data, movie_id):  
    return(movie_data[movie_data.movie_id == movie_id].title.values[0])


def get_id(movie_data, movie_title):
    return(movie_data[movie_data.title == movie_title].movie_id.values[0]) 


def adjust_title(title, max_char=False):
    if "(" in title:
        i = title.index("(")
        title = title[:i]
    if max_char:
        title = title[:max_char] + ".." if len(title) > max_char else title
    return title


def plot_latent_space(df_movies, df_sim_matrix, decomposed_matrix, factor=1, n=10):
    if factor < 1:
        factor = 1
    elif factor > len(df_sim_matrix):
        factor = len(df_sim_matrix)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in df_sim_matrix.sort_values(by = factor, ascending = False).index[:n]:
        title = get_title(df_movies, i)
        title = adjust_title(title)
        plt.scatter(decomposed_matrix[i-1][0], decomposed_matrix[i-1][1], c="black", 
                            marker=None, 
                            alpha=0, s=0)
        ax.text(decomposed_matrix[i-1][0], decomposed_matrix[i-1][1],
                "{}".format(title),
                ha='center', va='top', 
                color='k')
    plt.title(f"{n} películas más representativas del Espacio Latente {factor}")   
    plt.show()

def recomendar(df_movies, df_sim_matrix, titles, top_n=10):
    def topn(title, df_sim_matrix, top_n):
        print(f'Las {top_n} películas más similares a "{title}" son:')
        for n,item in enumerate(df_sim_matrix.sort_values(by = get_id(df_movies,title), 
                                                          ascending = False).index[1:top_n+1]):
            print(f' {n+1}.\t{get_title(df_movies,item)}')
    interact(
        topn,
        title = widgets.Dropdown(
            options=titles,
            value="Toy Story (1995)",  
            description='Película:',
            disabled=False,
        ),
        df_sim_matrix=widgets.fixed(df_sim_matrix),
        top_n=widgets.IntSlider(min=1, max=20, step=1, value=10, description='N:')
    )
