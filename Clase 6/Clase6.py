#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from ipywidgets import widgets, interact, fixed
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, _plot_dendrogram
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from matplotlib.patches import Circle, Ellipse


def clustering_data(df):
    # Define circles for the first plot
    circle1_names = {'Bernhard', 'Dennis', 'Andrew'}
    circle2_names = set(df['Name']) - circle1_names
    
    # Compute the mean position and radius for the first plot
    circle1_center = df[df['Name'].isin(circle1_names)][['Age', 'Salary']].mean()
    circle2_center = df[df['Name'].isin(circle2_names)][['Age', 'Salary']].mean()
    circle1_radius = np.max(np.linalg.norm(df[df['Name'].isin(circle1_names)][['Age', 'Salary']].values
                                           - circle1_center.values, axis=1))
    circle2_radius = np.max(np.linalg.norm(df[df['Name'].isin(circle2_names)][['Age', 'Salary']].values
                                           - circle2_center.values, axis=1))
    
    # Define circles for the second plot
    circle1_names_2 = {'Eve'}
    circle2_names_2 = {'Bernhard', 'Andrew'} 
    circle3_names_2 = set(df['Name']) - circle1_names_2 - circle2_names_2 - {'Dennis'}
    
    # Compute the mean position and radius for the second plot
    circle1_center_2 = df[df['Name'].isin(circle1_names_2)][['Age', 'Salary']].mean()
    circle2_center_2 = df[df['Name'].isin(circle2_names_2)][['Age', 'Salary']].mean()
    circle3_center_2 = df[df['Name'].isin(circle3_names_2)][['Age', 'Salary']].mean()
    circle1_radius_2 = np.max(np.linalg.norm(df[df['Name'].isin(circle1_names_2)][['Age', 'Salary']].values 
                                             - circle1_center_2.values, axis=1))
    circle2_radius_2 = np.max(np.linalg.norm(df[df['Name'].isin(circle2_names_2)][['Age', 'Salary']].values 
                                             - circle2_center_2.values, axis=1))
    circle3_radius_2 = np.max(np.linalg.norm(df[df['Name'].isin(circle3_names_2)][['Age', 'Salary']].values 
                                             - circle3_center_2.values, axis=1))

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # First plot
    axs[0].scatter(df['Age'], df['Salary'], s=50, 
                   facecolors='white', edgecolors='black')
    # Annotate each point with the corresponding Name
    for i, name in enumerate(df['Name']):
        axs[0].annotate(name, (df['Age'][i], df['Salary'][i]), 
                        textcoords="offset points", xytext=(0,5), ha='center')

    axs[0].set_title('Clustering 1')
    axs[0].set_xlabel('Age')
    axs[0].set_ylabel('Salary')


    # Draw circles for the first plot    
    ellipse1 = Ellipse((circle1_center['Age'], circle1_center['Salary']), 
                       angle=30,
                       width=circle1_radius*2.5, height=circle1_radius*1.2, 
                       color='deepskyblue', fill=False)
    
    ellipse2 = Ellipse((circle2_center['Age'], circle2_center['Salary']),
                       angle=30,
                       width=circle2_radius*2.2, height=circle2_radius*1.7, 
                       color='magenta', fill=False)
    
    axs[0].add_patch(ellipse1)
    axs[0].add_patch(ellipse2)

    # Second plot
    axs[1].scatter(df['Age'], df['Salary'], s=50, 
                   facecolors='white', edgecolors='black')
    # Annotate each point with the corresponding Name
    for i, name in enumerate(df['Name']):
        axs[1].annotate(name, (df['Age'][i], df['Salary'][i]), 
                        textcoords="offset points", xytext=(0,5), ha='center')

    axs[1].set_title('Clustering 2')
    axs[1].set_xlabel('Age')
    axs[1].set_ylabel('Salary')

    # Draw circles for the second plot
    circle1 = Circle((circle1_center_2['Age'], circle1_center_2['Salary']), 
                     radius=circle1_radius_2+5, 
                     color='green', fill=False)
    circle2 = Ellipse((circle2_center_2['Age'], circle2_center_2['Salary']), 
                      angle=-30,
                      width=circle2_radius_2*3.5, height=circle2_radius_2*1.5,
                      color='blue', fill=False)
    circle3 = Ellipse((circle3_center_2['Age'], circle3_center_2['Salary']), 
                      angle=-30,
                      width=circle3_radius_2*3, height=circle3_radius_2*2,
                      color='red', fill=False)
    axs[1].add_patch(circle1)
    axs[1].add_patch(circle2)
    axs[1].add_patch(circle3)

    plt.tight_layout()
    plt.show()


def kmeans_visualization(data, 
                         k=2, 
                         centroids='k-means++', 
                         iterations=0, 
                         n_init=10,
                         colormap_name='brg',
                         marker_centroid='s',
                         x_label = None,
                         y_label = None,
                         annotations = None,
                         font_size=10):
    
    def plot_kmeans(data, 
                    k, 
                    centroids, 
                    iterations, 
                    n_init, 
                    colormap_name, 
                    x_label, 
                    marker_centroid, 
                    y_label, 
                    annotations,
                    font_size):
        
        cmap = plt.colormaps.get_cmap(colormap_name)
        if iterations is None: iterations = 300
        if centroids in ('k-means++','random'):
            if k is None:
                print("Debe indicar K o los centroides iniciales.")
                return
            if k < 1:
                print("El valor de K debe ser mayor que cero.")
                return
            if k > len(data):
                print("El valor de K debe ser menor o igual al número de datos.")
                return
            cc = range(k)
        else:
            if isinstance(centroids, list):
                if len(centroids) > 0:
                    cc = range(len(centroids))
                    if k is None:
                        k = len(centroids)
                    else:
                        if k != len(centroids):                            
                            print("El valor de K no coincide con el número de centroides.")
                            return
                else:
                    print("El número de centroides debe ser mayor que cero.")
                    return    
            else:
                print("Los centroides iniciales no son válidos.")
                return
            centroids = np.array(centroids)        
        norm = plt.Normalize(vmin=min(cc), vmax=max(cc))
        if iterations == 0:
            # Plot the data
            plt.scatter(data[:, 0], data[:, 1], c='black', s=70)
            if centroids not in ('k-means++','random'):
                # Plot centroids
                plt.scatter(centroids[:,0], centroids[:,1], cmap=cmap, edgecolors='w',
                            c=cc, alpha=0.4, marker=marker_centroid, s=250, linewidth=2)
                for i,cx in enumerate(centroids):
                    plt.annotate(f"C{i+1}", (cx[0], cx[1]), c=cmap(norm(cc[i])),
                                 textcoords="offset points", xytext=(5,0), ha='left')
            plt.title(f"K-Means\nIteración {iterations}", fontsize = 15)
        else:
            if iterations == 1:
                if centroids not in ('k-means++','random'):
                    old_centers = centroids
            else:
                prekmeans = KMeans(n_clusters=k, init=centroids, 
                                   n_init=n_init, max_iter=iterations-1).fit(data)
                old_centers = prekmeans.cluster_centers_


            kmeans = KMeans(n_clusters=k, init=centroids, n_init=n_init, 
                            max_iter=iterations).fit(data)
            centers = kmeans.cluster_centers_

            # Plot the data
            plt.scatter(data[:, 0], data[:, 1], cmap=cmap,
                        c=kmeans.labels_, s=70)
            if centroids not in ('k-means++','random'):
                # Plot previous centroids
                plt.scatter(old_centers[:,0], old_centers[:,1], cmap=cmap, facecolors='none',
                            alpha=0.4, marker=marker_centroid, s=250, 
                            linewidth=2, edgecolors=cmap(norm(cc)))
            # Plot centroids
            plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], cmap=cmap,
                        c=cc, alpha=0.5, marker=marker_centroid, s=250, linewidth=2, edgecolors='w')
            for i,cx in enumerate(kmeans.cluster_centers_):
                plt.annotate(f"C{i+1}", (cx[0], cx[1]), c=cmap(norm(cc[i])),
                             textcoords="offset points", xytext=(5,0), ha='left')


            plt.title(f"K-Means\nIteración {iterations}", fontsize = 15) 
            if centroids not in ('k-means++','random'):
                for i in range(len(centers)):
                    plt.annotate('', centers[i], old_centers[i],
                                 arrowprops=dict(arrowstyle='->', linewidth=1,
                                                 color=cmap(norm(cc[i])) ))

        if annotations is not None:
            # Annotate each point with the corresponding Name
            for i,name in enumerate(annotations):
                plt.annotate(name, (data[i, 0], data[i, 1]), fontsize=font_size,
                             textcoords="offset points", xytext=(0,5), ha='center')

        plt.grid(linestyle="dotted")
        if centroids not in ('k-means++','random'):
            # Concatenate the arrays
            combined_data = np.concatenate([data, centroids])
        else:
            combined_data = np.array(data)
        max_x = np.max(combined_data[:, 0])
        max_y = np.max(combined_data[:, 1])
        min_x = np.min(combined_data[:, 0])
        min_y = np.min(combined_data[:, 1])
        plt.xlim((min_x,max_x))
        plt.ylim((min_y,max_y))

        # Set the original axis limits
        original_xlim = plt.xlim()
        original_ylim = plt.ylim()

        # Define margins
        x_margin = (max_x-min_x)/10
        y_margin = (max_y-min_y)/10

        # Add margins to the axis limits
        new_xlim = (original_xlim[0] - x_margin, original_xlim[1] + x_margin)
        new_ylim = (original_ylim[0] - y_margin, original_ylim[1] + y_margin)

        # Set the updated axis limits
        plt.xlim(new_xlim)
        plt.ylim(new_ylim)

        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)

        plt.show()
    
    if centroids not in ('k-means++','random'):
        k = len(centroids)
    
    interact(
        plot_kmeans,
        
        data=widgets.fixed(data),
        k=widgets.fixed(k),  # Fixed value for 'k' based on the number of centroids
        centroids=widgets.fixed(centroids),  # Fixed value for 'centroids'
        
        iterations=widgets.IntSlider(min=0, max=5, step=1, value=0, description='Iteración'),
        
        n_init=widgets.fixed(n_init),  # Fixed value for 'n_init' or customize as needed
        colormap_name=widgets.fixed(colormap_name),  # Fixed value for 'colormap_name' or customize as needed
        x_label=widgets.fixed(x_label),  # Fixed value for 'x_label' or customize as needed
        marker_centroid=widgets.fixed(marker_centroid),  # Fixed value for 'marker_centroid' or customize as needed
        y_label=widgets.fixed(y_label),  # Fixed value for 'y_label' or customize as needed
        annotations=widgets.fixed(annotations),  # Fixed value for 'annotations' or customize as needed
        font_size=widgets.fixed(font_size)
    )


# In[9]:


def range_SSE(data,j):
    SSE = []
    for k in range(1,j):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        SSE.append(kmeans.inertia_)
    return SSE


# In[10]:


def elbow(data,j):
    if len(data) < j:
        print("El número de datos ({}) no puede ser menor al número de clusters ({}).".format(len(data),j))
        return 
    plt.figure(figsize = (6,3))
    plt.plot(range(1,j+1), range_SSE(data,j+1), marker='o', linestyle='-',
             color='black', markeredgecolor='k', markerfacecolor='white')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Inercia (SSE)')
    plt.grid(linestyle="dotted")
    plt.xticks(range(1,j+1))
    plt.show()


# In[12]:


from sklearn.datasets import make_moons


# In[13]:


def plot_kmeans_moons(data, k=2):
    # Configura los subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de puntos para make_moons
    axs[0].scatter(data[:, 0], data[:, 1], c='black')
    axs[0].grid(linestyle="dotted")
    axs[0].set_title("Make Moons")
    
    # Aplica K-Means
    kmeans = KMeans(n_clusters=k).fit(data)
    
    # Gráfico de puntos para K-Means
    axs[1].scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap=plt.cm.brg)
    axs[1].grid(linestyle="dotted")
    axs[1].set_title("K-Means")
    
    # Muestra los subplots
    plt.show()


# In[15]:


from sklearn.cluster import DBSCAN


# In[16]:


def types_of_points(labels, core_samples_mask):
    result = []
    for k,j in zip(labels,core_samples_mask):
        if k == -1:
            result.append('Noise')
        else:
            if j:
                result.append('Core')
            else:
                result.append('Border')
    return np.array(result)


# In[17]:


def density_based_points(data, 
                         Eps=0.1, 
                         MinPts=2, 
                         annotations=None, 
                         font_size=10,
                         x_label=None,
                         y_label=None
                        ):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=Eps, min_samples=MinPts)
    labels = dbscan.fit_predict(data)
    
    # Plot points with different colors based on DBSCAN labels
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    
    classes = types_of_points(labels,core_samples_mask)
    
    for i,c in enumerate(set(['Noise','Border','Core'])):
        if c == "Noise":
            col = 'k'  
            mark = '.'
            edge_color=None
        elif c == "Border":
            col = 'y'  
            mark = '^'
            edge_color='k'
        else:
            col = 'r'  
            mark = 'd'
            edge_color=None
        plt.scatter(data[classes == c, 0], 
                    data[classes == c, 1], 
                    c=col, 
                    marker=mark,
                    label=c,
                    edgecolor=edge_color, 
                    )
        
    # Plot circles around core points
    for i, (x, y) in enumerate(data):
        c = plt.Circle((x, y), 
                       Eps, 
                       color='k', 
                       fill=None, 
                       alpha=0.2, 
                       linestyle='dashed')
        plt.gca().add_patch(c)
        
    if annotations is not None:
        # Annotate each point with the corresponding Name
        for i,name in enumerate(annotations):
            plt.annotate(name, (data[i, 0], data[i, 1]), fontsize=font_size,
                         textcoords="offset points", xytext=(0,5), ha='center')   
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    # Add legend
    plt.legend()
    plt.show()


# In[19]:


def dbscan_visualization(data, 
                        Eps=(0.1,1,0.1,0.5), 
                        MinPts=(2,10,1,5), 
                        scale=0.5, 
                        annotations=None, 
                        font_size=10, 
                        x_label=None, 
                        y_label=None):

    def plot_dbscan(data, 
                    Eps, 
                    MinPts, 
                    scale, 
                    annotations, 
                    font_size, 
                    x_label, 
                    y_label):    

        model = DBSCAN(eps=Eps, min_samples=MinPts).fit(data)    
        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True        
        colors = [plt.cm.brg(each) for each in np.linspace(0, 1, len(set(model.labels_)))]
        for k, col in zip(set(model.labels_), colors):
            if k == -1:
                col = [0, 0, 0, 0.1]            
            class_member_mask = (model.labels_ == k)
            xy = data[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=scale*20)
            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=scale*10)        
        n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
        if annotations is not None:
            # Annotate each point with the corresponding Name
            for i,name in enumerate(annotations):
                plt.annotate(name, (data[i, 0], data[i, 1]), fontsize=font_size,
                             textcoords="offset points", xytext=(0,5), ha='center')   
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        plt.title(f"DBSCAN\n{n_clusters} clusters", fontsize = 15)             
        plt.grid(linestyle="dotted")
        plt.show()
        
    interact(
        plot_dbscan,
        data=widgets.fixed(data),
        Eps = widgets.FloatSlider(min=Eps[0], max=Eps[1], step=Eps[2], value=Eps[3], description='Eps'),
        MinPts=widgets.IntSlider(min=MinPts[0], max=MinPts[1], step=MinPts[2], value=MinPts[3], description='MinPts'),
        scale=widgets.fixed(scale),  
        x_label=widgets.fixed(x_label),  # Fixed value for 'x_label' or customize as needed        
        y_label=widgets.fixed(y_label),  # Fixed value for 'y_label' or customize as needed
        annotations=widgets.fixed(annotations),  # Fixed value for 'annotations' or customize as needed
        font_size=widgets.fixed(font_size)
    )


# In[24]:


from sklearn.cluster import AgglomerativeClustering


# In[25]:


def hac_visualization(X, 
                      method='single', 
                      labels=None, 
                      font_size=10, 
                      leaf_font_size=None, 
                      leaf_rotation=None):
   
    # Inicializar variables
    n = X.shape[0]
    linkage_matrix = linkage(X, method)

    def plot_step(i):
        
        clusters = dendrogram(linkage_matrix, labels=labels, no_plot=True)
        
        if i < 1:
            cut_height = 0
        else:
            cut_height = linkage_matrix[i-1,2]
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))                        
        
        cut_dendrogram = {'icoord': [], 
                          'dcoord': [], 
                          'ivl': [], 
                          'leaves': [], 
                          'color_list': [], 
                          'leaves_color_list': []}
        
        for xs, ys, tags, leaves, clr, leaves_color in zip(clusters['icoord'], 
                                                           clusters['dcoord'], 
                                                           clusters['ivl'],
                                                           clusters['leaves'], 
                                                           clusters['color_list'], 
                                                           clusters['leaves_color_list']
                                                          ):
            if all(y <= cut_height for y in ys):
                cut_dendrogram['icoord'].append(xs)
                cut_dendrogram['dcoord'].append(ys)
                cut_dendrogram['ivl'].append(tags)
                cut_dendrogram['leaves'].append(leaves)
                cut_dendrogram['color_list'].append(clr)
                cut_dendrogram['leaves_color_list'].append(leaves_color)

        mh = max(linkage_matrix[:, 2])
        
        _plot_dendrogram(icoords=cut_dendrogram['icoord'], 
                         dcoords=cut_dendrogram['dcoord'], 
                         ivl=clusters['ivl'], 
                         p=30, 
                         n=n, 
                         mh=mh, 
                         orientation='top',
                         no_labels=False, 
                         color_list=cut_dendrogram['color_list'],
                         leaf_font_size=leaf_font_size,
                         leaf_rotation=leaf_rotation,
                         contraction_marks=None,
                         ax=axs[1],
                         above_threshold_color='C0')
        
        axs[0].grid(linestyle="dotted")
        axs[1].grid(linestyle="dotted")
        
        assignments = fcluster(linkage_matrix, t=cut_height, criterion='distance')
        
        color_cycle = [f'C{i}' for i in range(10)]

        # Map numbers to colors using the color cycle
        colors = [color_cycle[number % len(color_cycle)] for number in assignments]
        
        axs[0].scatter(X[:,0], X[:,1], c=colors)

        if labels is not None:
            for label, x, y in zip(labels, X[:, 0], X[:, 1]):             
                axs[0].annotate(label, xy=(x, y), fontsize=font_size,
                             textcoords="offset points", xytext=(0,5), ha='center')  
       
          
    # Crear interfaz interactiva
    interact(plot_step, i=widgets.IntSlider(min=0, max=n-1, step=1, value=0, description='Nivel'))


# In[27]:


def comparing_hac(data, k):
    plt.figure(figsize=(20, 3))
    prox = ["single","complete","average","ward"]
    for i in range(len(prox)):
        plt.subplot(1, 4, i+1)
        plt.title(prox[i])
        hac = AgglomerativeClustering(n_clusters=k, linkage=prox[i]).fit(data)
        plt.scatter(data[:, 0], data[:, 1], c=hac.labels_, cmap=plt.cm.brg)        
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    plt.show()

