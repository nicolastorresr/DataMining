#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings("ignore")
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error


# In[2]:


def clustering(n_samples=200, 
               centers=3, 
               cluster_std=0.60, 
               random_state=0, 
               n_clusters=3,
               cmap='brg'):
    X, y_true = make_blobs(n_samples=n_samples,
                           centers=centers,
                           cluster_std=cluster_std,
                           random_state=random_state)
    X = X[:, ::-1]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5)) 
    axs[0].axis('equal')
    axs[1].axis('equal')
    
    axs[0].set_title('Datos no etiquetados')
    axs[1].set_title('Clustering')
    
    axs[0].scatter(X[:, 0], X[:, 1], c='k', zorder=2)    
    
    scatter = axs[1].scatter(X[:, 0], X[:, 1], c=labels, 
                             cmap=cmap, zorder=2)
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() 
             for i, center in enumerate(centers)]
    for i, (c, r) in enumerate(zip(centers, radii)):
        circle = plt.Circle(c, r, fc='none', lw=3, alpha=0.5, 
                            zorder=1, edgecolor=scatter.to_rgba(i))
        axs[1].add_patch(circle)
        
    plt.show()


# In[5]:


def classification(n_samples=30,
                   centers=2,
                   cluster_std=2.5, 
                   random_state=754,
                   C1=0.1,
                   gamma_1='auto',
                   C2=1,
                   gamma_2='auto',
                   C3=10,
                   gamma_3='auto',
                   new_data = False,
                   color_1='C8',
                   color_2='dodgerblue', 
                   marker_1 = 's',
                   marker_2 = 'o',
                   marker_size=70):
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    X, y = make_blobs(n_samples=n_samples,
                      centers=centers, 
                      cluster_std=cluster_std, 
                      random_state=random_state)
    if new_data:
        X2, y2 = make_blobs(10, centers=centers, 
                            cluster_std=cluster_std+0.5, random_state=random_state)
    
    model1 = svm.SVC(C=C1, gamma=gamma_1).fit(X, y)
    model2 = svm.SVC(C=C2, gamma=gamma_2).fit(X, y)
    model3 = svm.SVC(C=C3, gamma=gamma_3).fit(X, y)
    
    ax[0].scatter(X[y==1][:, 0], X[y==1][:, 1], marker=marker_1, color=color_1, s=marker_size)
    ax[0].scatter(X[y==0][:, 0], X[y==0][:, 1], marker=marker_2, color=color_2, s=marker_size)   
    
    ax[0].text(0.02, 0.98, 
               "Training Error: Accuracy = {0:.2f}".format(accuracy_score(y, model1.predict(X))),
                ha='left', va='top', transform=ax[0].transAxes, color='gray')
    
    if new_data:
        ax[0].scatter(X2[(y2==1) & (y2==model1.predict(X2))][:, 0], 
                      X2[(y2==1) & (y2==model1.predict(X2))][:, 1], 
                      marker=marker_1, color=None, edgecolor='blue', facecolor=color_1, s=marker_size)
        ax[0].scatter(X2[(y2==1) & (y2!=model1.predict(X2))][:, 0], 
                      X2[(y2==1) & (y2!=model1.predict(X2))][:, 1], 
                      marker=marker_1, color=None, edgecolor='red', facecolor=color_1, s=marker_size)
        
        ax[0].scatter(X2[(y2==0) & (y2==model1.predict(X2))][:, 0], 
                      X2[(y2==0) & (y2==model1.predict(X2))][:, 1], 
                      marker=marker_2, color=None, edgecolor='blue', facecolor=color_2, s=marker_size)
        ax[0].scatter(X2[(y2==0) & (y2!=model1.predict(X2))][:, 0], 
                      X2[(y2==0) & (y2!=model1.predict(X2))][:, 1], 
                      marker=marker_2, color=None, edgecolor='red', facecolor=color_2, s=marker_size)
        
        ax[0].text(0.02, 0.91, 
                   "Testing Error: Accuracy = {0:.2f}".format(accuracy_score(y2, model1.predict(X2))),
                   ha='left', va='top', transform=ax[0].transAxes, color='tomato')
    
    DecisionBoundaryDisplay.from_estimator(model1, 
                                           X, 
                                           plot_method="contour", 
                                           colors="k", 
                                           levels=[0], 
                                           alpha=0.5, 
                                           ax=ax[0])
    
    ax[1].scatter(X[y==1][:, 0], X[y==1][:, 1], marker=marker_1, color=color_1, s=marker_size)
    ax[1].scatter(X[y==0][:, 0], X[y==0][:, 1], marker=marker_2, color=color_2, s=marker_size)  
    
    ax[1].text(0.02, 0.98, 
               "Training Error: Accuracy = {0:.2f}".format(accuracy_score(y, model2.predict(X))),
                ha='left', va='top', transform=ax[1].transAxes, color='gray')
    
    if new_data:
        ax[1].scatter(X2[(y2==1) & (y2==model2.predict(X2))][:, 0], 
                      X2[(y2==1) & (y2==model2.predict(X2))][:, 1], 
                      marker=marker_1, color=None, edgecolor='blue', facecolor=color_1, s=marker_size)
        ax[1].scatter(X2[(y2==1) & (y2!=model2.predict(X2))][:, 0], 
                      X2[(y2==1) & (y2!=model2.predict(X2))][:, 1], 
                      marker=marker_1, color=None, edgecolor='red', facecolor=color_1, s=marker_size)
        
        ax[1].scatter(X2[(y2==0) & (y2==model2.predict(X2))][:, 0], 
                      X2[(y2==0) & (y2==model2.predict(X2))][:, 1], 
                      marker=marker_2, color=None, edgecolor='blue', facecolor=color_2, s=marker_size)
        ax[1].scatter(X2[(y2==0) & (y2!=model2.predict(X2))][:, 0], 
                      X2[(y2==0) & (y2!=model2.predict(X2))][:, 1], 
                      marker=marker_2, color=None, edgecolor='red', facecolor=color_2, s=marker_size)
        ax[1].text(0.02, 0.91, 
                   "Testing Error: Accuracy = {0:.2f}".format(accuracy_score(y2, model2.predict(X2))),
                   ha='left', va='top', transform=ax[1].transAxes, color='tomato')
        
    DecisionBoundaryDisplay.from_estimator(model2, 
                                           X, 
                                           plot_method="contour", 
                                           colors="k", 
                                           levels=[0], 
                                           alpha=0.5, 
                                           ax=ax[1])
    
    ax[2].scatter(X[y==1][:, 0], X[y==1][:, 1], marker=marker_1, color=color_1, s=marker_size)
    ax[2].scatter(X[y==0][:, 0], X[y==0][:, 1], marker=marker_2, color=color_2, s=marker_size)    
    
    ax[2].text(0.02, 0.98, 
               "Training Error: Accuracy = {0:.2f}".format(accuracy_score(y, model3.predict(X))),
                ha='left', va='top', transform=ax[2].transAxes, color='gray')
    
    if new_data:
        ax[2].scatter(X2[(y2==1) & (y2==model3.predict(X2))][:, 0], 
                      X2[(y2==1) & (y2==model3.predict(X2))][:, 1], 
                      marker=marker_1, color=None, edgecolor='blue', facecolor=color_1, s=marker_size)
        ax[2].scatter(X2[(y2==1) & (y2!=model3.predict(X2))][:, 0], 
                      X2[(y2==1) & (y2!=model3.predict(X2))][:, 1], 
                      marker=marker_1, color=None, edgecolor='red', facecolor=color_1, s=marker_size)
        
        ax[2].scatter(X2[(y2==0) & (y2==model3.predict(X2))][:, 0], 
                      X2[(y2==0) & (y2==model3.predict(X2))][:, 1], 
                      marker=marker_2, color=None, edgecolor='blue', facecolor=color_2, s=marker_size)
        ax[2].scatter(X2[(y2==0) & (y2!=model3.predict(X2))][:, 0], 
                      X2[(y2==0) & (y2!=model3.predict(X2))][:, 1], 
                      marker=marker_2, color=None, edgecolor='red', facecolor=color_2, s=marker_size)
        ax[2].text(0.02, 0.91, 
                   "Testing Error: Accuracy = {0:.2f}".format(accuracy_score(y2, model3.predict(X2))),
                   ha='left', va='top', transform=ax[2].transAxes, color='tomato')
        
    DecisionBoundaryDisplay.from_estimator(model3, 
                                           X, 
                                           plot_method="contour", 
                                           colors="k", 
                                           levels=[0], 
                                           alpha=0.5, 
                                           ax=ax[2])
    
    
    plt.show()


# In[7]:


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


# In[8]:


def make_data(N=30, err=0.8, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


# In[9]:


def regression(deg_1=1, 
              deg_2=5, 
              deg_3=20,
              N=30, 
              err=0.8, 
              rseed=1,
              new_data = False,
             ):
    X, y = make_data(N=N, err=err, rseed=rseed)
    xfit = np.linspace(-0.1, 1.0, 1000)[:, None]
    model1 = PolynomialRegression(deg_1).fit(X, y)
    model2 = PolynomialRegression(deg_2).fit(X, y)
    model3 = PolynomialRegression(deg_3).fit(X, y)
    
    if new_data:
        X2, y2 = make_data(10, rseed=42)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

    ax[0].scatter(X.ravel(), y, c='dodgerblue')
    ax[0].plot(xfit.ravel(), model1.predict(xfit), color='gray')
    ax[0].axis([-0.1, 1.0, -2, 14])
    ax[0].set_title('Grado {}'.format(deg_1))
    
    ax[0].text(0.02, 0.98, 
               "Training Error: MAE = {0:.2f}".format(mean_absolute_error(y, model1.predict(X))),
                ha='left', va='top', transform=ax[0].transAxes, color='dodgerblue')
    
    if new_data:
        ax[0].scatter(X2.ravel(), y2, c='tomato')
        ax[0].text(0.02, 0.91, 
                   "Testing Error: MAE = {0:.2f}".format(mean_absolute_error(y2, model1.predict(X2))),
                   ha='left', va='top', transform=ax[0].transAxes, color='tomato')

    ax[1].scatter(X.ravel(), y, c='dodgerblue')
    ax[1].plot(xfit.ravel(), model2.predict(xfit), color='gray')
    ax[1].axis([-0.1, 1.0, -2, 14])
    ax[1].set_title('Grado {}'.format(deg_2))
    
    ax[1].text(0.02, 0.98, 
               "Training Error: MAE = {0:.2f}".format(mean_absolute_error(y, model2.predict(X))),
               ha='left', va='top', transform=ax[1].transAxes, color='dodgerblue')
    
    if new_data:
        ax[1].scatter(X2.ravel(), y2, c='tomato')
        ax[1].text(0.02, 0.91, 
                   "Testing Error: MAE = {0:.2f}".format(mean_absolute_error(y2, model2.predict(X2))),
                   ha='left', va='top', transform=ax[1].transAxes, color='tomato')
    
    ax[2].scatter(X.ravel(), y, c='dodgerblue')
    ax[2].plot(xfit.ravel(), model3.predict(xfit), color='gray')
    ax[2].axis([-0.1, 1.0, -2, 14])
    ax[2].set_title('Grado {}'.format(deg_3))
    
    ax[2].text(0.02, 0.98, 
               "Training Error: MAE = {0:.2f}".format(mean_absolute_error(y, model3.predict(X))),
               ha='left', va='top', transform=ax[2].transAxes, color='dodgerblue')
    
    if new_data:
        ax[2].scatter(X2.ravel(), y2, c='tomato')
        ax[2].text(0.02, 0.91, 
                   "Testing Error: MAE = {0:.2f}".format(mean_absolute_error(y2, model3.predict(X2))),
                   ha='left', va='top', transform=ax[2].transAxes, color='tomato')


# In[11]:


def sparsity(n_data = 50, n_parts = 5, n_dimensions = 10):
    dimensions = np.arange(1, n_dimensions + 1)
    data_per_part = n_data / (n_parts ** dimensions)
    plt.figure(figsize=(5, 3))
    plt.plot(dimensions, data_per_part, marker='o', linestyle='-')
    plt.xlabel('Dimensionalidad')
    plt.ylabel('Datos por parte')
    plt.xticks(dimensions)
    plt.grid(True)
    plt.show()


# In[13]:


def sampling(n=8000, size_1 = 2000, size_2 = 100):
    x = np.linspace(0, 10, n)
    y = np.sin(x)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(x, y, '.', markersize=1, c='k')
    axs[0].set_title('{} puntos'.format(n))
    random_indices_size_1 = np.random.choice(len(x), size=size_1, replace=False)
    axs[1].plot(x[random_indices_size_1], y[random_indices_size_1], '.', markersize=1, c='k')
    axs[1].set_title('Muestra aleatoria de {} puntos'.format(size_1))
    random_indices_size_2 = np.random.choice(len(x), size=size_2, replace=False)
    axs[2].plot(x[random_indices_size_2], y[random_indices_size_2], '.', markersize=1, c='k')
    axs[2].set_title('Muestra aleatoria de {} puntos'.format(size_2))
    plt.tight_layout()
    plt.show()


# In[ ]:




