#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC
from mpl_toolkits import mplot3d

def discriminative(data='blobs',
                   n_samples=50, 
                   centers=2,
                   random_state=0, 
                   cluster_std=0.60,
                   noise=0.1,
                   factor=0.1,
                   color_1='C8',
                   color_2='dodgerblue', 
                   marker_1 = 's',
                   marker_2 = 'o',
                   edgecolor_1 = 'k',
                   edgecolor_2 = 'k',
                   marker_size=30, 
                   boundary=False, 
                   svm=False, 
                   show_vectors=False,
                   kernel="linear",
                   C=1,
                   z=False):
    
    if data == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=centers,
                      random_state=random_state, cluster_std=cluster_std)
    elif data == 'circles':
        X, y = make_circles(n_samples=n_samples, random_state=random_state,
                            factor=factor, noise=noise)
    else:
        try:
            X,y = data
            X = np.array(X)
            y = np.array(y)
        except:
            print("data is not valid.")
            return
    
    max_x = np.max(X[:, 0])
    max_y = np.max(X[:, 1])
    min_x = np.min(X[:, 0])
    min_y = np.min(X[:, 1])
    
    if z:
        ax = plt.subplot(projection='3d')
        if callable(z):
            Z = z(X)
        else:
            Z = np.array(z)
        ax.scatter3D(X[y==1][:, 0], X[y==1][:, 1], Z[y==1], edgecolor=edgecolor_1,
                     marker=marker_1, color=color_1, s=marker_size)
        ax.scatter3D(X[y==0][:, 0], X[y==0][:, 1], Z[y==0], edgecolor=edgecolor_2,
                     marker=marker_2, color=color_2, s=marker_size)  
        
        ax.view_init(elev=20, azim=30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        if boundary:
            #x1_lim = [0,3]
            #x2_lim = [0,3]
            x1s = np.linspace(min_x, max_x, 20)
            x2s = np.linspace(min_y, max_y, 20)
            x1, x2 = np.meshgrid(x1s, x2s)
            ax.plot_surface(x1s, x2, np.zeros_like(x1)+boundary, 
                            color="slategrey", alpha=0.5)
        
    else:
        plt.scatter(X[y==1][:, 0], X[y==1][:, 1], edgecolor=edgecolor_1,
                    marker=marker_1, color=color_1, s=marker_size)
        plt.scatter(X[y==0][:, 0], X[y==0][:, 1], edgecolor=edgecolor_2,
                    marker=marker_2, color=color_2, s=marker_size)  

        
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


        if boundary and not svm:
            xfit = np.linspace(new_xlim[0],new_xlim[1])                
            m, b = boundary        
            yfit = m * xfit + b
            plt.plot(xfit, yfit, color='slategrey')
            distances = np.abs((m * X[:, 0] - X[:, 1] + b) / np.sqrt(m**2 + 1))
            d = np.min(distances)
            plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                         color='lightgray', alpha=0.5)

            plt.text(0.05, 0.05,
                     "Margin: {0:.2f}".format(d),
                     ha='left', va='bottom', transform=plt.gca().transAxes,
                     color='slategrey')

        if svm:
            get_xlim = plt.xlim()
            get_ylim = plt.ylim() 
            model = SVC(kernel=kernel, C=C).fit(X, y)
            tmp_y, tmp_x = np.meshgrid(np.linspace(get_ylim[0], get_ylim[1]),
                                       np.linspace(get_xlim[0], get_xlim[1]))
            joint_xy = np.vstack([tmp_x.ravel(), tmp_y.ravel()]).T
            tmp_z = model.decision_function(joint_xy).reshape(tmp_x.shape)
            plt.contour(tmp_x, tmp_y, tmp_z, colors='slategrey', levels=[-1, 0, 1], linestyles=['--','-','--'])   


            # Get the support vectors and their coefficients
            support_vectors = model.support_vectors_
            support_vectors_coefficients = np.abs(model.dual_coef_)
            if show_vectors:
                plt.scatter(support_vectors[:,0],
                            support_vectors[:,1],
                            s=250, linewidth=3, facecolor='none',
                            edgecolor='slategrey')

            # Compute the distances from support vectors to the decision boundary
            if kernel == 'linear':
                distances_to_boundary = np.abs(model.decision_function(support_vectors)) / np.linalg.norm(model.coef_)

                # Find the maximum margin
                max_margin = np.min(distances_to_boundary)
                plt.text(0.05, 0.05,
                         "Margin: {0:.2f}".format(max_margin),
                         ha='left', va='bottom', transform=plt.gca().transAxes,
                         color='slategrey')

        plt.grid(linestyle="dotted")
        plt.show()

import itertools
from sklearn.inspection import DecisionBoundaryDisplay 
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

def scale_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val)
    scaled_arr = np.round(scaled_arr).astype(int)  # Round to nearest integer
    return scaled_arr

def multiclass_svm(n_samples=10, 
                   centers=3, 
                   n_features=2, 
                   random_state=22,
                   method=False):
    
    X, y = make_blobs(n_samples=n_samples, 
                      centers=centers, 
                      n_features=n_features, 
                      random_state=random_state)
    
    max_x = np.max(X[:, 0])
    max_y = np.max(X[:, 1])
    min_x = np.min(X[:, 0])
    min_y = np.min(X[:, 1])

    # Define margins
    x_margin = (max_x-min_x)/10
    y_margin = (max_y-min_y)/10

    # Add margins to the axis limits
    new_xlim = (min_x - x_margin, max_x + x_margin)
    new_ylim = (min_y - y_margin, max_y + y_margin)

    
    if method is False:
        plt.scatter(X[y==0][:, 0], X[y==0][:, 1], edgecolor='k',color='lightgray',marker='o')
        plt.scatter(X[y==1][:, 0], X[y==1][:, 1], edgecolor='k',color='lightgray',marker='s')
        plt.scatter(X[y==2][:, 0], X[y==2][:, 1], edgecolor='k',color='lightgray',marker='D')
        plt.grid(linestyle="dotted")
        # Set the updated axis limits
        plt.xlim(new_xlim)
        plt.ylim(new_ylim)
        plt.show()
    else:
        if method.lower() in ['ovr', 'ovo']:
            _, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))
            markers = ['o', 's', 'D']
            colors = ['C0', 'C3'] 
            if method.lower() == 'ovr':

                for i in range(len(np.unique(y))):

                    y_i = np.where(y==i,1,0)
                    clf = SVC().fit(X, y_i)
                    disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", 
                                                                  cmap=plt.cm.coolwarm,alpha=0.8, ax=axes[i])

                    predictions = clf.predict(X)

                    axes[i].scatter(X[y==0][:, 0], X[y==0][:, 1], marker='o', edgecolors="k",
                                    color=[colors[ii] for ii in predictions[y==0]])
                    axes[i].scatter(X[y==1][:, 0], X[y==1][:, 1], marker='s', edgecolors="k",
                                    color=[colors[ii] for ii in predictions[y==1]])
                    axes[i].scatter(X[y==2][:, 0], X[y==2][:, 1], marker='D', edgecolors="k",
                                    color=[colors[ii] for ii in predictions[y==2]])
                
                    axes[i].set_xlim(new_xlim)
                    axes[i].set_ylim(new_ylim)

                clf = OneVsRestClassifier(SVC()).fit(X, y)

            elif method.lower() == 'ovo':

                for n, tupla in enumerate(list(itertools.combinations(range(len(np.unique(y))),2))):
                    i,j=tupla
                    y_i = y[np.logical_or(y == i, y == j)]
                    X_i = X[ [k for k in range(len(y)) if (y[k] == i) or (y[k] == j)] ]
                    
                    axes[n].set_xlim(new_xlim)
                    axes[n].set_ylim(new_ylim)

                    clf = SVC().fit(X_i, y_i)  
                    disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", 
                                                                  cmap=plt.cm.coolwarm,alpha=0.8, ax=axes[n])

                    predictions = clf.predict(X_i)

                    predictions = scale_array(predictions)
                    

                    axes[n].scatter(X_i[y_i==0][:, 0], X_i[y_i==0][:, 1], 
                                    edgecolor='k',
                                    color=[colors[ii] for ii in predictions[y_i==0]],
                                    marker='o')
                    axes[n].scatter(X_i[y_i==1][:, 0], X_i[y_i==1][:, 1], 
                                    edgecolor='k',
                                    color=[colors[ii] for ii in predictions[y_i==1]],
                                    marker='s')
                    axes[n].scatter(X_i[y_i==2][:, 0], X_i[y_i==2][:, 1], 
                                    edgecolor='k',
                                    color=[colors[ii] for ii in predictions[y_i==2]],
                                    marker='D')
                    

                clf = OneVsOneClassifier(SVC()).fit(X, y)
            
            
            disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", 
                                                          cmap=plt.cm.coolwarm,alpha=0.8, ax=axes[3])

            colors = ['C0', 'lightgray', 'C3']
            predictions = clf.predict(X)

            axes[3].scatter(X[y==0][:, 0], X[y==0][:, 1], marker='o', edgecolors="k",
                            color=[colors[i] for i in predictions[y==0]])
            axes[3].scatter(X[y==1][:, 0], X[y==1][:, 1], marker='s', edgecolors="k",
                            color=[colors[i] for i in predictions[y==1]])
            axes[3].scatter(X[y==2][:, 0], X[y==2][:, 1], marker='D', edgecolors="k",
                            color=[colors[i] for i in predictions[y==2]])
            
            axes[3].set_xlim(new_xlim)
            axes[3].set_ylim(new_ylim)

            plt.show()
        else:
            print('invalid method')


from sklearn.datasets import fetch_lfw_people

def plot_faces(faces, 
               n_rows=3, 
               n_cols=10, 
               X_test=None,
               y_pred=None, 
               y_test=None,
               title=False):
    
    if n_rows == 1 and n_cols == 1:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        if X_test is not None and y_pred is not None and y_test is not None:
            ax.imshow(X_test[0].reshape(62, 47), cmap='binary_r')
            ax.set_ylabel(faces.target_names[y_pred[0]].split()[-1],
                   color='black' if y_pred[0] == y_test[0] else 'red')
        else:
            ax.imshow(faces.images[0], cmap='binary_r')
            ax.set_xlabel(faces.target_names[faces.target[0]])
    elif n_rows == 1:
        fig, ax = plt.subplots(1, n_cols, figsize=(2*n_cols, 2),
                               subplot_kw={'xticks': [], 'yticks': []},
                               gridspec_kw=dict(wspace=0.1))
        for i in range(n_cols):
            
            if X_test is not None and y_pred is not None and y_test is not None:
                ax[i].imshow(X_test[i].reshape(62, 47), cmap='binary_r')
                ax[i].set_ylabel(faces.target_names[y_pred[i]].split()[-1],
                   color='black' if y_pred[i] == y_test[i] else 'red')
            else:
                ax[i].imshow(faces.images[i], cmap='binary_r')
                ax[i].set_xlabel(faces.target_names[faces.target[i]])
    elif n_cols == 1:
        fig, ax = plt.subplots(n_rows, 1, figsize=(4, 2*n_rows),
                               subplot_kw={'xticks': [], 'yticks': []},
                               gridspec_kw=dict(hspace=0.15))
        for i in range(n_rows):
            
            if X_test is not None and y_pred is not None and y_test is not None:
                ax[i].imshow(X_test[i].reshape(62, 47), cmap='binary_r')
                ax[i].set_ylabel(faces.target_names[y_pred[i]].split()[-1],
                   color='black' if y_pred[i] == y_test[i] else 'red')
            else:
                ax[i].imshow(faces.images[i], cmap='binary_r')
                ax[i].set_xlabel(faces.target_names[faces.target[i]])

    else:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows),
                               subplot_kw={'xticks': [], 'yticks': []},
                               gridspec_kw=dict(hspace=0.15, wspace=0.1))
        for i, axi in enumerate(ax.flat):
            
            if X_test is not None and y_pred is not None and y_test is not None:
                axi.imshow(X_test[i].reshape(62, 47), cmap='binary_r')
                axi.set_ylabel(faces.target_names[y_pred[i]].split()[-1],
                   color='black' if y_pred[i] == y_test[i] else 'red')
            else:
                axi.imshow(faces.images[i], cmap='binary_r')
                axi.set_xlabel(faces.target_names[faces.target[i]])
    if title:
        fig.suptitle(title, size=14)
    plt.show()

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

def plot_confusion_matrix(faces, y_test, y_pred):
    plt.figure(figsize=(6,4))
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=faces.target_names, yticklabels=faces.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()
