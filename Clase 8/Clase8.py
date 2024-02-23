#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from ipywidgets import interact,fixed,widgets
import pandas as pd
from random import sample
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import Ridge,Lasso,LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error


import warnings
warnings.filterwarnings("ignore")


def linear_regression(x_coord, 
                      y_coord,
                      reg = False,
                      x_test = None,
                      x_label = None,
                      y_label = None):
    
    x_coord = np.array(x_coord)
    y_coord = np.array(y_coord)
    
    plt.scatter(x_coord, y_coord, c='dodgerblue', zorder=3)
    
    if reg:       
        plt.plot(x_coord, y_coord, c='gray', linestyle='-', zorder=2)
    
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.grid(linestyle="dotted")
    
    if x_test:
        pendiente = (y_coord[1]-y_coord[0])/(x_coord[1]-x_coord[0])
        intercepto = y_coord[0] - pendiente*x_coord[0]
        y_hat = pendiente*x_test + intercepto
        
        plt.scatter(x_test, y_hat, c='tomato', zorder=4)
        
        plt.hlines(y=y_hat, xmin=0, xmax=x_test, linestyles="dashdot", color='tomato')
        plt.vlines(x=x_test, ymin=0, ymax=y_hat, linestyles="dashdot", color='tomato')
        
        plt.text(0.05, 0.95,
                 "Prediction: {1} = {0:.2f}".format(y_hat,y_label),
                 ha='left', va='top', transform=plt.gca().transAxes,
                 color='tomato')
        
        x_test = np.array([x_test])
        y_hat = np.array([y_hat])
 
        max_x = max(np.concatenate([x_coord, x_test]))
        max_y = max(np.concatenate([y_coord, y_hat]))
        min_x = min(np.concatenate([x_coord, x_test]))
        min_y = min(np.concatenate([y_coord, y_hat]))
        
        plt.plot([min_x,max_x], 
                 [min_y,max_y] ,
                 c='gray', linestyle='--', zorder=1)
        
    else:
        max_x = max(x_coord)
        max_y = max(y_coord)
        min_x = min(x_coord)
        min_y = min(y_coord)

        
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
    
    plt.show()



def least_squares(x_coord, 
                  y_coord,
                  reg = False,
                  x_test = None,
                  x_label = None,
                  y_label = None,
                 ):
    x_coord = np.array(x_coord)
    y_coord = np.array(y_coord)
    
    plt.scatter(x_coord, y_coord, c='dodgerblue', zorder=3)
    x = list()
    y = list()
    for i in sample(range(len(x_coord)),2):
        x.append(x_coord[i])
        y.append(y_coord[i])
    
    x_test_max = max(x_coord)
    x_test_min = min(x_coord)
    pendiente = (y[1]-y[0])/(x[1]-x[0])
    intercepto = y[0] - pendiente*x[0]
    y_hat_max = pendiente*x_test_max + intercepto
    y_hat_min = pendiente*x_test_min + intercepto

    x_test_min = np.array([x_test_min])
    x_test_max = np.array([x_test_max])
    y_hat_min = np.array([y_hat_min])
    y_hat_max = np.array([y_hat_max])

    plt.plot(np.concatenate([x_test_min, x, x_test_max]), 
             np.concatenate([y_hat_min, y, y_hat_max]),
             c='gray', linestyle='-', zorder=2)
    
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    
    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(np.array(x_coord).reshape(-1, 1))
    
    residuals = y_coord - y_pred
    
    for x, yactual, yfitted in zip(x_coord, y_coord, y_pred): 
        plt.plot((x, x), (yactual, yfitted), '--', color='tomato')

    RSS = np.sum((y_coord - y_pred) ** 2)    
    
    plt.text(0.05, 0.95,
                 "Training Error: RSS = {0:.2f}".format(RSS),
                 ha='left', va='top', transform=plt.gca().transAxes,
                 color='gray')
    
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.grid(linestyle="dotted")
    
    plt.show()



def cost_function(X):
    return X**2

def gradient(X):
    return 2*X



def plot_gradient_descent(x_coord, y_coord, iterations=0, learning_rate=0.25, x_start=2):

    def gradient_descent(x_coord, y_coord, iterations, learning_rate, x_start): 
        plt.figure(figsize=(6, 4))
        x_path = np.empty(iterations+1,)
        x_path[0] = x_start
        plt.text(x_path[0],cost_function(x_path[0]),
                     "{}".format(1),
                     ha='left', va='top',
                     color='black', fontsize=8)
        for i in range(iterations):
            derivative = gradient(x_path[i])
            x_path[i+1] = x_path[i]-(derivative*learning_rate)   
            plt.text(x_path[i+1],cost_function(x_path[i+1]),
                     "{}".format(i+2),
                     ha='left', va='top',
                     color='black', fontsize=8)
        plt.plot(x_coord, y_coord)
        plt.plot(x_path,cost_function(x_path),'-o', ms=10)
        plt.grid(linestyle="dotted")
        plt.show()
        
    interact(
        gradient_descent,
        x_coord=widgets.fixed(x_coord),
        y_coord=widgets.fixed(y_coord),
        iterations=widgets.IntSlider(min=0, max=10, step=1, value=0, description='Step'),
        learning_rate=widgets.FloatSlider(min=0.1, max=1.5, step=0.05, value=0.25, description='Learning rate'),
        x_start=widgets.IntSlider(min=0, max=5, step=1, value=2, description='Start'),
    )



def calculate_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)



def plot_polynomial_regression(x_coord, 
                               y_coord, 
                               list_of_degrees, 
                               alpha=None, 
                               regularizer=False):  
    num_degrees = len(list_of_degrees)
    fig, axes = plt.subplots(1, num_degrees, figsize=(5*num_degrees, 4))
    if num_degrees == 1:
        axes = [axes]  
    for i, degree in enumerate(list_of_degrees):
        modelo = np.poly1d(np.polyfit(x_coord, y_coord, degree))
        y_pred = modelo(x_coord)
        recta = np.linspace(1, 22, 100)
        axes[i].scatter(x_coord, y_coord, c='dodgerblue')
        axes[i].plot(recta, modelo(recta), c="gray")
        axes[i].text(0.05, 0.95,
                     "{}".format(modelo),
                     ha='left', va='top', transform=axes[i].transAxes,
                     color='gray', fontsize=8)
        axes[i].text(0.95, 0.05,
                     "Training Error: RSS = {0:.2f}".format(calculate_rss(y_coord, y_pred)),
                     ha='right', va='bottom', transform=axes[i].transAxes,
                     color='gray', fontsize=8)
        axes[i].set_title(f'Degree {degree}')
        
        if regularizer:            
            alpha = 0.0 if alpha is None else alpha
            x, x_line = np.expand_dims(x_coord,1), np.expand_dims(recta,1)    
            gen_poly_terms = PolynomialFeatures(degree=degree)
            X_train_with_poly = gen_poly_terms.fit_transform(x)
            if regularizer == "ridge":
                poly_regression = Ridge(alpha=alpha)
            elif regularizer == "lasso":
                poly_regression = Lasso(alpha=alpha)
            else:
                break
            poly_regression.fit(X_train_with_poly, y_coord)
            poly_vals=gen_poly_terms.transform(x_line)
            y_pred=poly_regression.predict(poly_vals)        
            axes[i].plot(x_line,
                         y_pred, 
                         color='black',
                         linestyle='--',
                         )
    plt.tight_layout()
    plt.show()

def plot_faces(data, targets, n_person=3, n_faces=10):
    if n_person > 40:
        n_person = 40
    if n_faces > 10:
        n_faces = 10
    if n_person == 1 and n_faces == 1:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(data[0].reshape(64, 64), cmap='binary_r')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(targets[0], fontsize=20)
    elif n_person == 1:
        fig, ax = plt.subplots(1, n_faces, figsize=(2*n_faces, 2),
                               subplot_kw={'xticks': [], 'yticks': []},
                               gridspec_kw=dict(wspace=0.1))
        for i in range(min(n_faces, len(data))):
            ax[i].imshow(data[i].reshape(64, 64), cmap='binary_r')
        ax[0].set_ylabel(targets[0], fontsize=20)
    elif n_faces == 1:
        fig, ax = plt.subplots(n_person, 1, figsize=(4, 2*n_person),
                               subplot_kw={'xticks': [], 'yticks': []},
                               gridspec_kw=dict(hspace=0.1))
        i = 0
        k = 0
        while k < n_person:
            ax[k].imshow(data[i].reshape(64, 64), cmap='binary_r')
            ax[k].set_ylabel(targets[i], fontsize=20)
            k += 1
            i += 10
    else:
        fig, ax = plt.subplots(n_person, n_faces, figsize=(2*n_faces, 2*n_person),
                               subplot_kw={'xticks': [], 'yticks': []},
                               gridspec_kw=dict(hspace=0.1, wspace=0.1))
        col = 0
        row = 0
        i = 0
        k = 0
        while k < n_faces * n_person:
            ax[row, col].imshow(data[i].reshape(64, 64), cmap='binary_r')
            k += 1
            col += 1
            if col % n_faces == 0:
                if col < 10:
                    i = i + (10 - col)
                ax[row, 0].set_ylabel(targets[i], fontsize=20)
                row += 1
                col = 0
            i += 1
    
    plt.show()
    
def plot_test_faces(X, X_, y_, n_faces = 5):
    if n_faces > len(X):
        n_faces = len(X)
        
    sub_set_faces = sample(range(0,len(X)), n_faces)
    
    fig, ax = plt.subplots(3, n_faces, figsize=(2*n_faces, 2*3),
                           subplot_kw={'xticks':[], 'yticks':[]},
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i,indx in enumerate(sub_set_faces):
        ax[0, i].imshow(X[indx].reshape(64, 64), cmap='binary_r')        
        ax[1, i].imshow(X_[indx].reshape(32, 64), cmap='binary_r')
        ax[2, i].imshow(y_[indx].reshape(32, 64), cmap='binary_r')

    ax[0, 0].set_ylabel('face')
    ax[1, 0].set_ylabel('upper')
    ax[2, 0].set_ylabel('lower')
        
    plt.show()
    
def plot_predicted_faces(X_test, y_test, lr_predict, ridge_predict, n_faces = 5):
    if n_faces > len(y_test):
        n_faces = len(y_test)
        
    sub_set_faces = sample(range(0,len(y_test)), n_faces)
    
    fig, ax = plt.subplots(3, n_faces, figsize=(2*n_faces, 2*3),
                           subplot_kw={'xticks':[], 'yticks':[]},
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i,indx in enumerate(sub_set_faces):
        ax[0, i].imshow(np.hstack((X_test[indx], y_test[indx])).reshape(64, 64), cmap='binary_r')        
        ax[1, i].imshow(np.hstack((X_test[indx], lr_predict[indx])).reshape(64, 64), cmap='binary_r')
        ax[2, i].imshow(np.hstack((X_test[indx], ridge_predict[indx])).reshape(64, 64), cmap='binary_r')

    ax[0, 0].set_ylabel('Cara Real')
    ax[1, 0].set_ylabel('Linear Regression')
    ax[2, 0].set_ylabel('Ridge regression')
        
    plt.show()