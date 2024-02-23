#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def ReLU(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)

def derivada(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

def plot_activation_functions(derivatives=False):
    z = np.linspace(-5, 5, 200)
    
    fig, ax = plt.subplots(1, 4, figsize=(15, 3))
    
    for i,fun in enumerate(["Sigmoid","Tanh","ReLU","Leaky ReLU"]):
        ax[i].set_title(fun)
        ax[i].grid(True)
        ax[i].axis([-5, 5, -1.2, 1.2])
        
    ax[0].plot(z, sigmoide(z), "b", ls="-", lw=2)
    ax[1].plot(z, np.tanh(z), "g", ls='-', lw=2)
    ax[2].plot(z, ReLU(z), "r", ls="-", lw=2)
    ax[3].plot(z, leaky_relu(z), "orange", ls="-", lw=2)
    if derivatives:
        ax[0].plot(z, derivada(sigmoide,z), "lightblue", ls="dashed", lw=2, alpha=1)
        ax[1].plot(z, derivada(np.tanh,z), "lightgreen", ls='dashed', lw=2)
        ax[2].plot(z, derivada(ReLU,z), "lightpink", ls="dashed", lw=2)
        ax[3].plot(z, derivada(leaky_relu,z), "gold", ls="dashed", lw=2)
    
    plt.show()


import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD 
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from keras.utils.np_utils import to_categorical


def plot_results(X, y, model, history):
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    #plt.figure(figsize=(15,3))
    #plt.subplot(131)
    
    ax[0].plot(history.history['accuracy'], c='C2')
    ax[0].plot(history.history['loss'], c='C1')
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['accuracy', 'loss'], loc='center right')

    
    plt.subplot(132)
    
    if len(set(y)) == 2:
        amin, bmin = X.min(axis=0) - 0.1
        amax, bmax = X.max(axis=0) + 0.1
        hticks = np.linspace(amin, amax, 101)
        vticks = np.linspace(bmin, bmax, 101)

        aa, bb = np.meshgrid(hticks, vticks)
        ab = np.c_[aa.ravel(), bb.ravel()]
        c = model.predict(ab, verbose=0)
        cc = c.reshape(aa.shape)

        cm = plt.cm.coolwarm
        cm_bright = ListedColormap(['dodgerblue', 'tomato'])

        contour = plt.contourf(aa, bb, cc, cmap=cm, alpha=0.6)

        ax_c = fig.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, 0.25, 0.5, 0.75, 1])

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, ec="k")
        plt.title(f"Accuracy: {round(accuracy_score(y, np.round(model.predict(X, verbose=0)).astype(int)),2)}")

        plt.xlim(amin, amax)
        plt.ylim(bmin, bmax)
        
        y_pred = np.round(model.predict(X, verbose=0)).astype(int)

    else:
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, ec="k")
        plt.title(f"Accuracy: {round(accuracy_score(y, np.argmax(model.predict(X, verbose=0), axis=1)),2)}")
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
 
    
    plt.subplot(133)
    

    conf_mat = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(cmap="Purples", ax=ax[2])
    plt.show()