#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay


# In[3]:


def plot_decision_tree(dfIris, depth=2, font_size=12, seed=42):
    depth = max(depth, 1)
    codes = {'setosa':'orange', 'versicolor':'mediumspringgreen', 'virginica':'darkviolet'}    
    X = dfIris[['petal_width','petal_length']]
    y = dfIris.species
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(X, y)
    disp = DecisionBoundaryDisplay.from_estimator(tree_clf, X, cmap=plt.cm.viridis_r, 
                                                  response_method="predict",alpha=0.5, ax=ax1)
    disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y.map(codes), edgecolor="k", alpha=1)
    if depth > 3:
        font_size = 8
    tree.plot_tree(tree_clf, feature_names=["petal_width","petal_length"], 
                   filled = True, impurity=False, fontsize=font_size, ax=ax2)
    ax1.text(0.02, 0.98,
            "Accuracy: {0:.2f}".format(accuracy_score(y, tree_clf.predict(X))),
            ha='left', va='top', transform=ax1.transAxes,
            color='k')
    plt.show()


# In[25]:


def plot_cor(bag_clf, X_test):
    X_test = np.array(X_test)
    coef = []
    for i in range(len(bag_clf.estimators_)):
        for j in range(len(bag_clf.estimators_)):
            coef.append(stats.pearsonr(bag_clf.estimators_[i].predict(X_test), 
                                       bag_clf.estimators_[j].predict(X_test))[0])
    coef = np.array(coef).reshape(len(bag_clf.estimators_), len(bag_clf.estimators_))
    sns.heatmap(coef, cmap='coolwarm', annot=True, vmin=-1, vmax=1)
    plt.title("Correlación bivariada entre árboles")
    plt.show()


# In[29]:


def plot_feature_importance(x, height):
    plt.figure(figsize=(10, 4))
    plt.bar(x, height, edgecolor='k')
    plt.grid(linestyle=':')
    plt.title("Feature Importance")
    plt.show()

