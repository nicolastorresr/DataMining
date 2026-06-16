#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from shapely.geometry import Point

def generative(data='blobs',
               n_samples=100, 
               centers=2,
               random_state=2, 
               cluster_std=1.50,
               distribution=False,
               new_data=False,
               new_samples=2000,
               color_1='dodgerblue',
               color_2='tomato', 
               c_map_1 = 'Blues',
               c_map_2 = 'Reds',
               marker_1 = 'o',
               marker_2 = 's',
               edgecolor_1 = 'k',
               edgecolor_2 = 'k',
               marker_size=30, 
               Xnew=None,
               ):
    
    if data == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=centers,
                      random_state=random_state, cluster_std=cluster_std)
    else:
        try:
            X,y = data
            X = np.array(X)
            y = np.array(y)
        except:
            print("data is not valid.")
            return
        
    n_col = 2 if distribution and Xnew is None else 1
    
    fig, axs = plt.subplots(1, n_col, figsize=(6*n_col, 5))
    
    ax = axs if n_col == 1 else axs[0]
        
        
    ax.scatter(X[y==0][:, 0], X[y==0][:, 1], edgecolor=edgecolor_1,
            marker=marker_1, color=color_1, s=marker_size)
    ax.scatter(X[y==1][:, 0], X[y==1][:, 1], edgecolor=edgecolor_2,
            marker=marker_2, color=color_2, s=marker_size)  

    max_x = np.max(X[:, 0])
    max_y = np.max(X[:, 1])
    min_x = np.min(X[:, 0])
    min_y = np.min(X[:, 1])
    
    ax.set_xlim((min_x,max_x))
    ax.set_ylim((min_y,max_y))

    # Set the original axis limits
    original_xlim = ax.set_xlim()
    original_ylim = ax.set_ylim()

    # Define margins
    x_margin = (max_x-min_x)/10
    y_margin = (max_y-min_y)/10

    # Add margins to the axis limits
    new_xlim = (original_xlim[0] - x_margin, original_xlim[1] + x_margin)
    new_ylim = (original_ylim[0] - y_margin, original_ylim[1] + y_margin)

    # Set the updated axis limits
    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)
    
    if Xnew:
        model = GaussianNB()
        model.fit(X, y)
                    
        Xnew = np.array(Xnew)

        if Xnew.ndim < 2:
            Xnew = Xnew.reshape(1, -1)
        yprob = model.predict_proba(Xnew)

        max_x = max(np.concatenate([Xnew[:,0],  X[:, 0]]))
        max_y = max(np.concatenate([Xnew[:,1],  X[:, 1]]))
        min_x = min(np.concatenate([Xnew[:,0],  X[:, 0]]))
        min_y = min(np.concatenate([Xnew[:,1],  X[:, 1]]))

        ax.set_xlim((min_x,max_x))
        ax.set_ylim((min_y,max_y))

        # Set the original axis limits
        original_xlim = ax.set_xlim()
        original_ylim = ax.set_ylim()

        # Define margins
        x_margin = (max_x-min_x)/10
        y_margin = (max_y-min_y)/10

        # Add margins to the axis limits
        new_xlim = (original_xlim[0] - x_margin, original_xlim[1] + x_margin)
        new_ylim = (original_ylim[0] - y_margin, original_ylim[1] + y_margin)

        # Set the updated axis limits
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

        ynew = model.predict(Xnew)

        ax.scatter(Xnew[ynew==0][:, 0], Xnew[ynew==0][:, 1], edgecolor=edgecolor_1,
                marker='*', color=color_1, s=100, alpha=0.9)
        ax.scatter(Xnew[ynew==1][:, 0], Xnew[ynew==1][:, 1], edgecolor=edgecolor_2,
                marker='*', color=color_2, s=100, alpha=0.9)


    if distribution:
        
        xg = np.linspace(new_xlim[0], new_xlim[1], 60)
        yg = np.linspace(new_ylim[0], new_ylim[1], 40)
        xx, yy = np.meshgrid(xg, yg)
        Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T
        
        
        
        for label, (color, c_map) in enumerate(zip([color_1, color_2],[c_map_1, c_map_2])):
            mask = (y == label)
            mu, std = X[mask].mean(0), X[mask].std(0)
            P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
            Pm = np.ma.masked_array(P, P < 0.03)
            ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                          cmap=c_map)
            ax.contour(xx, yy, P.reshape(xx.shape),
                       levels=[0.01, 0.1, 0.5, 0.9],
                       colors=color, alpha=0.2)
            ax.set_title("Naive Bayes Model")
            
            if Xnew is not None:
                
                if yprob.shape[0] == 1:
                    output = "{0:.2f}".format(yprob[0][label])
                
            else:
                output = "µ = ({0:.2f},{1:.2f}) σ = ({2:.2f},{3:.2f})".format(mu[0],mu[1],std[0],std[1])
                gaussian = (1 / (std[0] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xg - mu[0]) / std[0]) ** 2)
                axs[1].plot(xg, gaussian, color=color, label=f'Class {label} Gaussian')
                axs[1].grid(linestyle="dotted")
                axs[1].set_title('Gaussian Distributions')
            
            if label == 0:
                ax.text(0.98, 0.02,
                output,
                ha='right', va='bottom', transform=ax.transAxes,
                color=color)
            else:
                ax.text(0.02, 0.98,
                output,
                ha='left', va='top', transform=ax.transAxes,
                color=color)
        
    if new_data:
        
        model = GaussianNB()
        model.fit(X, y)
        rng = np.random.RandomState(0)
        
        
        Xnew = [new_xlim[0], new_ylim[0]] + [new_ylim[1]+10, new_xlim[1]+10] * rng.rand(new_samples, 2)

        ynew = model.predict(Xnew)

        ax.scatter(Xnew[ynew==0][:, 0], Xnew[ynew==0][:, 1], edgecolor=edgecolor_1,
                marker=marker_1, color=color_1, s=marker_size, alpha=0.1)
        ax.scatter(Xnew[ynew==1][:, 0], Xnew[ynew==1][:, 1], edgecolor=edgecolor_2,
                marker=marker_2, color=color_2, s=marker_size, alpha=0.1) 
        
    ax.grid(linestyle="dotted")
    plt.show()


def generate_land_points(num_points, world, seed=42):
    # Set seed for random number generation
    np.random.seed(seed)

    # Generate random latitude and longitude coordinates
    lon = np.random.uniform(low=-180, high=180, size=num_points)
    lat = np.random.uniform(low=-90, high=90, size=num_points)

    # Create GeoDataFrame for points
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat))

    # Set CRS of points GeoDataFrame
    points.crs = world.crs  # Set CRS to match world GeoDataFrame

    # Perform spatial join with world GeoDataFrame
    points = gpd.sjoin(points, world, how="inner", predicate='intersects')

    # If not enough points found on land, generate additional points
    while len(points) < num_points:
        additional_lon = np.random.uniform(low=-180, high=180)
        additional_lat = np.random.uniform(low=-90, high=90)
        additional_point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([additional_lon], [additional_lat]))
        additional_point.crs = world.crs  # Set CRS to match world GeoDataFrame
        additional_point = gpd.sjoin(additional_point, world, how="inner", predicate='intersects')
        points = pd.concat([points, additional_point], ignore_index=True)

    return points

def generate_points(num_points, world, seed=42):
    # Set seed for random number generation
    np.random.seed(seed)
    
    # Generate random latitude and longitude coordinates
    lon = np.random.uniform(low=-180, high=180, size=num_points)
    lat = np.random.uniform(low=-90, high=90, size=num_points)
    
    # Create GeoDataFrame for points
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat))

    # Set CRS of points GeoDataFrame
    points.crs = world.crs  # Set CRS to match world GeoDataFrame

    # Perform spatial join with world GeoDataFrame
    points = gpd.sjoin(points, world, how="left", predicate='intersects')

    # Assign color to points based on land or water
    points['color'] = 'dodgerblue'
    points.loc[points.index_right.isna(), 'color'] = 'tomato'
    
    return points

def plot_single_point(p_lat, p_lon, ax, world, label=None):
    # Create a GeoDataFrame for the single point
    point_df = gpd.GeoDataFrame(geometry=[Point(p_lon, p_lat)])

    # Set CRS of point GeoDataFrame to match world GeoDataFrame
    point_df.crs = world.crs  

    # Check if the point falls on land
    point_land = gpd.sjoin(point_df, world, how="inner", predicate='intersects')

    # Plot the point
    if len(point_land) > 0:
        ax.scatter(p_lon, p_lat, c='dodgerblue', marker='o', s=50, edgecolor='k', zorder=10) 
    else:
        ax.scatter(p_lon, p_lat, c='tomato', marker='o', s=50, edgecolor='k', zorder=10) 
    
    # Add text above the point if label is provided
    if label:
        ax.text(p_lon, p_lat, label, verticalalignment='bottom', horizontalalignment='left', 
               fontsize=8, fontweight='normal')

def first_generative_model(num_points, new_points=None, points_labels=None,
                           true_distribution=False,sample_space=False, 
                           seed=42, marker_size=10):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(15, 5))
    
    world = gpd.read_file("world_map.shp")
    
    if new_points:
        
        # If a single point is passed as tuple
        if isinstance(new_points, tuple):
            p_lat, p_lon = new_points
            if points_labels:
                label = points_labels[i]
            else:
                label = chr(65)
            plot_single_point(p_lat, p_lon, ax, world, label)

        # If a list of points is passed
        elif isinstance(new_points, list):
        
            # Iterate over points
            for i, point in enumerate(new_points):
                if isinstance(point, tuple) and len(point) == 2:
                    p_lat, p_lon = point
                    if points_labels:
                        label = points_labels[i]
                    else:
                        label = chr(65 + i)
                    plot_single_point(p_lat, p_lon, ax, world, label)
                else:
                    print(f"Illegal point format at index {i}: {point}")
                    continue
            

    if true_distribution:
        world.plot(ax=ax, color='lightgray')
        
        if sample_space:
            plt.text(90, -75,
                'Belongs to the true data-\ngenerating distribution',
                ha='left', va='top', fontsize=8,
                color='dodgerblue')
            plt.text(-180, 15,
                'Does not belong to the true\ndata-generating distribution',
                ha='left', va='top', fontsize=8,
                color='tomato')
        else:
            plt.text(-80, -25,
                    'The true distribution',
                    ha='right', va='top',
                    color='gray')

    if sample_space:
        # Generate sample points on everywhere
        sample_points = generate_points(num_points, world, seed=seed)
        points_color = sample_points['color']
        plt.title(f'{num_points} points in the sample space')
    else:
        # Generate sample points on land only
        sample_points = generate_land_points(num_points, world, seed=seed)
        points_color = 'k'
        # Set title
        plt.title(f'{num_points} points generated by an unknown distribution')

    # Plot sample points
    sample_points.plot(ax=ax, color=points_color, markersize=marker_size)

    

    # Show the plot
    plt.show()
    
    
def calcular_probabilidad_covid_sintomas(prob_covid, prob_sintomas_dado_covid, prob_sintomas_sin_covid):
    # Calcula la probabilidad de no tener COVID-19
    prob_no_covid = 1 - prob_covid
    
    # Calcula la probabilidad marginal de presentar síntomas
    prob_sintomas = (prob_sintomas_dado_covid * prob_covid) + (prob_sintomas_sin_covid * prob_no_covid)
    
    # Calcula la probabilidad de tener COVID-19 dado que presenta síntomas usando el teorema de Bayes
    prob_covid_dado_sintomas = (prob_sintomas_dado_covid * prob_covid) / prob_sintomas
    
    return round(prob_covid_dado_sintomas,3)



import ipywidgets as widgets
from IPython.display import display

def calcular_probabilidad_covid_sintomas(prob_covid, prob_sintomas_dado_covid, prob_sintomas_sin_covid):
    # If both prob_sintomas_dado_covid and prob_sintomas_sin_covid are zero, return zero probability
    if prob_sintomas_dado_covid == 0 and prob_sintomas_sin_covid == 0:
        return 0
    
    # Calcula la probabilidad de no tener COVID-19
    prob_no_covid = 1 - prob_covid
    
    # Calcula la probabilidad marginal de presentar síntomas
    prob_sintomas = (prob_sintomas_dado_covid * prob_covid) + (prob_sintomas_sin_covid * prob_no_covid)
    
    # If prob_sintomas is zero, return zero probability
    if prob_sintomas == 0:
        return 0
    
    # Calcula la probabilidad de tener COVID-19 dado que presenta síntomas usando el teorema de Bayes
    prob_covid_dado_sintomas = (prob_sintomas_dado_covid * prob_covid) / prob_sintomas
    
    return prob_covid_dado_sintomas

# Función para actualizar el gráfico cuando cambian los valores de los sliders
def update_plot(prob_covid, prob_sintomas_dado_covid, prob_sintomas_sin_covid):
    probabilidad_covid_sintomas = calcular_probabilidad_covid_sintomas(prob_covid, 
                                                                       prob_sintomas_dado_covid, 
                                                                       prob_sintomas_sin_covid)
    
    labels = ['Probabilidad de tener COVID-19', 'Probabilidad de no tener COVID-19']
    probabilidades = [probabilidad_covid_sintomas, 1 - probabilidad_covid_sintomas]

    plt.bar(labels, probabilidades, color=['tomato', 'dodgerblue'], edgecolor='k')
    plt.title('Probabilidad de tener COVID-19 dado que presenta síntomas')
    plt.grid(linestyle="dotted")
    plt.ylabel('Probabilidad')
    plt.ylim(0, 1)
    plt.show()

# Crea sliders interactivos para los parámetros
prob_covid_slider = widgets.FloatSlider(value=0.05, min=0, max=1, step=0.01, 
                                        description='Prob. COVID-19:')
prob_covid_slider.style.description_width = 'initial' 
prob_covid_slider.layout.width = '400px' 

prob_sintomas_dado_covid_slider = widgets.FloatSlider(value=0.8, min=0, max=1, step=0.01, 
                                                      description='Prob. síntomas | COVID-19:')
prob_sintomas_dado_covid_slider.style.description_width = 'initial'  
prob_sintomas_dado_covid_slider.layout.width = '400px'  

prob_sintomas_sin_covid_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01, 
                                                     description='Prob. síntomas | No COVID-19:')
prob_sintomas_sin_covid_slider.style.description_width = 'initial' 
prob_sintomas_sin_covid_slider.layout.width = '400px'  


# Crea un widget interactivo con los sliders y la función de actualización
interactive_plot = widgets.interactive(update_plot,
                                       prob_covid=prob_covid_slider,
                                       prob_sintomas_dado_covid=prob_sintomas_dado_covid_slider,
                                       prob_sintomas_sin_covid=prob_sintomas_sin_covid_slider)




def deaggregate_statistics(dataframe):
    final_gender = []
    final_dept = []
    final_admit = []

    for _, row_serie in dataframe.iterrows():
        for _ in range(1, row_serie[3] + 1):
            final_admit.append(row_serie[0])
            final_gender.append(row_serie[1])
            final_dept.append(row_serie[2])

    return pd.DataFrame({
        'Admit': final_admit,
        'Gender': final_gender,
        'Dept': final_dept
    })



import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def plot_statistics(df_deagg):
    plt.figure(figsize=(15, 3))

    for i, (n, v) in enumerate(df_deagg.items()):
        plt.subplot(1, 3, i + 1)
        value_counts_percent = df_deagg[n].value_counts(normalize=True) * 100  
        order = value_counts_percent.index.tolist()
        sns.countplot(y=df_deagg[n], order=order)
        plt.title('{}'.format(n))
        plt.tight_layout()

        ax = plt.gca()
        total = len(df_deagg[n])
        for rect in ax.patches:
            percentage = rect.get_width() / total * 100
            ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, f'{percentage:.1f}%', ha='left', va='center')

    plt.show()


def plot_roc_auc(classifier, X_test_mat, y_test_vec):
    plt.figure(figsize=(5, 3))
    y_hat_pr = classifier.predict_proba(X_test_mat)[:,1]
    fpr, tpr, _ = roc_curve(y_test_vec, y_hat_pr)
    auc = round(roc_auc_score(y_test_vec, y_hat_pr), 3)
    plt.plot(fpr, tpr, label="AUC = {}".format(auc), color='tomato', lw=4)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.axis([0, 1, 0, 1])
    plt.plot([0, 1], [0, 1], color='slategrey')
    plt.legend()
    plt.show()


def compare_priors(X_train, X_test, y_train, y_test, priors_list, fontsize=12, row_height=2):
    headers = ["A priori", "Accuracy", "Recall", "Precision", "F1", "AUC"]
    table_data = []
    
    for prior in priors_list:
        tmp_clf = GaussianNB(priors=prior)
        tmp_clf.fit(X_train, y_train)
        tmp_class = tmp_clf.predict(X_test)
        tmp_pr = tmp_clf.predict_proba(X_test)[:, 1]
        tmp_acc = accuracy_score(y_test, tmp_class).round(3)
        tmp_rec = recall_score(y_test, tmp_class).round(3)
        tmp_prec = precision_score(y_test, tmp_class).round(3)
        tmp_f1 = f1_score(y_test, tmp_class).round(3)
        tmp_auc = roc_auc_score(y_test, tmp_pr).round(3)
        table_data.append([prior, tmp_acc, tmp_rec, tmp_prec, tmp_f1, tmp_auc])
    
    fig, ax = plt.subplots(figsize=(15,1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', fontsize=fontsize, cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, row_height)  
    
    plt.show()

