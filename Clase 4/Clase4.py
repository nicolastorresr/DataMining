#!/usr/bin/env python
# coding: utf-8

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from PIL import Image
from wordcloud import WordCloud
from ChernoffFace import *
import matplotlib.cm
from scipy.stats import gaussian_kde
from skimage import io
import plotly.express as px

def titanic_countplot(titanic_df):
    # Set up the subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(6, 5))

    # Define the classes and survival labels
    classes = ['1st', '2nd', '3rd']
    survival_labels = ['Died', 'Survived']

    # Iterate over rows (class)
    for i, row in enumerate(axes):
        # Iterate over columns (survival)
        for j, ax in enumerate(row):
            # Filter data for the current class and survival status
            class_data = titanic_df[titanic_df['Pclass'] == i + 1]
            survival_data = class_data[class_data['Survived'] == j]

            # Create a barplot for gender count
            sns.countplot(x='Sex', data=survival_data, 
                          palette={'female': 'C1', 'male': 'C0'}, ax=ax, alpha=0.9)

            # Set labels and title
            ax.set_xlabel('')
            ax.set_ylabel('')

            if i == 0:
                ax.set_title(survival_labels[j])
            
            ax.yaxis.grid(True, linestyle='-')

            # Remove y-axis numbers from the second column
            if j == 1:
                ax.set_yticklabels([])

    # Add x-label at the right of each row with class
    for i, ax_row in enumerate(axes):
        ax_row[1].text(2.1, 0.5, f'{classes[i]}', va='center', ha='left', 
                       rotation=-90, transform=ax_row[0].transAxes)

    # Adjust layout
    plt.tight_layout()
    fig.text(0, 0.5, 'Count', va='center', rotation='vertical')

    # Show the plot
    plt.show()


def titanic_hist(titanic_df):
    # Create subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Define bin widths
    bin_widths = [1, 3, 5, 15]

    # Plot histograms with different bin widths
    for i, bin_width in enumerate(bin_widths):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # Plot histogram
        ax.hist(titanic_df['Age'].dropna(), 
                bins = int((titanic_df['Age'].max() - titanic_df['Age'].min()) / bin_width),
                color='skyblue', edgecolor='black')

        # Set titles and labels
        ax.set_title(f'Bin Width: {bin_width} years')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Enable the x-axis grid
        ax.grid(axis='y', linestyle='-', which='both')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def iris_color_kde(iris):
    # Create a figure and axis
    plt.figure(figsize=(7, 4))

    # Plot the density distributions with different line styles and fill
    setosa_kde = sns.kdeplot(data=iris[iris['species'] == 'setosa']['sepal_length'], 
                             fill=True, linestyle='solid', color='C0', linewidth=2, alpha=0.4)
    versicolor_kde = sns.kdeplot(data=iris[iris['species'] == 'versicolor']['sepal_length'], 
                                 fill=True, linestyle='solid', color='C1', linewidth=2, alpha=0.4)
    virginica_kde = sns.kdeplot(data=iris[iris['species'] == 'virginica']['sepal_length'], 
                                fill=True, linestyle='solid', color='C2', linewidth=2, alpha=0.4)

    # Set labels and title
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Density')

    # Plot the density distributions with different line styles
    styles = ['solid', 'solid', 'solid']
    colors = ['C0', 'C1', 'C2']
    labels = ['setosa', 'versicolor', 'virginica']

    for label, style, color in zip(labels, styles, colors):
        subset = iris[iris['species'] == label]['sepal_length']
        kde = gaussian_kde(subset)
        x_vals = sorted(subset)
        y_vals = kde(x_vals)
        max_density = max(y_vals)    
        plt.text(subset.mean(), max_density, label, color=color, fontsize=12, ha='center', va='bottom')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the plot
    plt.show()


def fuel_color_scatter(fuel_econ):
    # Create a figure and axis
    plt.figure(figsize=(6, 5))

    # Define colors for each origin category
    colors = {'usa': 'skyblue', 'japan': 'red', 'europe': 'green'}

    # Iterate over each unique 'origin' category and plot with different markers and colors
    markers = ['s', 'o', '^']
    for origin, marker in zip(fuel_econ['origin'].unique(), markers):
        subset = fuel_econ[fuel_econ['origin'] == origin]
        plt.scatter(
            subset['displacement'],
            subset['mpg'],
            marker=marker,
            s=50,  # Adjust the size as needed
            #facecolors='none',  # Transparent fill
            edgecolors=colors[origin],  # Set the edge color to black
            label=origin,
            alpha=0.5,
            color=colors[origin],  # Set the color based on the 'origin'
        )

    # Set labels and title
    plt.xlabel('Engine Displacement (L)')
    plt.ylabel('City Fuel Economy (mpg)')
    plt.title('City Fuel Economy vs Engine Displacement by Drive Type')
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the plot
    plt.show()


def california_scatter(cities):
    # Extract the data we're interested in
    lat, lon = cities['latd'], cities['longd']
    population, area = cities['population_total'], cities['area_total_km2']

    # Scatter the points, using size and color but no label
    plt.scatter(lon, lat, label=None,
               c=np.log10(population), cmap='viridis',
               s=area, linewidth=0, alpha=0.5)
    plt.axis()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.colorbar(label='log$_{10}$(population)')
    plt.clim(3, 7)

    # Here we create a legend:
    # we'll plot empty lists with the desired size and label
    for area in [100, 300, 500]:
        plt.scatter([], [], c='k', alpha=0.3, s=area,
                   label=str(area) + ' km$^2$')
    plt.legend(scatterpoints=1, frameon=False,
              labelspacing=1, title='City Area')

    plt.title('California Cities: Area and Population')
    plt.show()


def temperatures_errorbar(temperature_df):
    # Set the size of the figure
    plt.figure(figsize=(7, 4))

    # Plot points and error bars using plt.errorbar
    plt.errorbar(x=temperature_df['Month'], y=temperature_df['Temperature'],color='black',
                 yerr=temperature_df['Std_Dev_Temperature'], fmt='o', 
                 capsize=5, label='Mean Temperature with Error Bars')

    # Set plot labels and title
    plt.xlabel('Month')
    plt.ylabel('Mean Temperature (°F)')
    #plt.title('Mean Daily Temperatures in Lincoln, Nebraska (2016) with Error Bars')
    plt.grid(False)
    # Display legend
    #plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Show the plot
    plt.show()


def explain_boxplot():
    # Generate example data
    np.random.seed(42)
    data = np.random.normal(size=100)

    # Set up the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot cloud of points on the left subplot
    sns.stripplot(y=data, ax=axes[0], color='black', jitter=True, alpha=1)
    axes[0].set_title('Cloud of Points')

    # Plot boxplot on the right subplot
    sns.boxplot(y=data, ax=axes[1], color='lightgray')
    axes[1].set_title('Boxplot')

    # Calculate boxplot details for annotation
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    lower_fence = q1 - 1.5 * iqr

    # Add annotations to the boxplot
    axes[1].annotate('Median', xy=(0.5, np.median(data)), xytext=(0.5, np.median(data) ),
                     arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', lw=1), ha='center')
    axes[1].annotate('Q1', xy=(0.5, q1), xytext=(0.5, q1 ),
                     arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', lw=1), ha='center')
    axes[1].annotate('Q3', xy=(0.5, q3), xytext=(0.5, q3 ),
                     arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', lw=1), ha='center')
    axes[1].annotate('Upper Fence', xy=(0.5, upper_fence), xytext=(0.5, upper_fence-0.1),
                     arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', lw=1), ha='center')
    axes[1].annotate('Lower Fence', xy=(0.5, lower_fence), xytext=(0.5, lower_fence+0.1),
                     arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', lw=1), ha='center')

    # Show the plot
    plt.tight_layout()
    plt.show()

def temperatures_boxplot(temperature_df):
    # Set the size of the figure
    plt.figure(figsize=(7, 4))

    # Plot boxplots using sns.boxplot
    sns.boxplot(x=temperature_df['Month'], y=temperature_df['Temperature'], color='lightgray')

    # Set plot labels and title
    plt.xlabel('Month')
    plt.ylabel('Temperature (°F)')
    plt.title('Distribution of Daily Temperatures in Lincoln, Nebraska (2016) with Boxplots')

    # Remove the grid
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show the plot
    plt.show()

def temperatures_ridgeline(temperature_df, months):
    # Save the current theme settings
    current_theme = sns.axes_style()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(temperature_df, row="Month",aspect=5, height=0.6)
    g.map_dataframe(sns.kdeplot, "Temperature", fill=True, alpha=1)
    g.map_dataframe(sns.kdeplot, "Temperature", color='black')
    g.fig.subplots_adjust(hspace=-.1)
    g.set_axis_labels('Temperature (°F)', '')
    g.fig.suptitle('Ridgeline Plot of Mean Daily Temperatures in Lincoln, Nebraska (2016) by Month', y=1.02)
    g.set_titles("")
    g.set_ylabels(None)
    g.set(yticks=[])
    g.despine(left=True)
    for (month, ax) in zip(months, g.axes.flat):
        ax.set_ylabel(month[:3])
    g.fig.set_size_inches(5, 5)
    # Restore the original theme settings after the function execution
    sns.set_theme(style=current_theme)

def explain_violin():
    # Generate example data
    np.random.seed(42)
    data = np.random.normal(size=100)

    # Set up the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot cloud of points on the left subplot
    sns.stripplot(y=data, ax=axes[0], color='black', jitter=True, alpha=1)
    axes[0].set_title('Cloud of Points')

    # Plot violin plot on the right subplot
    sns.violinplot(y=data, ax=axes[1], color='lightgray')
    axes[1].set_title('Violin Plot')

    # Calculate violin plot details for annotation
    median = np.median(data)
    upper_fence = max(data)
    lower_fence = min(data)

    # Add annotations to the violin plot
    axes[1].annotate('Median', xy=(0.5, median), xytext=(0.5, median),
                     arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', lw=1), ha='center')

    axes[1].annotate('Upper Fence', xy=(0.5, upper_fence), xytext=(0.5, upper_fence),
                     arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', lw=1), ha='center')
    axes[1].annotate('Lower Fence', xy=(0.5, lower_fence), xytext=(0.5, lower_fence),
                     arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.7', lw=1), ha='center')

    # Show the plot
    plt.tight_layout()
    plt.show()

def temperatures_violin(temperature_df):
    # Set the size of the figure
    plt.figure(figsize=(7, 4))

    # Plot violin plots using sns.violinplot
    sns.violinplot(x=temperature_df['Month'], y=temperature_df['Temperature'], color='lightgray')

    # Set plot labels and title
    plt.xlabel('Month')
    plt.ylabel('Temperature (°F)')
    plt.title('Distribution of Daily Temperatures in Lincoln, Nebraska (2016) with Violin Plots')

    # Remove the grid
    plt.grid(False)

    # Show the plot
    plt.show()

from sklearn.datasets import load_digits
from sklearn.datasets import fetch_olivetti_faces

from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize

def plot_corr(data, marker='s', figsize=None, show_grid=False, cmap='RdBu'):  
    
    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    
    fig, ax = plt.subplots(1, 2, figsize=figsize, subplot_kw={'aspect':'equal'})
    
    # Create a scatter plot of correlations as colored circles
    heatmap = sns.scatterplot(
        x=data.columns.repeat(len(data.columns)),
        y=np.tile(data.columns, len(data.columns)),
        hue=data.values.flatten(),
        hue_norm=(-1,1),
        size=np.abs(data.values.flatten()),
        sizes=(500, 2000),
        palette=cmap,
        marker=marker,
        ax=ax[0],
        legend=False,
    )
    # Optional: If you want to remove the colorbar
    if show_grid:
        heatmap.grid(True, color='lightgray')
    
    # Set the original axis limits
    original_xlim = ax[0].set_xlim()
    original_ylim = ax[0].set_ylim()

    # Define margins
    margin = 0.2

    # Add margins to the axis limits
    new_xlim = (original_xlim[0] - margin, original_xlim[1] + margin)
    new_ylim = (original_ylim[0] + margin, original_ylim[1] - margin)

    # Set the updated axis limits
    ax[0].set_xlim(new_xlim)
    ax[0].set_ylim(new_ylim)
    
    ax[1].set_xlim(-0.5, M.shape[1] - 0.5)
    ax[1].set_ylim(-0.5, M.shape[0] - 0.5)
    ax[1].invert_yaxis()

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel() + 0.01
    h = 1 - np.abs(M).ravel() - 0.01
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           norm=Normalize(vmin=-1, vmax=1),
                           transOffset=ax[1].transData, array=M.ravel(), cmap=cmap)
    ax[1].add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax[1].set_xticks(np.arange(M.shape[1]))
        ax[1].set_xticklabels(data.columns, rotation=0)
        ax[1].set_yticks(np.arange(M.shape[0]))
        ax[1].set_yticklabels(data.index)
    
    cb = plt.colorbar(ec, ax=ax[1])

    plt.tight_layout()
    plt.show()