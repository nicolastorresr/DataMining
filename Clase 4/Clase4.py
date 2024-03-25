#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
from matplotlib import image as mpimg
from mpl_toolkits import mplot3d
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array

def aesthetics_types():
    # Create the figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.5)

    # Plot 1: Position
    axes[0, 0].arrow(0.2, 0.4, 0.7, 0, head_width=0.03, head_length=0.05, 
                     fc='black', ec='black', linewidth=0.75)
    axes[0, 0].arrow(0.5, 0.2, 0, 0.5, head_width=0.03, head_length=0.05, 
                     fc='black', ec='black', linewidth=0.75)
    axes[0, 0].text(0.5, 0.8, 'y', size=12, va='bottom', ha='center', fontfamily='serif')
    axes[0, 0].text(1, 0.4, 'x', size=12, va='center', ha='left', fontfamily='serif')
    axes[0, 0].set_xlim(-0.2, 1.2)
    axes[0, 0].set_xlim(-0.2, 1.2)
    axes[0, 0].set_ylim(-0.2, 1.2)
    axes[0, 0].set_title('Position')

    # Plot 2: Shape
    markers = ['o', 's', 'D', '^']
    for i, marker in enumerate(markers):
        sns.scatterplot(x=[i/4 + 0.5], y=[0.5], marker=marker, edgecolor='black',
                        s=250, ax=axes[0, 1], color='lightgray')
    axes[0, 1].set_title('Shape')
    axes[0, 1].set_xticks([])  # Remove x-axis ticks
    axes[0, 1].set_xticklabels([])  # Remove x-axis labels
    axes[0, 1].set_xlim(0.4,1.4)

    # Plot 3: Size
    scatter = sns.scatterplot(data=pd.DataFrame({'x': (.5 + np.arange(4))/4, 
                                                 'y': 0.5, 'size': [2, 5, 8, 11]}),
                              x='x', y='y', size='size', ax=axes[0, 2], edgecolor='black',
                              sizes=[50, 200, 400, 800], color='lightgray', marker='o')
    axes[0, 2].set_title('Size')
    axes[0, 2].get_legend().remove()
    axes[0, 2].set_xlim(0.1,1.0)
    scatter.set(xticks=[], yticks=[], xlabel='', ylabel='')

    # Plot 4: Color
    sns.heatmap(data=pd.DataFrame({'x': 0.15 + 0.2333*np.arange(4)}), 
                cmap='rainbow', ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title('Color')

    # Plot 5: Line Width
    for i, width in enumerate(reversed(range(1, 5))):
        axes[1, 1].plot([0.05, 0.95], [(1.5 + i)/6, (1.5 + i)/6], 
                        linewidth=width, color='black')
    axes[1, 1].set_title('Line Width')

    # Plot 6: Line Type
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    for i, linestyle in enumerate(linestyles):
        axes[1, 2].plot([0.05, 0.95], [(1.5 + i)/6, (1.5 + i)/6], 
                        linestyle=linestyle, linewidth=2, color='black')
        axes[1, 2].set_xticklabels([])
    axes[1, 2].set_title('Line Type')
    axes[1, 2].set_xticklabels([])

    # Set common labels for subplots
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])

    # Add labels to the entire figure
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)

    plt.show()

def visual_encoding():
    # Sample data
    data = np.array([1, 2, 3])

    # Create a figure with subplots (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(12, 2))
    
    # Bar plot with 3 bins (color: blue)
    ax1 = axes[0]
    ax1.bar(range(3), data, color='C0', width=0.5, alpha=0.7)
    ax1.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(3)
    ax1.spines['left'].set_linewidth(3)

    # Scatter plot with 3 points (color: blue)
    ax2 = axes[1]
    ax2.scatter(range(3), data, color='C0', s=300, alpha=0.7)
    ax2.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim([-1, 3])
    ax2.set_ylim([0, 4])
    ax2.spines['bottom'].set_linewidth(3)
    ax2.spines['left'].set_linewidth(3)

    # Scatter plot with 3 points (1 blue, 2 red)
    ax3 = axes[2]
    ax3.scatter(range(3), data, color=['C3', 'C0', 'C3'], s=300, alpha=0.7)
    ax3.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlim([-1, 3])
    ax3.set_ylim([0, 4])
    ax3.spines['bottom'].set_linewidth(3)
    ax3.spines['left'].set_linewidth(3)

    # Scatter plot with 3 points (different sizes)
    ax4 = axes[3]
    ax4.scatter(range(3), data, color=['C3', 'C0', 'C3'], s=10 * np.array([30, 70, 120]), alpha=0.7)
    ax4.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_xlim([-1, 3])
    ax4.set_ylim([0, 4])
    ax4.spines['bottom'].set_linewidth(3)
    ax4.spines['left'].set_linewidth(3)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5) 
    # Show the plot
    plt.show()

def occlusion():
    # Generate random data for both plots
    data = np.random.rand(5, 5)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Use consistent colors for both plots
    colors = ['skyblue', 'lightgreen', 'gold', 'violet', 'teal']

    # Clear 2D Barplot
    ax2 = axes[1]
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    bottoms = np.zeros(5)

    for i in range(5):
        ax2.bar(np.arange(5), data[i], bottom=bottoms, 
                color=colors[i], label=f'Category {i+1}')
        bottoms += data[i]

    ax2.set_title('Clear 2D Barplot')
    ax2.legend()

    # Distracting 3D Barplot
    ax1 = axes[0]
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1 = fig.add_subplot(121, projection='3d')

    # Create 3D barplot with correct dimensions
    xpos, ypos = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    dx = dy = 0.8
    dz = data.flatten(order='F')[:len(xpos.flatten())]

    for i, color in zip(range(len(xpos.flatten())), colors * len(ypos)):
        ax1.bar3d(xpos.flatten()[i], ypos.flatten()[i], 
                  np.zeros_like(xpos).flatten()[i], 
                  dx, dy, dz[i], shade=True, color=color)

    ax1.set_title('Distracting 3D Barplot')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def highlight_lines(highlighted_countries=[1, 5, 10, 15, 20]):
    # Sample data for average years of schooling for different countries
    years = np.array([1997, 1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017], dtype=int)
    countries = [f'Country {i}' for i in range(1, 21)]

    # Generate random slopes and starting points for each country
    slopes = np.random.uniform(0.5, 2, size=len(countries))
    starting_points = np.random.uniform(2, 14, size=len(countries))

    data = np.zeros((len(countries), len(years)))
    for i in range(len(countries)):
        data[i] = starting_points[i] + slopes[i] * (years - 1997)  # Example linear increase

    # Create a figure with subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Line plot for all countries
    ax1 = axes[0]
    for i in range(len(countries)):
        ax1.plot(years, data[i], label=countries[i])

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Years of Schooling')
    ax1.set_title('Average Years of Schooling Over Time - All Countries')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', color='grey', alpha=0.5)
    # Highlight specific countries in the right subplot
    ax2 = axes[1]
    highlighted_countries = [f'Country {i}' for i in highlighted_countries]
    highlighted_colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i in range(len(countries)):
        if countries[i] in highlighted_countries:
            ax2.plot(years, data[i], label=countries[i], 
                     color=highlighted_colors[highlighted_countries.index(countries[i])])
            ax2.text(years[-1], data[i, -1], countries[i], 
                     color=highlighted_colors[highlighted_countries.index(countries[i])],
                     verticalalignment='bottom', horizontalalignment='left', 
                     fontsize=8, fontweight='bold')
        else:
            ax2.plot(years, data[i], color='lightgrey')

    ax2.set_xlabel('Year')
    ax2.set_ylabel('Average Years of Schooling')
    ax2.set_title('Highlighted Countries')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # Remove legend from the second plot
    ax2.legend().set_visible(False)

    # Disable minor ticks for both plots
    ax1.set_xticks(years)
    ax1.set_xticks([], minor=True)
    ax2.grid(axis='y', linestyle='--', color='grey', alpha=0.5)
    ax2.set_xticks(years)
    ax2.set_xticks([], minor=True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def proportions(n=20):
    # Generate random data with at least 20 categories
    categories = [f'Category {i}' for i in range(1, n+1)]
    data = np.random.randint(1, 30, size=len(categories))

    # Create a figure with subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    # Pie plot
    ax1 = axes[0]
    ax1.pie(data, labels=None, autopct=None, 
            startangle=90, colors=plt.cm.Paired.colors)

    # Bar plot
    ax2 = axes[1]
    ax2.bar(np.arange(len(categories)), data, color=plt.cm.Paired.colors)
    ax2.set_xticks([])  # Remove x-labels

    # Add "%" to the y-axis labels
    def percentage_formatter(x, pos):
        return f"{x:.0f}%"

    ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def consistency():
    plt.figure(figsize=(10, 5))
    # Datos
    foods = ['Salmón', 'Banana', 'Naranja', 'Uva', 'Aceituna', 'Durazno']
    cantidades = [30, 45, 25, 20, 40, 10]
    colores = {'Banana': 'gold', 'Naranja': 'orange', 'Uva': 'purple', 
                      'Salmón': 'salmon', 'Aceituna': 'olive', 'Durazno': 'peachpuff'}

    # Configuración de la primera fila
    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(foods, cantidades, color=[colores[x] for x in foods], edgecolor='black', width=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax1.xaxis.tick_bottom()
    ax1.xaxis.set_tick_params(width=0.5)
    ax1.yaxis.set_tick_params(width=0.5)
    ax1.grid(axis='y', linestyle='-', color='white')

    plt.subplot(2, 2, 2)
    plt.pie(cantidades, labels=foods, colors=[colores[x] for x in foods])

    # Configuración de la segunda fila con escala de grises
    colores_grises = np.linspace(0, 1, len(foods))  # Crear una secuencia de grises de 0 a 1

    ax2 = plt.subplot(2, 2, 3)
    ax2.bar(foods, cantidades, color=plt.cm.gray(colores_grises), edgecolor='black', width=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_left()
    ax2.xaxis.tick_bottom()
    ax2.xaxis.set_tick_params(width=0.5)
    ax2.yaxis.set_tick_params(width=0.5)
    ax2.grid(axis='y', linestyle='-', color='white')

    plt.subplot(2, 2, 4)
    plt.pie(cantidades, labels=foods, colors=plt.cm.bwr(colores_grises))

    # Ajustes de diseño
    plt.tight_layout()
    plt.show()

def irrelevant_information(subset_states = ['California', 'Texas', 'Florida', 'New York', 'Illinois']):
    df = pd.read_csv("data/population.csv")

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Scatter plot with different colors for each state
    ax1 = axes[0]
    sns.scatterplot(x='Population_2000', y='Population_Growth_2000_2010', 
                    hue='State', data=df, palette='tab20', ax=ax1)
    ax1.set_title('Population Growth vs Population Size (Colored by State)')
    ax1.set_xlabel('Population Size in 2000')
    ax1.set_ylabel('Population Growth from 2000 to 2010')
    ax1.legend(title='State', loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), ncol=7,  prop={'size': 6})
    # Move the legend below the chart
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(linestyle='-', color='grey', alpha=0.3)
    
    # Scatter plot with colored regions and labeled subset of states
    ax2 = axes[1]
    sns.scatterplot(x='Population_2000', y='Population_Growth_2000_2010', 
                    hue='Region', data=df, palette='tab20', ax=ax2)
    
    subset_df = df[df['State'].isin(subset_states)]
    for i in range(len(subset_df)):
        ax2.text(subset_df['Population_2000'].iloc[i], 
                 subset_df['Population_Growth_2000_2010'].iloc[i], 
                 subset_df['State'].iloc[i])
    ax2.set_title('Population Growth vs Population Size (Colored by Region)')
    ax2.set_xlabel('Population Size in 2000')
    ax2.set_ylabel('Population Growth from 2000 to 2010')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.grid(linestyle='-', color='grey', alpha=0.3)
    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def scales_map(ylim=[1.40, 1.90]):
    # Provided data for average male height per country
    data = [1.84, 1.75, 1.69, 1.65, 1.62]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Load the human silhouette image
    img_path = "img/silhouette.png"
    img = mpimg.imread(img_path)
    
    # Plot the silhouette bars with adjusted size
    bar_width = 0.9
    bar_positions = np.arange(len(data)) - bar_width / 2

    for pos, value in zip(bar_positions, data):
        ax.imshow(img, extent=[pos, pos + bar_width, ylim[0], value], aspect='auto', zorder=1)

    # Customize the plot
    ax.set_xlim([-1, len(data)])
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title and labels
    ax.set_title("Average Male Height per Country")
    ax.set_ylabel("Height (meters)")
    ax.set_xlabel("Country")
    yticks = []
    ytickslabels = []
    # Add y-labels
    num_elements = int((ylim[1] - ylim[0]) / 0.1) + 1
    rango = np.linspace(ylim[0], ylim[1], num_elements)
    for i in list(rango):
        i = round(i,2)
        yticks.append(i)
        ytickslabels.append(str(i)+"0")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytickslabels)

    # Add x-labels with values
    countries = ["Netherlands", "UK", "Chile", "India", "Bolivia"]
    ax.set_xticks(bar_positions + bar_width / 2)
    ax.set_xticklabels([f"{country}\n({height:.2f} m)" for country, 
                        height in zip(countries, data)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', color='grey', alpha=0.2)
    # Show the plot
    plt.show()

def grayscale_cmap(cmap_name):
    """Return a grayscale version of the given colormap"""
    cmap = plt.colormaps[cmap_name]
    colors = cmap(np.arange(cmap.N))

    # Convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

def view_colormap(cmap_name):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.colormaps[cmap_name]
    colors = cmap(np.arange(cmap.N))

    grayscale_cmap_obj = grayscale_cmap(cmap_name)
    grayscale = grayscale_cmap_obj(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    
    ax[0].imshow([to_rgba_array(colors)], extent=[0, 10, 0, 1])
    ax[0].set_title('Original Colormap')

    ax[1].imshow([to_rgba_array(grayscale)], extent=[0, 10, 0, 1])
    ax[1].set_title('Grayscale Equivalent')

    plt.show()

def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')