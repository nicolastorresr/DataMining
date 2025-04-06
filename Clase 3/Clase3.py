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
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
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

def visualize_color_schemes(fig_width=14, fig_height=7, show_quantitative=True):
    # Set up the figure
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Add a placeholder axis for compatibility with Jupyter
    main_ax = fig.add_subplot(111)
    main_ax.axis('off')
        
    # Title and headers styling function
    def add_text(x, y, text, fontsize, weight='normal', ha='center', va='center', color='#333333'):
        fig.text(x, y, text, fontsize=fontsize, fontweight=weight, ha=ha, va=va, color=color)
    
    # Helper function to draw color scales
    def draw_color_scale(y_pos, colors, label, scale_type='sequential'):
        x_start = 0.06 if scale_type == 'sequential' else 0.56
        width = 0.38
        height = 0.05

        # Position for label
        if scale_type == 'sequential':
            label_x = 0.04
        else:
            label_x = 0.54

        # Add label
        add_text(label_x, y_pos + 0.025, label, fontsize=16, weight='normal', ha='right')

        # Draw color blocks
        n_colors = len(colors)
        block_width = width / n_colors

        for i, color in enumerate(colors):
            rect = patches.Rectangle((x_start + i * block_width, y_pos), block_width, height, 
                                     facecolor=color, edgecolor='white', linewidth=0.5)
            fig.add_artist(rect)

    
    # Section 1: Quantitative Data
    if show_quantitative:

        # Section 1: Quantitative Data
        add_text(0.5, 0.87, 'QUANTITATIVE DATA - SEQUENTIAL OR DIVERGING COLORS', fontsize=20, weight='bold')

        # Labels for sequential and diverging sections
        add_text(0.25, 0.8, 'Sequential scales', fontsize=18, weight='bold')
        add_text(0.75, 0.8, 'Diverging scales', fontsize=18, weight='bold')

        # Define sequential color scales
        blues_seq = ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"]
        viridis_seq = ["#440154", "#482878", "#3E4989", "#31688E", "#26828E", "#1F9E89", "#35B779", "#6DCD59", "#B4DE2C", "#FDE725"]
        ylorbr_seq = ["#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FE9929", "#EC7014", "#CC4C02", "#993404", "#662506"]

        # Define diverging color scales
        piyg_div = ["#C51B7D", "#DE77AE", "#F1B6DA", "#FDE0EF", "#F7F7F7", "#E6F5D0", "#B8E186", "#7FBC41", "#4D9221"]
        rdbu_div = ["#B2182B", "#D6604D", "#F4A582", "#FDDBC7", "#F7F7F7", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC"]
        brbg_div = ["#8C510A", "#BF812D", "#DFC27D", "#F6E8C3", "#F5F5F5", "#C7EAE5", "#80CDC1", "#35978F", "#01665E"]
    
        # Draw sequential scales
        draw_color_scale(0.7, blues_seq, 'Blues', 'sequential')
        draw_color_scale(0.6, viridis_seq, 'Viridis', 'sequential')
        draw_color_scale(0.5, ylorbr_seq, 'YlOrBr', 'sequential')

        # Draw diverging scales
        draw_color_scale(0.7, piyg_div, 'PiYG', 'diverging')
        draw_color_scale(0.6, rdbu_div, 'RdBu', 'diverging')
        draw_color_scale(0.50, brbg_div, 'BrBG', 'diverging')
    
    
    # Section 2: Categorical Data
    if not show_quantitative:
    
        # Section 2: Categorical Data
        add_text(0.5, 0.84, 'CATEGORICAL DATA - QUALITATIVE COLORS', fontsize=20, weight='bold')
        
        # Define categorical color scales
        colorblind = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#A9734E", "#F0A3BB", "#999999"]
        set2 = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]
        accent = ["#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#666666"]
        pastel1 = ["#FBB4AE", "#B3CDE3", "#CCEBC5", "#DECBE4", "#FED9A6", "#FFFFCC", "#E5D8BD", "#FDDAEC", "#F2F2F2"]

        # Draw categorical scales
        draw_color_scale(0.7, colorblind, 'Colorblind', 'sequential')
        draw_color_scale(0.6, set2, 'Set2', 'sequential')
        draw_color_scale(0.7, accent, 'Accent', 'diverging')
        draw_color_scale(0.6, pastel1, 'Pastel1', 'diverging')
    
    plt.tight_layout(rect=[0, 0.7, 1, 0.8])
    # Show the plot
    plt.show()

def plot_usage_guidelines(scale='categorical'):
    # Set up the figure with a light gray background
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure(figsize=(12, 3))

    gs = GridSpec(2, 4, height_ratios=[0.8, 1])

    # Column headers
    fig.text(0.18, 0.88, 'SCALES', fontsize=14, fontweight='bold')
    fig.text(0.42, 0.88, 'Sequential', fontsize=14, fontweight='bold', ha='center')
    fig.text(0.67, 0.88, 'Diverging', fontsize=14, fontweight='bold', ha='center')
    fig.text(0.90, 0.88, 'Categorical', fontsize=14, fontweight='bold', ha='center')
    
    fig.text(0.12, 0.5, scale, fontsize=14, fontweight='bold')

    # Define colors for scales
    # Sequential colors (red gradient)
    sequential_colors = ['#FBE5E5', '#F8B7B2', '#EF6F6A', '#D3292A', '#AB0520', '#720017']
    # Diverging colors
    diverging_colors = ['#E67717', '#FACA86', '#FFFFFF', '#8A62AA', '#3E2A5A']
    # Categorical colors
    categorical_colors = ['#1E5288', '#E67717', '#AB0520', '#6ec3ff', '#0C8344']

    # Function to draw check or X mark
    def draw_checkbox(ax, x, y, size=0.12, checked=True):
        rect = patches.Rectangle((x, y), size, size, linewidth=1, edgecolor='black', facecolor='white')
        ax.add_patch(rect)
        if checked:
            # Draw checkmark
            ax.plot([x+size*0.2, x+size*0.4, x+size*0.8], [y+size*0.5, y+size*0.3, y+size*0.7], 
                    color='black', linewidth=2)
        else:
            # Draw X
            ax.plot([x+size*0.2, x+size*0.8], [y+size*0.2, y+size*0.8], color='black', linewidth=2)
            ax.plot([x+size*0.2, x+size*0.8], [y+size*0.8, y+size*0.2], color='black', linewidth=2)

    # Create axes for each cell in our grid
    axes = {}
    
    for j, col in enumerate(['sequential', 'diverging', 'categorical']):
        ax = fig.add_subplot(gs[1, j+1])
        axes[f"{scale}_{col}"] = ax
        ax.axis('off')

    # Define checkbox statuses based on the image
    checkbox_status = {
        'categorical_sequential': False,
        'categorical_diverging': False,
        'categorical_categorical': True,
        'ordinal_sequential': True,
        'ordinal_diverging': True,
        'ordinal_categorical': False,
        'interval_sequential': True,
        'interval_diverging': False,
        'interval_categorical': False,
        'ratio_sequential': False,
        'ratio_diverging': True,
        'ratio_categorical': False
    }

    # Draw checkboxes with check or X marks
    for j, col in enumerate(['sequential', 'diverging', 'categorical']):
        key = f"{scale}_{col}"
        ax = axes[key]
        draw_checkbox(ax, -0.4, 0.4, checked=checkbox_status[key])

    # Draw the color scales
    # Sequential scales
    yyy=0.38
    
    ax = axes[f"{scale}_sequential"]

    # Draw sequential scale
    if scale == 'categorical':
        # A-E labels with color blocks
        for j, letter in enumerate(['A', 'B', 'C', 'D', 'E']):
            rect = patches.Rectangle((j*0.15, 0.4), 0.14, 0.2, 
                                     facecolor=sequential_colors[j], edgecolor='none')
            ax.add_patch(rect)
            ax.text(j*0.15 + 0.07, yyy, letter, ha='center')

    elif scale == 'ordinal':
        # Low to High gradient
        for j in range(6):
            rect = patches.Rectangle((j*0.15, 0.4), 0.14, 0.2, 
                                     facecolor=sequential_colors[j], edgecolor='none')
            ax.add_patch(rect)
        ax.text(0, yyy, 'Low', ha='left')
        ax.text(0.9, yyy, 'High', ha='right')

    elif scale == 'interval':
        # 1-6 gradient
        for j in range(6):
            rect = patches.Rectangle((j*0.15, 0.4), 0.14, 0.2, 
                                     facecolor=sequential_colors[j], edgecolor='none')
            ax.add_patch(rect)
            ax.text(j*0.15 + 0.07, yyy, str(j+1), ha='center')

    elif scale == 'ratio':
        # -5 to 5 gradient
        labels = ['-5', '-3', '-1', '1', '3', '5']
        for j in range(6):
            rect = patches.Rectangle((j*0.15, 0.4), 0.14, 0.2, 
                                     facecolor=sequential_colors[j], edgecolor='none')
            ax.add_patch(rect)
            ax.text(j*0.15 + 0.07, yyy, labels[j], ha='center')

    # Diverging scales
    ax = axes[f"{scale}_diverging"]

    # Draw divergent scale        
    for j in range(5):
        rect = patches.Rectangle((j*0.15, 0.4), 0.14, 0.2, 
                                 facecolor=diverging_colors[j], edgecolor='none')
        ax.add_patch(rect)

    # Add labels based on scale
    if scale == 'categorical':
        ax.text(0.07, yyy, 'A', ha='center')
        ax.text(0.22, yyy, 'B', ha='center')
        ax.text(0.37, yyy, 'C', ha='center')
        ax.text(0.52, yyy, 'D', ha='center')
        ax.text(0.67, yyy, 'E', ha='center')
    elif scale == 'ordinal':
        ax.text(0.1, yyy, 'Low', ha='center')
        ax.text(0.35, yyy, 'Medium', ha='center')
        ax.text(0.6, yyy, 'High', ha='center')
    elif scale == 'interval':
        ax.text(0.07, yyy, '1', ha='center')
        ax.text(0.22, yyy, '2', ha='center')
        ax.text(0.37, yyy, '3', ha='center')
        ax.text(0.52, yyy, '4', ha='center')
        ax.text(0.67, yyy, '5', ha='center')
    elif scale == 'ratio':
        ax.text(0.07, yyy, '-4', ha='center')
        ax.text(0.22, yyy, '-2', ha='center')
        ax.text(0.37, yyy, '0', ha='center')
        ax.text(0.52, yyy, '2', ha='center')
        ax.text(0.67, yyy, '4', ha='center')

    # Categorical scales
    ax = axes[f"{scale}_categorical"]

    # Draw categorical scales based on scale
    if scale == 'categorical':
        # A-E blocks with different colors
        for j, (letter, color) in enumerate(zip(['A', 'B', 'C', 'D', 'E'], categorical_colors)):
            rect = patches.Rectangle((j*0.15, 0.4), 0.14, 0.2, 
                                    facecolor=color, edgecolor='none')
            ax.add_patch(rect)
            ax.text(j*0.15 + 0.07, yyy, letter, ha='center')

    elif scale == 'ordinal':
        # Low, Medium, High with colors
        texts = ['Low', 'Medium', 'High']
        colors = [categorical_colors[0], categorical_colors[1], categorical_colors[2]]
        positions = [0, 0.3, 0.6]

        for j, (text, color, pos) in enumerate(zip(texts, colors, positions)):
            rect = patches.Rectangle((pos, 0.4), 0.14, 0.2, 
                                    facecolor=color, edgecolor='none')
            ax.add_patch(rect)
            ax.text(pos + 0.07, yyy, text, ha='center')

    elif scale == 'interval':
        # 1-2, 3-4, 5-6 groups
        texts = ['1-2', '3-4', '5-6']
        colors = [categorical_colors[0], categorical_colors[1], categorical_colors[2]]
        positions = [0, 0.3, 0.6]

        for j, (text, color, pos) in enumerate(zip(texts, colors, positions)):
            rect = patches.Rectangle((pos, 0.4), 0.14, 0.2, 
                                    facecolor=color, edgecolor='none')
            ax.add_patch(rect)
            ax.text(pos + 0.07, yyy, text, ha='center')

    elif scale == 'ratio':
        # -6 to -2, -2 to 2, 2 to 6 groups
        texts = ['-6 to -2', '-2 to 2', '2 to 6']
        colors = [categorical_colors[0], categorical_colors[1], categorical_colors[2]]
        positions = [0, 0.3, 0.6]

        for j, (text, color, pos) in enumerate(zip(texts, colors, positions)):
            rect = patches.Rectangle((pos, 0.4), 0.14, 0.2, 
                                    facecolor=color, edgecolor='none')
            ax.add_patch(rect)
            ax.text(pos + 0.07, yyy, text, ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()