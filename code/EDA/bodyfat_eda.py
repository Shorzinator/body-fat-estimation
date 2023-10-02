import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from code.utility.data_loader import load_data
from code.utility.path_utils import get_path_from_root


# 1. Descriptive Statistics
def descriptive_statistics(data):
    statistics = data.describe()
    statistics.to_csv(os.path.join(get_path_from_root('image', 'EDA', 'statistics'), 'descriptive_statistics.csv'))


# 2. Distribution Visualization
# We'll visualize the distribution of each variable to understand its spread, central tendency, and shape
def distribution(data):
    for column in data.columns:
        if column != 'IDNO':
            plt.figure(figsize=(10, 5))
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(os.path.join(get_path_from_root('image', 'EDA', 'distribution_visualization'),
                                     f'{column}_distribution.png'), dpi=500, bbox_inches='tight')


# 3. Outlier Detection
def outlier_detection():
    for column in data.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.savefig(os.path.join(get_path_from_root('image', 'EDA', 'boxplot_outliers'),
                                 f'{column}_outlier.png'), dpi=500, bbox_inches='tight')


# 4. Correlation analysis
def corr_matrix():
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(get_path_from_root('image', 'EDA', 'correlation_matrix'),
                             'correlation_matrix.png'), dpi=1000, bbox_inches='tight')

    correlation_matrix.to_csv(os.path.join(get_path_from_root('image', 'EDA', 'correlation_matrix'),
                                           'correlation_matrix.csv'))


# 5. Visualize relationships
def pair_plot():
    sns.pairplot(data)
    plt.show()


if __name__ == "__main__":
    # Load the data
    data = load_data()

    descriptive_statistics(data)
    distribution(data)
    outlier_detection(data)
    corr_matrix(data)
    pair_plot(data)
