import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from code.utility.data_loader import load_data
from code.utility.path_utils import get_path_from_root


# 1. Descriptive Statistics
def descriptive_statistics(data):
    statistics = data.describe()
    statistics.to_csv(os.path.join(get_path_from_root('results/EDA/statistics'), 'descriptive_statistics.csv'))


# 2. Distribution Visualization
# We'll visualize the distribution of each variable to understand its spread, central tendency, and shape
def distribution(data):
    for column in data.columns:
        if column != 'IDNO':
            plt.figure(figsize=(10, 5))
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(os.path.join(get_path_from_root('results/EDA/distribution_visualization'),
                                     f'{column}_distribution.png'), dpi=500, bbox_inches='tight')

            skewness_values = data.skew()
            skewness_values.to_csv()


# 3. Outlier Detection
def outlier_detection():
    for column in data.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.savefig(os.path.join(get_path_from_root('results/EDA/boxplot_outliers'),
                                 f'{column}_outlier.png'), dpi=500, bbox_inches='tight')

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_counts = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    outlier_counts.to_csv(os.path.join(get_path_from_root('results/EDA/statistics'), 'outlier_count.csv'))


# 4. Correlation analysis
def corr_matrix():
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(get_path_from_root('results/EDA/correlation_matrix'),
                             'correlation_matrix.png'), dpi=1000, bbox_inches='tight')

    correlation_matrix.to_csv(os.path.join(get_path_from_root('results/EDA/statistics'), 'correlation_matrix.csv'))

    return correlation_matrix


# 5. Visualize relationships
def visualize_relationships(data):
    threshold = 0.5
    correlation_with_bodyfat = data.corr()['BODYFAT'].drop('BODYFAT', axis=0)
    most_correlated_features = correlation_with_bodyfat[correlation_with_bodyfat.abs() > threshold].index.tolist()

    for feature in most_correlated_features:
        plt.scatter(data[feature], data['BODYFAT'])
        plt.xlabel(feature)
        plt.ylabel('BODYFAT')
        plt.savefig(os.path.join(get_path_from_root('results/EDA/visualize_relationships',
                                                    f'{feature}_x_BODYFAT.png')), dpi=1000, bbox_inches='tight')


def kde_plots(data):
    for column in data.columns:
        if column != 'IDNO':
            plt.figure(figsize=(10, 5))
            sns.kdeplot(data[column])
            plt.title(f'KDE of {column}')
            plt.savefig(os.path.join(get_path_from_root('results/EDA/KDE',
                                                        f'{column}_KDE.png')), dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    # Load the data
    data = load_data()
    """
    descriptive_statistics(data)
    distribution(data)
    outlier_detection(data)
    corr_matrix(data)
    visualize_relationships(data)
    """
    kde_plots(data)
