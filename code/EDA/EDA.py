import logging
import os.path

import matplotlib.pyplot as plt
import seaborn as sns

from code.utility.data_loader import load_data
from code.utility.path_utils import get_path_from_root

RESULTS_EDA_DIR = 'results/EDA'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_results(data, operation, result_type, filename):
    """
    Save the results to the specified location.

    :param data: EDA Results to be saved
    :param result_type: 'statistic', 'distribution_visualization', etc.
    :param filename: Desired name of the file being stored
    :param operation: EDA, preprocessing, modeling, etc.
    :param subdir: Subdirectory under the result_type directory
    """

    save_path = get_path_from_root('results', operation, result_type, filename)

    # Check if directory exists and if not, create it
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logger.info(f"Saving {filename} to {save_path}.\n")
    data.to_csv(save_path)


# 1. Descriptive Statistics
def descriptive_statistics(data, operation, result_type, filename):
    statistics = data.describe()
    save_results(statistics, operation, result_type, filename)


# 2. Distribution Visualization
# We'll visualize the distribution of each variable to understand its spread, central tendency, and shape
def distribution(data, operation, result_type, filename):
    logger.info('Exploring distribution of each of the features...\n')

    for column in data.columns:
        if column != 'IDNO':
            plt.figure(figsize=(10, 5))
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(get_path_from_root('results', operation, 'distribution_visualization',
                                           f'{column}_distribution.png'), dpi=500, bbox_inches='tight')
            plt.close()

    skewness_values = data.skew()
    save_results(skewness_values, operation, result_type, filename)


# 3. Outlier Detection
def outlier_detection(data, operation, result_type, filename):
    logger.info('Exploring outliers in each of the features...\n')

    for column in data.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.savefig(get_path_from_root('results', operation, 'boxplot_outliers', f'{column}_outlier.png'),
                    dpi=500,
                    bbox_inches='tight')
        plt.close()

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_counts = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    save_results(outlier_counts, operation, result_type, filename)


# 4. Correlation analysis
def corr_matrix(data, operation, result_type, filename):
    logger.info('Exploring correlation between each of the features...\n')

    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(
        get_path_from_root('results', f'{operation}', 'correlation_matrix', 'correlation_matrix.png'),
        dpi=1000,
        bbox_inches='tight'
    )
    save_results(correlation_matrix, operation, result_type, 'correlation_matrix.csv')

    # Saving correlation with the target variable
    correlation_with_bodyfat = data.corr()['BODYFAT'].drop('BODYFAT', axis=0)
    save_results(correlation_with_bodyfat, operation, result_type, filename + '.csv')
    plt.close()


# 5. Visualize relationships
def visualize_relationships(data):
    logger.info('Exploring relationships between each of the features...\n')

    threshold = 0.5
    correlation_with_bodyfat = data.corr()['BODYFAT'].drop('BODYFAT', axis=0)
    most_correlated_features = correlation_with_bodyfat[correlation_with_bodyfat.abs() > threshold].index.tolist()

    sns.set_style('whitegrid')

    for feature in most_correlated_features:
        plt.figure(figsize=(10, 6))

        # Using seaborn's reg-plot to plot scatter with a regression line
        sns.regplot(x=data[feature], y=data['BODYFAT'], scatter_kws={'s': 30, 'alpha': 0.7}, line_kws={'color': 'red'})

        # Title and labels
        plt.title(f'Relationship between {feature} and BODYFAT', fontsize=15)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('BODYFAT', fontsize=12)

        # Save the plot
        plt.savefig(
            get_path_from_root('results', 'EDA', 'visualize_relationships', f'{feature}_x_BODYFAT.png'),
            dpi=500,
            bbox_inches='tight'
        )
        plt.close()


# Kernel Density Estimation - Smoother form of distribution analysis
def kde_plots(data):
    for column in data.columns:
        if column != 'IDNO':
            plt.figure(figsize=(10, 5))
            sns.kdeplot(data[column])
            plt.title(f'KDE of {column}')
            plt.savefig(get_path_from_root('results', 'EDA', 'KDE', f'{column}_KDE.png'),
                        dpi=200,
                        bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    # Load the data
    data = load_data()
    descriptive_statistics(data, 'EDA', 'statistics', 'descriptive_statistics.csv')
    # distribution(data, 'EDA', 'statistics', 'skewness_values.csv')
    # outlier_detection(data, 'EDA', 'statistics', 'outlier_count.csv')
    # corr_matrix(data, 'EDA', 'statistics', 'correlation_with_bodyfat')
    # visualize_relationships(data)
    # kde_plots(data)
