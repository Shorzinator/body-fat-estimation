import logging
import os.path
import scipy.stats as stats

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
    """
    # Modify the operation if it's post_preprocessing
    if operation == 'post_preprocessing':
        operation_path = ['results', 'post_processing', result_type]
    else:
        operation_path = ['results', operation, result_type]

    save_path = get_path_from_root(*operation_path, filename)

    logger.info(f"Saving {filename} to {save_path}.\n")
    data.to_csv(save_path)


def generate_plot(data, column, operation, plot_type, file_suffix, plot_function, *args, **kwargs):
    """
    Helper function to generate and save plots.
    """
    plt.figure(figsize=(10, 5))
    plot_function(data[column], *args, **kwargs)
    plt.title(f'{plot_type} of {column}')
    plt.savefig(get_path_from_root('results', operation, plot_type, f'{column}_{file_suffix}.png'),
                dpi=500, bbox_inches='tight')
    plt.close()


def descriptive_statistics(data, filename, operation='EDA', result_type='statistics'):
    statistics = data.describe()
    save_results(statistics, operation, result_type, filename)


def distribution(data, operation='EDA', result_type='statistics'):
    logger.info('Exploring distribution of each of the features...\n')
    for column in data.columns:
        if column != 'IDNO':
            generate_plot(data, column, operation, 'Distribution', 'distribution', sns.histplot, kde=True)

    skewness_values = data.skew()
    save_results(skewness_values, operation, result_type, 'skewness_value.csv')


def outlier_detection(data, operation='EDA', result_type='statistics'):
    logger.info('Exploring outliers in each of the features...\n')
    for column in data.columns:
        generate_plot(data, column, operation, 'Boxplot', 'outlier', sns.boxplot)

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_counts = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    save_results(outlier_counts, operation, result_type, 'outlier_count.csv')


def corr_matrix(data, filename, operation='EDA', result_type='statistics'):
    logger.info('Exploring correlation between each of the features...\n')
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(get_path_from_root('results', operation, 'correlation_matrix', 'correlation_matrix.png'),
                dpi=1000, bbox_inches='tight')
    save_results(correlation_matrix, operation, result_type, 'correlation_matrix.csv')
    correlation_with_bodyfat = data.corr()['BODYFAT'].drop('BODYFAT', axis=0)
    save_results(correlation_with_bodyfat, operation, result_type, filename)


def visualize_relationships(data, operation='EDA'):
    logger.info('Exploring relationships between each of the features...\n')
    threshold = 0.5
    correlation_with_bodyfat = data.corr()['BODYFAT'].drop('BODYFAT', axis=0)
    most_correlated_features = correlation_with_bodyfat[correlation_with_bodyfat.abs() > threshold].index.tolist()
    sns.set_style('whitegrid')
    for feature in most_correlated_features:
        plt.figure(figsize=(10, 6))
        sns.regplot(x=data[feature], y=data['BODYFAT'], scatter_kws={'s': 30, 'alpha': 0.7}, line_kws={'color': 'red'})
        plt.title(f'Relationship between {feature} and BODYFAT')
        plt.savefig(get_path_from_root('results', operation, 'visualize_relationships', f'{feature}_x_BODYFAT.png'),
                    dpi=500, bbox_inches='tight')
        plt.close()


def kde_plots(data, operation='EDA'):
    for column in data.columns:
        if column != 'IDNO':
            generate_plot(data, column, operation, 'KDE', 'KDE', sns.kdeplot)


def qq_plots(data, operation='EDA'):
    logger.info('Generating Q-Q plots for each feature...\n')
    for column in data.columns:
        if column != 'IDNO':
            plt.figure(figsize=(10, 5))
            stats.probplot(data[column], plot=plt)
            plt.title(f'Q-Q plot of {column}')
            plt.savefig(get_path_from_root('results', operation, 'QQ_plots', f'{column}_QQ.png'),
                        dpi=500, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    data = load_data()
    # descriptive_statistics(data, 'descriptive_statistics.csv')
    # distribution(data, 'skewness_values.csv')
    # outlier_detection(data, 'outlier_count.csv')
    # corr_matrix(data, 'correlation_with_bodyfat')
    # visualize_relationships(data)
    # kde_plots(data)
    qq_plots(data)
