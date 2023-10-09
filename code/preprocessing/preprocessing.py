import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from code.EDA.EDA import *
from code.utility.data_loader import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_bmi(data):
    """
    Calculate BMI using weight and height.
    Convert weight from lbs to kg and height from inches to meters for calculation.
    :param data: Input data.
    :return: Data with added BMI column and dropping ADIPOSITY.
    """
    data['BMI'] = (data['ADIPOSITY'] + (data['WEIGHT'] * 0.453592) / ((data['HEIGHT'] * 0.0254)**2)) / 2
    data.drop('ADIPOSITY', axis=1, inplace=True)
    return data


def check_vif(data, threshold=10):
    """
    Check Variance Inflation Factor (VIF) and drop features with VIF above threshold.
    :param data: Input data.
    :param threshold: VIF Threshold.
    :return: Data without high VIF features.
    """

    features = data.columns
    vif_data = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif_df = pd.DataFrame({'Feature': features, 'VIF': vif_data})
    features_to_drop = vif_df[vif_df['VIF'] > threshold]['Feature'].tolist()
    return data.drop(columns=features_to_drop, inplace=True)


def scale_data(data):
    """
    Scales the data using standard scaling.
    :param data: Input data
    :return: Scaled input data
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns), scaler


def knn_imputation(data, n_neighbors=5):
    """
    Imputes missing values using KNN imputation.

    :param data: Input data
    :param n_neighbors: Parameter for KNNImputer
    :return: Data with imputed values
    """

    # Scaling since KNNImputer is sensitive to the scale of the data.
    scaled_data, scaler = scale_data(data)

    # Applying the KNNImputer to the data
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(scaled_data)

    # Inverting the scaling to ensure subsequent steps aren't applied to unnaturally scaled data
    data = scaler.inverse_transform(imputed_data)

    return pd.DataFrame(data, columns=data.columns)


def mean_imputation(data):
    """
    Imputes missing values using mean imputation.

    :param data: Input data
    :return: Data imputed using simple imputer
    """
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(data)

    return pd.DataFrame(imputed_data, columns=data.columns)


def transform_features(data):
    """
    Apply transformation to features with skewness.

    :param data: Input data.
    :return: Data with transformed features.
    """

    # Check for skewness in each feature
    skewed_feats = data.apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]

    # Apply log transformation
    for feat in skewed_feats.index:
        data[feat] = np.log1p(data[feat])

    return data


def handle_outliers(data, factor=1.5):
    """
    Handle outliers using the IQR range.

    :param data: Input data
    :param factor:  Factor for IQR range
    :return: Data with outliers handled
    """

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # Cap the data
    clipped_data = data.clip(lower_bound, upper_bound, axis=1)

    return clipped_data


def drop_highly_correlated_features(data, threshold=0.9):
    """
    Drop features that have multicollinearity higher than the threshold.

    :param data: Input Data
    :param threshold: Correlation threshold
    :return: Data with less correlated features
    """

    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data.drop(columns=to_drop, inplace=True)

    logging.info()

    return data


def final_robust_scaling(data, robust=True):
    """
    Apply final scaling to the data

    :param robust: Use RobustScale if True, otherwise StandardScaler
    :param data: Input data
    :return: Robust Scaled data
    """
    if robust:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    return pd.DataFrame(scaled_data, columns=data.columns)


def preprocessing_checks(data):
    """
    Run EDA functions to check the integrity of preprocessed data.

    :param data: Preprocessed input data
    :return: None
    """
    print("Running descriptive statistics...")
    descriptive_statistics(data)
    print("Checking data distributions...")
    distribution(data)
    print("Detecting outliers post preprocessing...")
    outlier_detection(data)
    print("Checking correlation matrix...")
    corr_matrix(data)
    print("Visualizing relationships with the target variable...")
    visualize_relationships(data)


if __name__ == "__main__":
    df = load_data()

    # Dropping IDNO
    df.drop(columns=['IDNO'], inplace=True)

    data_1 = handle_outliers(df)
    data_2 = knn_imputation(data_1)
    data_3 = transform_features(data_2)
    data_4 = drop_highly_correlated_features(data_3)
    data_5 = final_robust_scaling(data_4)

    descriptive_statistics(data_5, 'post_preprocessing', 'statistics', 'pp_descriptive_statistics.csv')
    distribution(data_5, 'post_preprocessing', 'statistics', 'pp_skewness_values.csv')
    outlier_detection(data_5, 'post_preprocessing', 'statistics', 'pp_outlier_count.csv')
    corr_matrix(data_5, 'post_preprocessing', 'statistics', 'pp_correlation_with_bodyfat')
    visualize_relationships(data_5)
