import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from project_scripts.EDA.EDA import *
from project_scripts.utility.data_loader import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pd.set_option('display.max_columns', None)


def handle_anomalies(data):
    """
    Handle simple and intricate anomalies in the data.

    :param data: Input data
    :return: Data with no impossible values.
    """
    # Simple Anomalies
    data = data[data['BODYFAT'] >= 2]
    data = data[data['AGE'] >= 18]  # Adult age
    data = data[data['AGE'] <= 90]  # Reasonable upper limit for dataset

    # Age and Body Fat
    data = data[~((data['AGE'] <= 25) & (data['BODYFAT'] > 35))]
    data = data[~((data['AGE'] >= 70) & (data['BODYFAT'] < 5))]

    # Anthropometric Measurements
    data = data[data['WRIST'] < data['NECK']]
    data = data[data['THIGH'] > data['FOREARM']]

    # Body Fat and Anthropometric Measurements
    condition_1 = data['BODYFAT'] < 2
    condition_2 = data['HIP'] > 40
    data = data[~(condition_1 & condition_2)]

    return data


def calculate_bmi(data):
    """
    Calculate BMI using weight and height.
    Convert weight from lbs to kg and height from inches to meters for calculation.
    :param data: Input data.
    :return: Data with added BMI column and dropping ADIPOSITY.
    """
    data['BMI'] = (data['ADIPOSITY'] + (data['WEIGHT'] * 0.453592) / ((data['HEIGHT'] * 0.0254) ** 2)) / 2
    data.drop('ADIPOSITY', axis=1, inplace=True)
    return data


def check_vif(data, threshold=15):
    """
    Check Variance Inflation Factor (VIF) and drop features with VIF above the threshold.
    :param data: Input data.
    :param threshold: VIF Threshold.
    :return: Data without high VIF features.
    """

    features = data.columns
    vif_data = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif_df = pd.DataFrame({'Feature': features, 'VIF': vif_data})
    logging.info(f"VIF values: {vif_df}")
    features_to_drop = vif_df[vif_df['VIF'] > threshold]['Feature'].tolist()
    logging.info(f"Features being dropped for having a high VIF: {features_to_drop}")
    data.drop(columns=features_to_drop, inplace=True)
    return data


def scale_data(data):
    """
    Scales the data using standard scaling.
    :param data: Input data
    :return: Scaled input data
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns), scaler


def knn_imputation(data, n_neighbors=5, save_imputer=True):
    """
    Imputes missing values using KNN imputation.

    :param save_imputer: Whether to save imputer pkl file or not.
    :param data: Input data
    :param n_neighbors: Parameter for KNNImputer
    :return: Data with imputed values
    """

    # Retain column names
    column_names = data.columns

    # Scaling since KNNImputer is sensitive to the scale of the data.
    scaled_data, scaler = scale_data(data)

    # Applying the KNNImputer to the data
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(scaled_data)

    if save_imputer:
        joblib.dump(imputer, 'knn_imputer.pkl')

    # Inverting the scaling to ensure subsequent steps aren't applied to unnaturally scaled data
    data_array = scaler.inverse_transform(imputed_data)

    return pd.DataFrame(data_array, columns=column_names)


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


def final_robust_scaling(data, save_scaler=True):
    """
    Apply final scaling to the data

    :param save_scaler: Whether to save the scalar's state
    :param data: Input data
    :return: Robust Scaled data
    """
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    if save_scaler:
        joblib.dump(scaler,  "robust_scaler.pkl")

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


def main_preprocessing(data, use_rfm=True):
    data = data.drop(columns=['IDNO', 'DENSITY'])

    data = handle_anomalies(data)

    data = calculate_bmi(data)

    # Dropping HEIGHT and WEIGHT when using BMI
    data = data.drop(columns=['HEIGHT', 'WEIGHT'])

    data = handle_outliers(data)

    data = knn_imputation(data)

    data = transform_features(data)

    data = final_robust_scaling(data)

    return data


if __name__ == "__main__":
    df = load_data()

    # Preprocessing for the BMI case
    preprocessed_data_bmi = main_preprocessing(df)
    preprocessed_data_bmi.to_csv(os.path.join(get_path_from_root('data', 'preprocessed'), 'preprocessed_data.csv'),
                                 index=False)

    """
    descriptive_statistics(preprocessed_data, 'descriptive_statistics.csv', 'statistics',
                           'pp_descriptive_statistics.csv')
    distribution(preprocessed_data, 'post_processing', 'pp_skewness_values')
    outlier_detection(preprocessed_data, 'post_processing', 'pp_outlier_count')
    corr_matrix(preprocessed_data, 'post_preprocessing', 'statistics', 'pp_correlation_with_bodyfat.csv')
    visualize_relationships(preprocessed_data)
    qq_plots(preprocessed_data, 'post_processing')
    """
