import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import TO_DROP
from project_scripts.eda.EDA import *
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

    # Anthropometric Measurements
    data = data[data['WRIST'] < data['NECK']]

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


def log_vif(data):
    """
    Log Variance Inflation Factor (VIF) for each feature.
    :param data: Input data.
    :return: None
    """

    features = data.columns
    vif_data = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif_df = pd.DataFrame({'Feature': features, 'VIF': vif_data})
    logging.info(f"VIF values: {vif_df}")


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

    # Separate the target variable 'BODYFAT'
    target = data['BODYFAT'].copy()
    data_without_target = data.drop('BODYFAT', axis=1)

    # Retain column names of predictors
    column_names = data_without_target.columns

    # Scaling since KNNImputer is sensitive to the scale of the data.
    scaled_data, scaler = scale_data(data_without_target)

    # Applying the KNNImputer to the data
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(scaled_data)

    joblib.dump(imputer, get_path_from_root("project_scripts", "pickle_files", "knn_imputer.pkl"))
    joblib.dump(scaler, get_path_from_root("project_scripts", "pickle_files", "standard_scaler.pkl"))

    # Inverting the scaling to ensure subsequent steps aren't applied to unnaturally scaled data
    data_array = scaler.inverse_transform(imputed_data)

    # Convert to DataFrame
    imputed_df = pd.DataFrame(data_array, columns=column_names)
    imputed_df = imputed_df.reset_index(drop=True)
    target = target.reset_index(drop=True)

    # Add back the 'BODYFAT' column
    imputed_df['BODYFAT'] = target

    return imputed_df


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
    bodyfat = data['BODYFAT'].copy().values.reshape(-1, 1)  # Reshaping it to 2D array
    data_without_target = data.drop('BODYFAT', axis=1)

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data_without_target)

    target_scaler = RobustScaler()  # Using a separate scaler for the target
    scaled_target = target_scaler.fit_transform(bodyfat)
    if save_scaler:
        joblib.dump(scaler,  get_path_from_root("project_scripts", "pickle_files", "robust_scaler_features.pkl"))
        joblib.dump(target_scaler, get_path_from_root("project_scripts", "pickle_files", "robust_scaler_target.pkl"))

    scaled_df = pd.DataFrame(scaled_data, columns=data_without_target.columns)
    scaled_df_target = pd.DataFrame(scaled_target, columns=['BODYFAT'])
    scaled_df['BODYFAT'] = scaled_df_target
    return scaled_df


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


def check_for_nan(data, step_name):
    """
    Log the number of missing values for each column.

    :param data: Input data.
    :param step_name: The name of the preprocessing step.
    :return: None
    """
    missing_values = data.isnull().sum()
    columns_with_nan = missing_values[missing_values > 0]
    if not columns_with_nan.empty:
        logger.info(f"After {step_name}, columns with NaN values:\n{columns_with_nan}")
    else:
        logger.info(f"After {step_name}, no columns with NaN values.")


def main_preprocessing(data, use_rfm=True):
    # data = data.drop(columns=TO_DROP)
    data = data.drop(columns=TO_DROP)

    check_for_nan(data, 'dropping columns')

    data = handle_anomalies(data)
    check_for_nan(data, 'handle_anomalies')

    data = calculate_bmi(data)
    check_for_nan(data, 'calculate_bmi')

    # Dropping HEIGHT and WEIGHT when using BMI
    data = data.drop(columns=['HEIGHT', 'WEIGHT'])
    check_for_nan(data, 'dropping HEIGHT and WEIGHT')

    data = handle_outliers(data)
    check_for_nan(data, 'handle_outliers')

    data = knn_imputation(data)
    check_for_nan(data, 'knn_imputation')

    data = transform_features(data)
    check_for_nan(data, 'transform_features')

    data = final_robust_scaling(data)
    check_for_nan(data, 'final_robust_scaling')

    log_vif(data)

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
