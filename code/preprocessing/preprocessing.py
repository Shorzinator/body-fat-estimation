import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler


def scale_data(data):
    """
    Scales the data using standard scaling.
    :param data: Input data
    :return: Scaled input data
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)


def knn_imputation(data, n_neighbors=5, scaled=True):
    """
    Imputes missing values using KNN imputation.

    :param scaled: Is the input data scaled or not?
    :param data: Input data
    :param n_neighbors: Parameter for KNNImputer
    :return: Data with imputed values
    """
    # KNNImputer is sensitive to scale
    scaler = StandardScaler()

    if not scaled:
        scaled_data = scaler.fit_transform(data)
        scaled = True

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(data)

    if scaled:
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


if __name__ == "__main__":
    pass
