import logging

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, Ridge

from project_scripts.utility.model_utils import bootstrap_evaluation, kfold_evaluation
from project_scripts.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dynamic_feature_sequence(data, input_features):
    """
    Return a list of features sorted by their mean absolute correlation with the input features

    :param data: Input data.
    :param input_features: Input features.
    :return: Feature sequence in which predictions would be made.
    """

    correlation = data.corr().abs()
    mean_correlations = correlation.drop(input_features, axis=1).loc[input_features].mean()
    sorted_features = mean_correlations.sort_values(ascending=False).index.tolist()
    return [feature for feature in sorted_features if feature != 'BODYFAT']


def train_and_save_model(X, y, feature_name):
    """
    Train a model for the given feature and save it.
    """
    # Train a Ridge regression model
    # model = Ridge(alpha=0.1, positive=False, solver='auto')
    model = LinearRegression()
    rmse_scores, r2_scores = kfold_evaluation(model, X, y)

    logging.info(f"Predicting {feature_name} using {X.columns.tolist()}")
    logging.info(f"RidReg - RMSE (KFold): {rmse_scores.mean()} +/- {rmse_scores.std()}")
    logging.info(f"RidReg - R^2 (KFold): {r2_scores.mean()} +/- {r2_scores.std()}\n")
    model.fit(X, y)

    # Save the trained model
    with open(f"model_for_{feature_name}.pkl", "wb") as f:
        pickle.dump(model, f)


def main():
    data = pd.read_csv(get_path_from_root("data", "preprocessed", "preprocessed_data.csv"))
    input_features = ["ABDOMEN", "BMI", "WRIST"]

    feature_sequence = get_dynamic_feature_sequence(data, input_features)

    for feature in feature_sequence:
        X = pd.DataFrame(data[input_features])
        y = pd.DataFrame(data[feature])

        train_and_save_model(X, y, feature)

        # Add the current feature to the input features for the next iteration
        input_features.append(feature)


if __name__ == "__main__":
    main()


"""
# Load the data
data = pd.read_csv(get_path_from_root("data", "preprocessed", "preprocessed_data.csv"))

# Features to be estimated
features_to_estimate = ['AGE', 'NECK', 'CHEST', 'HIP', 'THIGH', 'KNEE', 'ANKLE', 'BICEPS', 'FOREARM']

# Input features based on user's input
input_features = ['ABDOMEN', 'BMI', 'WRIST']

for feature in features_to_estimate:
    # Split data
    X = pd.DataFrame(data[input_features])
    y = pd.DataFrame(data[feature])

    # Train a linear regression model
    model = Ridge(alpha=0.1, positive=False, solver='auto')
    rmse_scores, r2_scores = kfold_evaluation(model, X, y)
    logging.info(f"Feature: {feature}")
    logging.info(f"RidReg - RMSE (KFold): {rmse_scores.mean()} +/- {rmse_scores.std()}")
    logging.info(f"RidReg - R^2 (KFold): {r2_scores.mean()} +/- {r2_scores.std()}\n")
    model.fit(X, y)

    # Save the trained model
    with open(f"model_for_{feature}.pkl", "wb") as f:
        pickle.dump(model, f)
"""

