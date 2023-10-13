import logging
import os
import pickle
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split

from code.utility.path_utils import get_path_from_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_residuals(model, X, y, model_name=""):
    """
    Plot residuals for regression model.
    """

    predictions = model.predict(X)
    residuals = y - predictions

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predictions, y=residuals, alpha=0.8)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f"Residuals Plot for {model_name}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.savefig(os.path.join(get_path_from_root("results", "modeling", "residual analysis"), f"resPlot_{model_name}"))

    # plt.show()


def evaluate_model(model, X, y, n_split=10, model_name=""):
    """
    Evaluate a model using k-fold cross-validation
    :param model: The model to be evaluated.
    :param X: Input features
    :param y: Outcome
    :param n_split: Number of splits wanting in k-fold
    :param model_name: Name of the model for logging purposes
    :return: Evaluated metrics
    """
    kf = KFold(n_splits=n_split, shuffle=True, random_state=42)

    # Calculating RMSE Scores
    mse_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(mse_scores)

    # Calculating R^2 Scores
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    # Logging the results
    logging.info(f"{model_name} - RMSE: {rmse_scores.mean()} +/- {rmse_scores.std()}")
    logging.info(f"{model_name} - R^2: {r2_scores.mean()} +/- {r2_scores.std()}\n")


"""
def hyperparameter_tuning(model, params, X, y):

    grid_search = GridSearchCV(model, params, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    return best_params
"""


def bootstrap_evaluation(model, X, y, n_iterations=1000, test_size=0.25):
    """
    Evaluate model using bootstrap resampling.
    """
    mse_scores = []
    r2_scores = []

    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = np.mean((predictions - y_test) ** 2)
        mse_scores.append(mse)

        r2 = model.score(X_test, y_test)
        r2_scores.append(r2)

    return mse_scores, r2_scores


def top_predictors(model, features):
    """
    Return top predictors based on the magnitude of their coefficients.
    """
    # Check if the model is an instance of RANSACRegressor
    coefficients = model.coef_
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    return np.array(features)[sorted_indices]


def main(use_bootstrap=True):
    bmi_data = pd.read_csv(get_path_from_root("data", "preprocessed", "preprocessed_data.csv"))
    X, y = bmi_data.drop('BODYFAT', axis=1), bmi_data['BODYFAT']

    splits = 5

    """
    # Linear Regression
    start_time_linreg = time.time()

    lin_reg = LinearRegression()
    if use_bootstrap:
        mse_scores, r2_scores = bootstrap_evaluation(lin_reg, X, y)
        logging.info(f"LinReg - RMSE (Bootstrap): {np.mean(np.sqrt(mse_scores))} +/- {np.std(np.sqrt(mse_scores))}")
        logging.info(f"LinReg - R^2 (Bootstrap): {np.mean(r2_scores)} +/- {np.std(r2_scores)}")
    else:
        evaluate_model(lin_reg, X, y, n_split=splits, model_name="Linear Regression")
        lin_reg.fit(X, y)  # Fit the model to the entire dataset

    plot_residuals(lin_reg, X, y, "Linear Regression")
    logging.info(f"Top Predictors (LinReg): {top_predictors(lin_reg, X.columns)[:3]}")

    end_time_linreg = time.time()
    logging.info(f"Elapsed time for LinReg: {(end_time_linreg - start_time_linreg):.2f} seconds\n")
    """

    # Ridge Regression
    start_time_RidReg = time.time()

    ridge = Ridge(alpha=0.1, positive=False, solver='auto')

    if use_bootstrap:
        mse_scores, r2_scores = bootstrap_evaluation(ridge, X, y)
        logging.info(f"RidReg - RMSE (Bootstrap): {np.mean(np.sqrt(mse_scores))} +/- {np.std(np.sqrt(mse_scores))}")
        logging.info(f"RidReg - R^2 (Bootstrap): {np.mean(r2_scores)} +/- {np.std(r2_scores)}")
        ridge.fit(X, y)
    else:
        evaluate_model(ridge, X, y, n_split=splits, model_name="Ridge Regression")
        ridge.fit(X, y)

    # Plotting Residuals
    plot_residuals(ridge, X, y, "Ridge Regression")

    # Printing top predictors
    logging.info(f"Top Predictors (RidReg): {top_predictors(ridge, X.columns)[:3]}")

    end_time_RidReg = time.time()
    logging.info(f"Elapsed time for RidReg: {(end_time_RidReg - start_time_RidReg):.2f} seconds\n")

    with open(get_path_from_root("code", "models", "ridge_model_kfold.pkl"), "wb") as model_file:
        pickle.dump(ridge, model_file)


if __name__ == "__main__":
    main(use_bootstrap=False)
