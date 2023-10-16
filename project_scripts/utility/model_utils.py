import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from project_scripts.utility.path_utils import get_path_from_root


def generate_regression_formula(model, feature_names):
    coefficients = model.coef_
    intercept = model.intercept_
    terms = [f"{intercept:.4f}"]

    for coef, name in zip(coefficients, feature_names):
        terms.append(f"({coef:.4f} * {name})")

    formula = "y = " + " + ".join(terms)

    return formula


def compute_edf(X, alpha):
    """
    Compute effective degrees of freedom for Ridge regression.
    """
    # Convert X to numpy array for matrix operations
    X = np.array(X)

    # Identity matrix of size p x p where p is number of predictors
    identity_matrix = np.identity(X.shape[1])

    # Compute trace of the hat matrix
    df_lambda = np.trace(X @ np.linalg.inv(X.T @ X + alpha * identity_matrix) @ X.T)
    return df_lambda


def compute_aic_bic(model, X, y):
    if isinstance(model, Ridge):
        edf = compute_edf(X, model.alpha)
    else:
        edf = X.shape[1] + 1    # Number of predictors + 1 for intercept

    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    n = len(y)
    log_likelihood = -n/2 * (np.log(2 * np.pi * mse) + 1)
    aic = 2 * edf - 2 * log_likelihood
    bic = np.log(n) * edf - 2 * log_likelihood
    return aic, bic


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


def kfold_evaluation(model, X, y, n_split=10, model_name=""):
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

    return rmse_scores, r2_scores


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
