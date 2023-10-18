import logging
import os
import pickle
import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from project_scripts.utility.path_utils import (get_path_from_root)
from project_scripts.utility.model_utils import bootstrap_evaluation, kfold_evaluation, plot_residuals, top_predictors, \
    compute_aic_bic, generate_regression_formula

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(use_bootstrap=True):
    # data = pd.read_csv(get_path_from_root("data", "preprocessed", "preprocessed_data.csv"))
    data = pd.read_csv(get_path_from_root("data", "raw", "BodyFat.csv"))

    # Dropping features that had an R^2 value lower than 0.7 when predicting them based on certain other features
    X, y = data.drop(['BODYFAT'], axis=1), data['BODYFAT']

    splits = 5

    # Ridge Regression
    start_time = time.time()

    model = LinearRegression()
    # model = Ridge(alpha=0.1, solver='auto', positive=False)

    if use_bootstrap:
        mse_scores, r2_scores = bootstrap_evaluation(model, X, y)
        logging.info(f"RMSE (Bootstrap): {np.mean(np.sqrt(mse_scores))} +/- {np.std(np.sqrt(mse_scores))}")
        logging.info(f"R^2 (Bootstrap): {np.mean(r2_scores)} +/- {np.std(r2_scores)}\n")
        model.fit(X, y)
    else:
        rmse_scores, r2_scores = kfold_evaluation(model, X, y, n_split=splits, model_name="RidReg")
        # Logging the results
        logging.info(f"RMSE (KFold): {rmse_scores.mean()} +/- {rmse_scores.std()}")
        logging.info(f"R^2 (KFold): {r2_scores.mean()} +/- {r2_scores.std()}\n")
        model.fit(X, y)

    # Plotting Residuals
    plot_residuals(model, X, y, "Linear_Regression")

    # Compute AIC, BIC
    aic, bic = compute_aic_bic(model, X, y)
    logging.info(f"AIC for Regression: {aic}")
    logging.info(f"BIC for Regression: {bic}\n")

    # Printing top predictors
    logging.info(f"Top Predictors: {top_predictors(model, X.columns)[:3]}\n")

    # Extract regression formula
    formula = generate_regression_formula(model, X.columns)
    logging.info(f"Regression formula: {formula}\n")

    end_time = time.time()
    logging.info(f"Elapsed time for LinReg: {(end_time - start_time):.2f} seconds\n")

    with open(get_path_from_root("project_scripts", "pickle_files", "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)


if __name__ == "__main__":
    main(use_bootstrap=True)
