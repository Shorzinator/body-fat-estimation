import logging
import os
import pickle
import time

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from project_scripts.utility.path_utils import (get_path_from_root)
from project_scripts.utility.model_utils import bootstrap_evaluation, kfold_evaluation, plot_residuals, top_predictors, \
    compute_aic_bic, generate_regression_formula

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(use_bootstrap=True):
    data = pd.read_csv(get_path_from_root("data", "preprocessed", "preprocessed_data.csv"))
    X, y = data.drop('BODYFAT', axis=1), data['BODYFAT']

    splits = 5

    # Ridge Regression
    start_time_RidReg = time.time()

    ridge = Ridge(alpha=0.1, positive=False, solver='auto')

    if use_bootstrap:
        mse_scores, r2_scores = bootstrap_evaluation(ridge, X, y)
        logging.info(f"RidReg - RMSE (Bootstrap): {np.mean(np.sqrt(mse_scores))} +/- {np.std(np.sqrt(mse_scores))}")
        logging.info(f"RidReg - R^2 (Bootstrap): {np.mean(r2_scores)} +/- {np.std(r2_scores)}\n")
        ridge.fit(X, y)
    else:
        rmse_scores, r2_scores = kfold_evaluation(ridge, X, y, n_split=splits, model_name="Ridge Regression")
        # Logging the results
        logging.info(f"RidReg - RMSE (KFold): {rmse_scores.mean()} +/- {rmse_scores.std()}")
        logging.info(f"RidReg - R^2 (KFold): {r2_scores.mean()} +/- {r2_scores.std()}\n")
        ridge.fit(X, y)

    # Plotting Residuals
    plot_residuals(ridge, X, y, "Ridge Regression")

    # Compute AIC, BIC
    # aic, bic = compute_aic_bic(ridge, X, y)
    # logging.info(f"AIC for Ridge Regression: {aic}")
    # logging.info(f"BIC for Ridge Regression: {bic}\n")

    # Printing top predictors
    logging.info(f"Top Predictors (RidReg): {top_predictors(ridge, X.columns)[:3]}\n")

    # Extract regression formula
    # formula = generate_regression_formula(ridge, X.columns)
    # logging.info(f"Regression formula: {formula}\n")

    end_time_RidReg = time.time()
    logging.info(f"Elapsed time for RidReg: {(end_time_RidReg - start_time_RidReg):.2f} seconds\n")

    with open(get_path_from_root("code", "models", "ridge_model_kfold.pkl"), "wb") as model_file:
        pickle.dump(ridge, model_file)


if __name__ == "__main__":
    main(use_bootstrap=True)
