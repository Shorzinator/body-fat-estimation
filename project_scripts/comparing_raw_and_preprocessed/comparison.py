import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, mean_squared_error
import shap
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler

from project_scripts.utility.path_utils import get_path_from_root

# Load Data
raw_data = pd.read_csv(get_path_from_root("data", "raw", "BodyFat.csv"))
preprocessed_data = pd.read_csv(get_path_from_root("data", "preprocessed", "preprocessed_data.csv"))

# Load saved robust scaler
feature_scaler = RobustScaler()
scaled_data = feature_scaler.fit_transform(raw_data.drop('BODYFAT', axis=1))

target_scaler = RobustScaler()  # Using a separate scaler for the target
scaled_target = target_scaler.fit_transform(raw_data['BODYFAT'].values.reshape(-1, 1))

# Split into features and target
X_raw, y_raw = raw_data.drop('BODYFAT', axis=1), raw_data['BODYFAT']
X_preprocessed, y_preprocessed = preprocessed_data.drop('BODYFAT', axis=1), preprocessed_data['BODYFAT']

# Scale the raw data using the saved scalars
X_raw = feature_scaler.transform(X_raw)
y_raw = target_scaler.transform(y_raw.values.reshape(-1, 1)).ravel()

# Split data into training and test sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
X_train_preprocessed, X_test_preprocessed, y_train_preprocessed, y_test_preprocessed = train_test_split(X_preprocessed,
                                                                                                        y_preprocessed,
                                                                                                        test_size=0.2,
                                                                                                        random_state=42)


# Model Training
clf_raw = LinearRegression().fit(X_train_raw, y_train_raw)
clf_preprocessed = LinearRegression().fit(X_train_preprocessed, y_train_preprocessed)

# 1. Learning Curves
train_sizes, train_scores, test_scores = learning_curve(clf_raw, X_raw, y_raw)
plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Raw Data Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Raw Data Cross-validation score')
plt.title('Learning Curves for Raw Data')
plt.legend()
plt.show()
plt.savefig('learning_curve_raw.png')

# For preprocessed data
train_sizes, train_scores, test_scores = learning_curve(clf_preprocessed, X_preprocessed, y_preprocessed)
plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Preprocessed Data Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Preprocessed Data Cross-validation score')
plt.title('Learning Curves for Preprocessed Data')
plt.legend()
plt.show()
plt.savefig('learning_curve_preprocessed.png')

# Predictions using the models
y_pred_raw = clf_raw.predict(X_test_raw)
y_pred_preprocessed = clf_preprocessed.predict(X_test_preprocessed)


# Prediction Error Plot
def prediction_error_plot(y_true, y_pred, title, filename):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor='w', s=100)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red')
    plt.title(title, fontsize=16)
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


prediction_error_plot(y_test_raw, y_pred_raw, 'Prediction Error Plot for Raw Data', 'prediction_error_raw.png')
prediction_error_plot(y_test_preprocessed, y_pred_preprocessed, 'Prediction Error Plot for Preprocessed Data',
                      'prediction_error_preprocessed.png')


# Residual Plots
def plot_residuals(y_true, y_pred, title, filename):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, edgecolor='w', s=100)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


plot_residuals(y_test_raw, y_pred_raw, 'Residual Plot for Raw Data', 'residuals_raw.png')
plot_residuals(y_test_preprocessed, y_pred_preprocessed, 'Residual Plot for Preprocessed Data', 'residuals_preprocessed.png')

# Coefficient Comparison
coefficients_raw = clf_raw.coef_
coefficients_preprocessed = clf_preprocessed.coef_

feature_names = X_raw.columns
coeff_comparison = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient (Raw)': coefficients_raw,
    'Coefficient (Preprocessed)': coefficients_preprocessed
})
print(coeff_comparison)

# Model Metrics Comparison
mse_raw = mean_squared_error(y_test_raw, y_pred_raw)
mse_preprocessed = mean_squared_error(y_test_preprocessed, y_pred_preprocessed)

rmse_raw = np.sqrt(mse_raw)
rmse_preprocessed = np.sqrt(mse_preprocessed)

r2_raw = clf_raw.score(X_test_raw, y_test_raw)
r2_preprocessed = clf_preprocessed.score(X_test_preprocessed, y_test_preprocessed)

metrics_comparison = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'R^2'],
    'Value (Raw)': [mse_raw, rmse_raw, r2_raw],
    'Value (Preprocessed)': [mse_preprocessed, rmse_preprocessed, r2_preprocessed]
})
print(metrics_comparison)

# Continue similar plots for other visualizations...

# Note: Some visualizations require further model configurations or additional functions.
# For instance, SHAP values require a separate SHAP explainer setup,
# and permutation importance requires retraining the model on permuted datasets.
