import pandas as pd
import pickle
from sklearn.linear_model import Ridge

from project_scripts.modeling.modeling import evaluate_model
from project_scripts.utility.path_utils import get_path_from_root

# Load the data
data = pd.read_csv(get_path_from_root("data", "preprocessed", "preprocessed_data.csv"))

# Features to be estimated
features_to_estimate = ['AGE', 'NECK', 'CHEST', 'HIP', 'THIGH', 'KNEE', 'ANKLE', 'BICEPS', 'FOREARM']

# Input features based on user's input
input_features = ['ABDOMEN', 'BMI', 'WRIST']

for feature in features_to_estimate:
    # Split data
    X = data[input_features]
    y = data[feature]

    # Train a linear regression model
    model = Ridge(alpha=0.1, positive=False, solver='auto')
    evaluate_model(model, X, y, n_split=5, model_name="Ridge Regression")
    model.fit(X, y)

    # Save the trained model
    with open(f"model_for_{feature}.pkl", "wb") as f:
        pickle.dump(model, f)
