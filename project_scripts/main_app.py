import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import joblib


def gpfr(*subdirs):
    """
    Construct a path based on the root directory.

    Args:
    *subdirs (str): List of subdirectories or files, e.g., "data", "mydata.csv"

    Returns:
    str: Full path from the root directory
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, *subdirs)


# Global constants
MODEL_PATH = gpfr("project_scripts", "pickle files", "ridge_model_bootstrap.pkl")
SCALER_PATH = gpfr("project_scripts", "preprocessing", "robust_scaler.pkl")
IMPUTER_PATH = gpfr("project_scripts", "preprocessing", "knn_imputer.pkl")


def load_resources():
    """
    Load all necessary resources such as a model, scaler, and imputer.
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = joblib.load(f)
    with open(IMPUTER_PATH, "rb") as f:
        imputer = joblib.load(f)

    return model, scaler, imputer


def estimate_missing_features(abdomen, bmi, wrist):
    input_data = pd.DataFrame({
        'ABDOMEN': [abdomen],
        'BMI': [bmi],
        'WRIST': [wrist]
    })

    # Dictionary to hold predicted values for missing features
    missing_features = {}

    features_to_estimate = ['AGE', 'NECK', 'CHEST', 'HIP', 'THIGH', 'KNEE', 'ANKLE', 'BICEPS', 'FOREARM']
    for feature in features_to_estimate:
        model_path = gpfr("project_scripts", "for_app", f"model_for_{feature}.pkl")
        model = joblib.load(model_path)

        # Start by setting all the missing features to their mean/median or any default value
        for missing_feature in features_to_estimate:
            if missing_feature not in input_data.columns:
                input_data[missing_feature] = [0]  # setting to 0 as a placeholder; it will be replaced soon

        # Update the already predicted features
        for k, v in missing_features.items():
            input_data[k] = v

        # Ensure the order of columns matches what the model expects
        prediction_input = input_data[model.feature_names_in_]

        # Predict the feature and store its value
        missing_features[feature] = model.predict(prediction_input)[0]

    return missing_features


def predict_bodyfat(model, scaler, imputer, abdomen, bmi, wrist):
    """
    Make a prediction based on the user input.
    """
    # Create a DataFrame with placeholder values for all features
    df = pd.DataFrame([[abdomen, bmi, wrist] + [0] * (len(model.feature_names_in_) - 3)],
                      columns=model.feature_names_in_)

    # Update the DataFrame with actual and estimated feature values
    estimated_features = estimate_missing_features(abdomen, bmi, wrist)
    for feature, value in estimated_features.items():
        df[feature] = value

    # Ensure the order of columns matches what the model expects
    df = df[model.feature_names_in_]

    # Transform and impute the data
    scaled_data = scaler.transform(df)
    imputed_data = imputer.transform(scaled_data)

    # Predict body fat
    prediction = model.predict(imputed_data)
    return prediction[0]


def app():
    st.title("Body Fat Prediction App")
    st.write("### Enter the values for Abdomen, BMI, and Wrist to predict the body fat percentage!")

    abdomen = st.number_input('Abdomen (in inches):', min_value=20.0, max_value=50.0, value=30.0, step=0.5)
    bmi = st.number_input('BMI:', min_value=10.0, max_value=40.0, value=20.0, step=0.5)
    wrist = st.number_input('Wrist (in inches):', min_value=5.0, max_value=10.0, value=6.0, step=0.25)

    estimated_features = estimate_missing_features(abdomen, bmi, wrist)
    # st.write("Estimated features based on input:")
    # st.write(estimated_features)

    model, scaler, imputer = load_resources()

    try:
        with st.spinner("Predicting..."):
            prediction = predict_bodyfat(model, scaler, imputer, abdomen, bmi, wrist)
        st.write("### Predicted Body Fat %:")
        st.write(str(round(prediction, 2)))
    except Exception as e:
        st.write("An error occurred:", e)

    if st.button("About"):
        st.write("""
                 This is a simple app to predict Body Fat % based on measurements of Abdomen, BMI, and Wrist.
                 The model used is Ridge Regression trained on a dataset of body measurements.
                 The features are first transformed using a RobustScaler and missing values are imputed using KNNImputer.
                 """)


if __name__ == "__main__":
    app()
