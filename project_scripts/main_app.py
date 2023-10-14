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
        missing_features[feature] = model.predict(input_data)[0]

    return missing_features


def predict_bodyfat(model, scaler, imputer, abdomen, bmi, wrist):
    """
    Make a prediction based on the user input.
    """
    df = pd.DataFrame([[abdomen, bmi, wrist]], columns=['ABDOMEN', 'BMI', 'WRIST'])
    estimated_features = estimate_missing_features(abdomen, bmi, wrist)
    for feature, value in estimated_features.items():
        df[feature] = value

    expected_columns = list(scaler.feature_names_in_)
    df = df[expected_columns]

    scaled_data = scaler.transform(df)
    imputed_data = imputer.transform(scaled_data)

    prediction = model.predict(imputed_data)
    return prediction[0]


def app():
    st.title("Body Fat Prediction App")
    st.write("### Enter the values for Abdomen, BMI, and Wrist to predict the body fat percentage!")

    abdomen = st.slider('Abdomen (in inches):', 20.0, 50.0, 30.0)
    bmi = st.slider('BMI:', 10.0, 40.0, 20.0)
    wrist = st.slider('Wrist (in inches):', 5.0, 10.0, 6.0)

    estimated_features = estimate_missing_features(abdomen, bmi, wrist)
    st.write("Estimated features based on input:")
    st.write(estimated_features)

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
