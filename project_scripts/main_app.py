import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import logging

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
MODEL_PATH = gpfr("project_scripts", "pickle_files", "model.pkl")
ROBUST_SCALER_PATH = gpfr("project_scripts", "pickle_files", "robust_scaler_features.pkl")
IMPUTER_PATH = gpfr("project_scripts", "pickle_files", "knn_imputer.pkl")
TARGET_SCALER_PATH = gpfr("project_scripts", "pickle_files", "robust_scaler_target.pkl")
STANDARD_SCALER_PATH = gpfr("project_scripts", "pickle_files", "standard_scaler.pkl")


def load_resources():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ROBUST_SCALER_PATH, "rb") as f:
        robust_scalar_features = joblib.load(f)
    with open(IMPUTER_PATH, "rb") as f:
        imputer = joblib.load(f)
    with open(TARGET_SCALER_PATH, "rb") as f:
        robust_scalar_target = joblib.load(f)
    with open(STANDARD_SCALER_PATH, "rb") as f:
        standard_scalar = joblib.load(f)

    return model, robust_scalar_features, imputer, robust_scalar_target, standard_scalar


def estimate_missing_features(abdomen, bmi, wrist):
    input_data = pd.DataFrame({
        'ABDOMEN': [abdomen],
        'BMI': [bmi],
        'WRIST': [wrist]
    })

    missing_features = {}

    features_to_estimate = ['CHEST', 'HIP', 'NECK', 'THIGH']
    for feature in features_to_estimate:
        model_path = gpfr("project_scripts", "pickle_files", f"model_for_{feature}.pkl")
        model = joblib.load(model_path)

        for missing_feature in features_to_estimate:
            if missing_feature not in input_data.columns:
                input_data[missing_feature] = [0]

        for k, v in missing_features.items():
            input_data[k] = v

        prediction_input = input_data[model.feature_names_in_]
        print("prediction_input:", prediction_input)
        missing_features[feature] = model.predict(prediction_input)[0]
        print("missing_feature:", missing_feature)

    return missing_features


def predict_bodyfat(model, robust_scalar_features, imputer, robust_scalar_target, standard_scalar, abdomen, bmi, wrist):
    logging.basicConfig(level=logging.INFO)

    df = pd.DataFrame([[abdomen, bmi, wrist] + [0] * (len(model.feature_names_in_) - 3)],
                      columns=model.feature_names_in_)

    df = df[model.feature_names_in_]

    estimated_features = estimate_missing_features(abdomen, bmi, wrist)
    for feature, value in estimated_features.items():
        df[feature] = value
    logging.info(f"Estimated features: \n{estimated_features}")

    df = df[model.feature_names_in_]
    logging.info(f"Original Data: \n{df}")

    # Standard Scaling
    scaled_data_standard = standard_scalar.transform(df)
    logging.info(f"After Standard Scaling: \n{scaled_data_standard}")

    # KNN Imputation
    imputed_data = imputer.transform(scaled_data_standard)
    logging.info(f"After KNN Imputation: \n{imputed_data}")

    # Inverse Standard Scaling
    unscaled_data = standard_scalar.inverse_transform(imputed_data)
    logging.info(f"After Inverse Standard Scaling: \n{unscaled_data}")

    # Robust Scaling for Features
    scaled_data_robust = robust_scalar_features.transform(unscaled_data)
    logging.info(f"After Robust Scaling: \n{scaled_data_robust}")

    # Making Predictions
    prediction_scaled = model.predict(scaled_data_robust)
    logging.info(f"Scaled Prediction: {prediction_scaled}")

    # Inverse Scaling using the Target Scaler
    prediction = robust_scalar_target.inverse_transform(prediction_scaled.reshape(-1, 1))
    logging.info(f"Final Prediction after Inverse Target Scaling: {prediction}")

    return prediction[0][0]



def app():
    st.title("Body Fat Prediction App")
    st.write("### Enter the values for Abdomen, BMI, and Wrist to predict the body fat percentage!")

    abdomen = st.number_input('Abdomen (in inches):', min_value=20.0, max_value=50.0, value=30.0, step=0.5)
    bmi = st.number_input('BMI:', min_value=10.0, max_value=40.0, value=20.0, step=0.5)
    wrist = st.number_input('Wrist (in inches):', min_value=5.0, max_value=10.0, value=6.0, step=0.25)

    # estimated_features = estimate_missing_features(abdomen, bmi, wrist)

    model, robust_scalar, imputer, target_scaler, standard_scalar = load_resources()

    try:
        with st.spinner("Predicting..."):
            prediction = predict_bodyfat(model, robust_scalar, imputer, target_scaler, standard_scalar, abdomen, bmi,
                                         wrist)
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
