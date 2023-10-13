import sys
sys.path.append("C:\\Users\\shour\\OneDrive\\Desktop\\Projects\\body-fat-estimation")

import pickle

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from project_scripts.utility.path_utils import get_path_from_root


# Global constants
MODEL_PATH = get_path_from_root("code", "pickle files", "ridge_model_bootstrap.pkl")
SCALER_PATH = get_path_from_root("code", "preprocessing", "robust_scaler.pkl")
IMPUTER_PATH = get_path_from_root("code", "preprocessing", "knn_imputer.pkl")


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


def predict_bodyfat(model, scaler, imputer, abdomen, bmi, wrist):
    """
        Make a prediction based on the user input.
    """
    df = pd.DataFrame([[abdomen, bmi, wrist]], columns=['ABDOMEN', 'BMI', 'WRIST'])
    scaled_data = scaler.transform(df)
    imputed_data = imputer.transform(scaled_data)

    prediction = model.predict(imputed_data)
    return prediction[0]


def app():
    st.title("BOdy Fat Prediction App")

    st.write("""
    ### Enter the values for Abdomen, BMI, and Wrist to predict the body fat percentage!
    """)

    # Getting user input
    abdomen = st.slider('Abdomen (in inches):', 20.0, 50.0, 30.0)
    bmi = st.slider('BMI:', 10.0, 40.0, 20.0)
    wrist = st.slider('Wrist (in inches):', 5.0, 10.0, 6.0)

    model, scaler, imputer = load_resources()

    try:
        with st.spinner("Predicting..."):
            prediction = predict_bodyfat(model, scaler, imputer, abdomen, bmi, wrist)

        st.write("""
                    ### Predicted Body Fat %:
                    """)
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
