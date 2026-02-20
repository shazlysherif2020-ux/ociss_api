import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ===============================
# Load Models
# ===============================

model1 = joblib.load("model1.pkl")
model2 = joblib.load("model2.pkl")

with open("metadata.json") as f:
    metadata = json.load(f)

model1_features = metadata["model1_features"]
model1_cat = metadata["model1_categorical"]

model2_features = metadata["model2_features"]
model2_cat = metadata["model2_categorical"]

st.title("5-Year Ovarian Cancer Survival Predictor")

# =====================================================
# DEFINE CATEGORICAL OPTIONS MANUALLY
# (Example – add all your categories here)
# =====================================================

category_options = {
    "Ethnicity": ["Cacuasian", "Middle Eastern", "Others"],
    "Smoking": ["Current smoker", "No smoker", "ex-smoker"],
    "Menopause": ["postmenopausal", "premenopausal"],
    "ECOG": ["ECOG-0", "ECOG-1", "ECOG-2", "ECOG-3", "ECOG-4"],
    # ADD ALL OTHER VARIABLES HERE
}

# =====================================================
# BASELINE INPUT (MODEL 1)
# =====================================================

st.header("Baseline Variables")

baseline_input = {}

for feature in model1_features:

    if feature in model1_cat:
        options = category_options.get(feature, ["No", "Yes"])
        baseline_input[feature] = st.selectbox(feature, options)
    else:
        baseline_input[feature] = st.number_input(feature, value=0.0)

input_df1 = pd.DataFrame([baseline_input])

# Ensure categorical type as string
for col in model1_cat:
    if col in input_df1.columns:
        input_df1[col] = input_df1[col].astype(str)

# Ensure column order
input_df1 = input_df1.reindex(columns=model1_features)

# =====================================================
# PREDICT MODEL 1
# =====================================================

if st.button("Calculate Baseline Survival"):

    prob1 = model1.predict_proba(input_df1)[0][1]

    st.success(f"Baseline 5-Year Survival: {prob1*100:.2f}%")

    # =====================================================
    # MODEL 2 – TREATMENT SECTION
    # =====================================================

    st.header("Treatment Variables")

    treatment_input = {}

    for feature in model2_features:

        if feature == "Model1_Prob":
            continue

        if feature in model2_cat:
            options = category_options.get(feature, ["No", "Yes"])
            treatment_input[feature] = st.selectbox(feature, options)
        else:
            treatment_input[feature] = st.number_input(feature, value=0.0)

    input_df2 = pd.DataFrame([treatment_input])

    input_df2["Model1_Prob"] = prob1

    # Convert categorical to string
    for col in model2_cat:
        if col in input_df2.columns:
            input_df2[col] = input_df2[col].astype(str)

    # VERY IMPORTANT
    input_df2 = input_df2.reindex(columns=model2_features)

    prob2 = model2.predict_proba(input_df2)[0][1]

    st.success(f"Treatment-Adjusted 5-Year Survival: {prob2*100:.2f}%")

    delta = prob2 - prob1
    st.info(f"Absolute Survival Change: {delta*100:.2f}%")
