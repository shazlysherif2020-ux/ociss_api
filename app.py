import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ===============================
# Load Models and Metadata
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

st.markdown("### Baseline Clinical Variables")

# ===============================
# USER INPUT – BASELINE
# ===============================

user_input = {}

for feature in model1_features:
    if feature in model1_cat:
        user_input[feature] = st.text_input(f"{feature} (categorical)")
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

input_df1 = pd.DataFrame([user_input])

for col in model1_cat:
    if col in input_df1.columns:
        input_df1[col] = input_df1[col].astype(str)

# ===============================
# PREDICT MODEL 1
# ===============================

if st.button("Calculate Baseline Survival"):

    prob1 = model1.predict_proba(input_df1)[0][1]
    st.success(f"Baseline 5-Year Survival Probability: {prob1*100:.2f}%")

    # ===============================
    # TREATMENT SECTION
    # ===============================

    st.markdown("### Add Treatment Variables")

    treatment_input = {}

    for feature in model2_features:
        if feature == "Model1_Prob":
            continue

        if feature in model2_cat:
            treatment_input[feature] = st.text_input(f"{feature} (categorical)")
        else:
            treatment_input[feature] = st.number_input(f"{feature}", value=0.0)

    input_df2 = pd.DataFrame([treatment_input])

    input_df2["Model1_Prob"] = prob1

    for col in model2_cat:
        if col in input_df2.columns:
            input_df2[col] = input_df2[col].astype(str)

    prob2 = model2.predict_proba(input_df2)[0][1]

    st.success(f"Treatment-Adjusted 5-Year Survival: {prob2*100:.2f}%")

    delta = prob2 - prob1
    st.info(f"Absolute Change in Survival: {delta*100:.2f}%")
