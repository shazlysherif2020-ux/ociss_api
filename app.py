import streamlit as st
import catboost as cb
import pandas as pd

st.set_page_config(page_title="OCISS Survival Calculator")

st.title("OCISS – 5-Year Survival Prediction")
st.write("Clinical Decision Support Tool for Ovarian Cancer")

st.warning("This tool is intended for clinical decision support and does not replace professional medical judgment.")

# Load model
model = cb.CatBoostClassifier()
model.load_model("ociss_model.cbm")

# ---------------- INPUTS ---------------- #

age = st.number_input("Age", min_value=18, max_value=100, value=60)
stage = st.selectbox("FIGO Stage", [1,2,3,4])
grade = st.selectbox("Tumor Grade", [1,2,3])
residual = st.selectbox("Residual Disease", ["No", "Yes"])

residual_value = 1 if residual == "Yes" else 0

# ---------------- CALCULATION ---------------- #

if st.button("Calculate 5-Year Survival Probability"):

    input_data = pd.DataFrame([{
        "age": age,
        "stage": stage,
        "grade": grade,
        "residual_disease": residual_value
    }])

    probability = model.predict_proba(input_data)[0][1]

    st.success(f"Estimated 5-Year Survival Probability: {probability*100:.2f}%")

    if probability > 0.7:
        st.info("Risk Category: Favorable")
    elif probability > 0.4:
        st.warning("Risk Category: Intermediate")
    else:
        st.error("Risk Category: High Risk")
