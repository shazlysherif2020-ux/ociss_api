import streamlit as st
import catboost as cb
import pandas as pd

st.set_page_config(page_title="OCISS Survival Calculator")

st.title("OCISS – 5-Year Survival Prediction")
st.warning("Clinical decision support tool. Does not replace medical judgment.")

# -------------------------
# Feature lists (FINAL)
# -------------------------

categorical_vars = [
    'Ethnicity',
    'abdominal_invasion',
    'Grade',
    'Ascites',
    'BRCA1', 'BRCA2', 'MSH6', 'PMS2','MLH','MSH2',
    'Cytology',
    'PMH',
    'Site',
    'Pelvic_invasions_others',
    'Abd_surg_others',
    'FH_breast',
    'Histology',
    'Symptom',
    'Pleura',
    'Distant_metastasis',
    'ECOG',
    'Pleura_cytology',
    'Uterus',
    'Menopause'
]

numeric_vars = [
    'BMI', 'Parity', 'Size_cm',
    'iliac_no', 'iliac_ln_size',
    'CA_125'
]

# -------------------------
# Load model
# -------------------------

model = cb.CatBoostClassifier()
model.load_model("ociss_model.cbm")

# -------------------------
# Build Input Form
# -------------------------

st.header("Categorical Variables")

input_dict = {}

for var in categorical_vars:
    value = st.text_input(var)   # free text for now (safer than wrong encoding)
    input_dict[var] = value

st.header("Numeric Variables")

for var in numeric_vars:
    value = st.number_input(var, value=0.0)
    input_dict[var] = value

# -------------------------
# Prediction
# -------------------------

if st.button("Calculate 5-Year Survival Probability"):

    # Create DataFrame
    df = pd.DataFrame([input_dict])

    # IMPORTANT: ensure correct column order
    df = df[categorical_vars + numeric_vars]

    # Ensure categorical columns are strings
    for col in categorical_vars:
        df[col] = df[col].astype(str)

    probability = model.predict_proba(df)[0][1]

    st.success(f"Estimated 5-Year Survival Probability: {probability*100:.2f}%")
