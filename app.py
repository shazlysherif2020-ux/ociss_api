import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="OCISS – 5-Year Survival Prediction", layout="wide")

st.title("OCISS – 5-Year Survival Prediction")
st.warning(
    "This tool is intended for clinical decision support only and does not replace "
    "professional medical judgment."
)

# -----------------------------
# Load model
# -----------------------------
model = CatBoostClassifier()
model.load_model("ociss_model.cbm")

# -----------------------------
# EXACT feature order from model
# -----------------------------
FEATURES = [
    'Ethnicity', 'Smoking', 'FH_breast', 'FH_ovarian', 'FH_uterine',
    'FH_colon', 'FH_Lynch', 'FH_BRCA', 'Menopause', 'PMH',
    'PH_breast', 'PH_uterine', 'PH_colon', 'PH_cancers_others',
    'Symptom', 'Site', 'Tubes', 'Uterus', 'Vagina',
    'Cytology', 'Pelvic_invasions_others', 'abdominal_invasion',
    'Ascites', 'Pleura', 'Pleura_cytology', 'aortic_ln',
    'iliac_ln', 'Inguinal_LN', 'Distant_metastasis', 'Distant_site',
    'BRCA1', 'BRCA2', 'MLH', 'MSH2', 'MSH6', 'PMS2', 'EPCAM',
    'Histology', 'Grade', 'Previous_hysterectomy',
    'Previous_ovarian_surgery', 'Previous_endometriosis',
    'Abd_surg_others', 'ECOG',
    'BMI', 'Parity', 'Size_cm',
    'aortic_no', 'aortic_ln_size',
    'iliac_no', 'iliac_ln_size',
    'CA_125'
]

# -----------------------------
# Categorical vs numeric split
# -----------------------------
NUMERIC_VARS = [
    'BMI', 'Parity', 'Size_cm',
    'aortic_no', 'aortic_ln_size',
    'iliac_no', 'iliac_ln_size',
    'CA_125'
]

CATEGORICAL_VARS = [f for f in FEATURES if f not in NUMERIC_VARS]

# -----------------------------
# Known category dictionaries
# (only where you provided them)
# -----------------------------
KNOWN_CATEGORIES = {
    "Ethnicity": ['Cacuasian', 'Middle Eastern', 'Others'],
    "Menopause": ['postmenopausal', 'premenopausal'],
    "Ascites": ['Absent', 'Marked', 'Mild'],
    "Pleura": ['Marked effusion', 'Minimal effusion', 'No effusion'],
    "Pleura_cytology": ['Negative', 'Not done', 'Positive'],
    "Distant_metastasis": ['No', 'Yes'],
    "Grade": ['1.0', '2.0', '3.0'],
    "ECOG": ['ECOG-0', 'ECOG-1', 'ECOG-2', 'ECOG-3', 'ECOG-4'],
    "Cytology": ['Negative', 'positive'],
    "Site": ['Bilateral', 'Left', 'Right'],
    "Abd_surg_others": ['No', 'Other', 'Yes'],
    "BRCA1": ['Negative', 'positive; germline', 'positive; somatic', 'unknown'],
    "BRCA2": ['Negative', 'Other', 'positive; germline', 'positive; somatic', 'unknown'],
    "MLH": ['Negative', 'Other', 'positive; germline', 'positive; somatic', 'unknown'],
    "MSH2": ['Negative', 'unknown'],
    "MSH6": ['Negative', 'Other', 'positive; germline', 'unknown'],
    "PMS2": ['Negative', 'positive; somatic', 'unknown'],
    "Histology": [
        'Adult Granulosa Cell Tumor', 'Carcinosarcoma', 'Dysgerminoma',
        'Endometrioid', 'HGSOC', 'Immature Teratomas',
        'Juvenile Granulosa Cell Tumor', 'LGSOC', 'Other',
        'Others; specify', 'Sertoli-Leydig Tumor',
        'Squamous Cell Carcinoma', 'Undifferentiated',
        'clear cell carcinoma', 'mucinous carcinoma'
    ]
}

# Default safe categories for unknown categorical variables
DEFAULT_CATEGORIES = ["unknown", "Yes", "No"]

# -----------------------------
# Build input form
# -----------------------------
input_data = {}

st.header("Categorical Variables")

for var in CATEGORICAL_VARS:
    if var in KNOWN_CATEGORIES:
        input_data[var] = st.selectbox(var, KNOWN_CATEGORIES[var])
    else:
        input_data[var] = st.selectbox(var, DEFAULT_CATEGORIES)

st.header("Numeric Variables")

for var in NUMERIC_VARS:
    value = st.text_input(f"{var} (leave blank if unknown)")
    if value.strip() == "":
        input_data[var] = np.nan
    else:
        try:
            input_data[var] = float(value)
        except ValueError:
            input_data[var] = np.nan

# -----------------------------
# Prediction
# -----------------------------
if st.button("Calculate 5-Year Survival Probability"):

    df = pd.DataFrame([input_data])

    # Enforce exact column order
    df = df[FEATURES]

    # Ensure categorical variables are strings
    for col in CATEGORICAL_VARS:
        df[col] = df[col].astype(str)

    # Build CatBoost Pool correctly
    pool = Pool(
        df,
        cat_features=CATEGORICAL_VARS
    )

    probability = model.predict_proba(pool)[0][1]

    st.success(
        f"Estimated 5-Year Survival Probability: {probability * 100:.2f}%"
    )
