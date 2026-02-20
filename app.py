import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# =========================================
# Load Model
# =========================================

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_oc_model.cbm")
    return model

model = load_model()

# =========================================
# Feature Lists
# =========================================

numeric_features = [
    'BMI', 'Parity', 'Size_cm', 'aortic_no', 'aortic_ln_size',
    'iliac_no', 'iliac_ln_size', 'CA_125', 'Age'
]

categorical_features = {
    "Ethnicity": ['Cacuasian', 'Middle Eastern', 'Others'],
    "Smoking": ['Current smoker', 'No smoker', 'ex-smoker'],
    "FH_breast": ['No', 'Yes=first degree', 'Yes=second degree'],
    "FH_ovarian": ['No', 'Yes=first degree', 'Yes=second degree'],
    "FH_uterine": ['No', 'Yes=first degree', 'Yes=second degree'],
    "FH_colon": ['No', 'Yes=first degree', 'Yes=second degree'],
    "FH_BRCA": ['No', 'Yes=first degree'],
    "Menopause": ['postmenopausal', 'premenopausal'],
    "PMH": ['Chronic renal disease', 'Hypertension', 'No', 'Others', 'Type 1 DM', 'Type 2 DM'],
    "PH_breast": ['No', 'Yes - Currently in remission', 'Yes - currently treated'],
    "PH_uterine": ['No', 'Yes - currently treated'],
    "PH_colon": ['No', 'Yes - Currently in remission', 'Yes - currently treated'],
    "PH_cancers_others": ['No', 'Yes'],
    "Symptom": ['AUB','Abdominal discomfort and/or bloating','Abdominal pain',
                'Asthenia and/or weight loss','Constitutional syndrome','Incidental',
                'Metastasis','Others','abdominal mass','change of bowel movment',
                'dyspnea','vaginal discharge'],
    "Site": ['Bilateral','Left','Right'],
    "Tubes": ['No','Yes'],
    "Uterus": ['No','Yes'],
    "Vagina": ['No','Yes'],
    "Cytology": ['Negative','positive'],
    "Pelvic_peritoneum": ['No','Yes'],
    "Douglas": ['No','Yes'],
    "Vesicouterine": ['No','Yes'],
    "Broad_ligament": ['No','Yes'],
    "Recto_sigmoid": ['No','Yes'],
    "Other_bowel": ['No','Yes'],
    "Bladder": ['No','Yes'],
    "Pelvic_NS": ['No','Yes'],
    "abdominal_invasion": ['No','omental cake','omental deposits < 2 cm',
                           'omental deposits >= 2 cm','peritoneal carcinomatosis'],
    "Ascites": ['Absent','Marked','Mild'],
    "Pleura": ['Marked effusion','Minimal effusion','No effusion'],
    "Pleura_cytology": ['Negative','Not done','Positive'],
    "aortic_ln": ['No','Yes'],
    "iliac_ln": ['No','Yes'],
    "Inguinal_LN": ['No','Yes'],
    "Distant_metastasis": ['No','Yes'],
    "Distant_site_LS": ['No','Yes'],
    "Distant_site_lungP": ['No','Yes'],
    "Distant_site_dia": ['No','Yes'],
    "Distant_site_stomach": ['No','Yes'],
    "Distant_site_retro": ['No','Yes'],
    "BRCA1": ['Negative','positive; germline','positive; somatic','unknown'],
    "BRCA2": ['Negative','positive; germline','positive; somatic','unknown'],
    "Mismatch": ['Negative','positive; germline','positive; somatic','unknown'],
    "Histology": ['Adult Granulosa Cell Tumor','Carcinosarcoma','Dysgerminoma',
                  'Endometrioid','HGSOC','Immature Teratomas','LGSOC','Others',
                  'Squamous Cell Carcinoma','Undifferentiated',
                  'clear cell carcinoma','mucinous carcinoma'],
    "Grade": ['1.0','2.0','3.0'],
    "Previous_hysterectomy": ['No','Yes'],
    "Previous_ovarian_surgery": ['No','Yes'],
    "Abd_surg_others": ['No','Yes'],
    "ECOG": ['ECOG-0','ECOG-1','ECOG-2','ECOG-3','ECOG-4']
}

# =========================================
# UI
# =========================================

st.title("Ovarian Cancer 5-Year Outcome Predictor")
st.write("Leave fields blank if unknown.")

input_data = {}

# Numeric inputs
st.header("Numeric Variables")

for feature in numeric_features:
    value = st.text_input(f"{feature} (optional)")
    if value.strip() == "":
        input_data[feature] = np.nan
    else:
        input_data[feature] = float(value)

# Categorical inputs
st.header("Categorical Variables")

for feature, categories in categorical_features.items():
    choice = st.selectbox(
        f"{feature} (optional)",
        options=[""] + categories
    )
    input_data[feature] = choice if choice != "" else np.nan

# =========================================
# Prediction
# =========================================

if st.button("Calculate Probability"):

    df_input = pd.DataFrame([input_data])

    # Ensure correct column order
    model_features = model.feature_names_
    df_input = df_input[model_features]

    # Convert categoricals to string
    for col in categorical_features.keys():
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)

    proba = model.predict_proba(df_input)[0,1]

    st.success(f"Predicted Probability: {round(proba*100,2)}%")

