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
# Feature order (EXACTLY as trained)
# -----------------------------
FEATURES = [
    'Ethnicity','Smoking','FH_breast','FH_ovarian','FH_uterine',
    'FH_colon','FH_Lynch','FH_BRCA','Menopause','PMH',
    'PH_breast','PH_uterine','PH_colon','PH_cancers_others',
    'Symptom','Site','Tubes','Uterus','Vagina',
    'Cytology','Pelvic_peritoneum','Douglas','Vesicouterine',
    'Broad_ligament','Recto_sigmoid','Other_bowel','Bladder','Pelvic_NS',
    'abdominal_invasion','Ascites','Pleura','Pleura_cytology',
    'aortic_ln','iliac_ln','Inguinal_LN',
    'Distant_metastasis','Distant_site_LS','Distant_site_lungP',
    'Distant_site_dia','Distant_site_stomach','Distant_site_retro',
    'BRCA1','BRCA2','MLH','MSH6','PMS2',
    'Histology','Grade','Previous_hysterectomy',
    'Previous_ovarian_surgery','Previous_endometriosis',
    'Abd_surg_others','ECOG',
    'BMI','Parity','Size_cm',
    'aortic_no','aortic_ln_size',
    'iliac_no','iliac_ln_size',
    'CA_125'
]

# -----------------------------
# Numeric variables
# -----------------------------
NUMERIC_VARS = [
    'BMI','Parity','Size_cm',
    'aortic_no','aortic_ln_size',
    'iliac_no','iliac_ln_size',
    'CA_125'
]

CATEGORICAL_VARS = [f for f in FEATURES if f not in NUMERIC_VARS]

# -----------------------------
# Exact category dictionary
# -----------------------------
CATEGORIES = {

    "Ethnicity": ['Cacuasian','Middle Eastern','Others'],
    "Smoking": ['Current smoker','No smoker','ex-smoker'],

    "FH_breast": ['No','Yes=first degree','Yes=second degree'],
    "FH_ovarian": ['No','Yes=first degree','Yes=second degree'],
    "FH_uterine": ['No','Yes=first degree','Yes=second degree'],
    "FH_colon": ['No','Yes=first degree','Yes=second degree'],
    "FH_Lynch": ['No','Other','Yes=first degree'],
    "FH_BRCA": ['No','Yes=first degree'],

    "Menopause": ['postmenopausal','premenopausal'],

    "PMH": ['Chronic renal disease','Hypertension','No','Other','Others','Type 1 DM','Type 2 DM'],

    "PH_breast": ['No','Yes - Currently in remission','Yes - currently treated'],
    "PH_uterine": ['No','Yes - currently treated'],
    "PH_colon": ['No','Other'],
    "PH_cancers_others": ['No','Yes'],

    "Symptom": [
        'AUB','Abdominal discomfort and/or bloating','Abdominal pain',
        'Asthenia and/or weight loss','Constitutional syndrome','DVT',
        'Incidental','Metastasis','Other','abdominal mass',
        'back pain','change of bowel movment','dyspnea','vaginal discharge'
    ],

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

    "abdominal_invasion": [
        'No','omental cake',
        'omental deposits < 2 cm',
        'omental deposits >= 2 cm',
        'peritoneal carcinomatosis'
    ],

    "Ascites": ['Absent','Marked','Mild'],
    "Pleura": ['Marked effusion','Minimal effusion','No effusion'],
    "Pleura_cytology": ['Negative','Not done','Positive'],

    "aortic_ln": ['No','Yes'],
    "iliac_ln": ['No','Yes'],
    "Inguinal_LN": ['No','Yes'],

    "Distant_metastasis": ['No','Yes'],
    "Distant_site_LS": ['No','Yes'],
    "Distant_site_lungP": ['No','Yes'],
    "Distant_site_dia": ['No','Other','Yes'],
    "Distant_site_stomach": ['No','Yes'],
    "Distant_site_retro": ['No','Yes'],

    "BRCA1": ['Negative','positive; germline','positive; somatic','unknown'],
    "BRCA2": ['Negative','Other','positive; germline','positive; somatic','unknown'],
    "MLH": ['Negative','Other','positive; germline','positive; somatic','unknown'],
    "MSH6": ['Negative','Other','positive; germline','unknown'],
    "PMS2": ['Negative','positive; somatic','unknown'],

    "Histology": [
        'Adult Granulosa Cell Tumor','Carcinosarcoma','Dysgerminoma',
        'Endometrioid','HGSOC','Immature Teratomas',
        'Juvenile Granulosa Cell Tumor','LGSOC','Other',
        'Others; specify','Sertoli-Leydig Tumor',
        'Squamous Cell Carcinoma','Undifferentiated',
        'clear cell carcinoma','mucinous carcinoma'
    ],

    "Grade": ['1.0','2.0','3.0'],

    "Previous_hysterectomy": ['No','Yes'],
    "Previous_ovarian_surgery": ['No','Yes'],
    "Previous_endometriosis": ['No','Other'],
    "Abd_surg_others": ['No','Other','Yes'],

    "ECOG": ['ECOG-0','ECOG-1','ECOG-2','ECOG-3','ECOG-4']
}

# -----------------------------
# Build form
# -----------------------------
input_data = {}

st.header("Categorical Variables")

for var in CATEGORICAL_VARS:
    input_data[var] = st.selectbox(var, CATEGORIES[var])

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
    df = df[FEATURES]

    for col in CATEGORICAL_VARS:
        df[col] = df[col].astype(str)

    pool = Pool(df, cat_features=CATEGORICAL_VARS)

    probability = model.predict_proba(pool)[0][1]

    st.success(
        f"Estimated 5-Year Survival Probability: {probability * 100:.2f}%"
    )
