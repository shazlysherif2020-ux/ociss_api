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
# Define Categories (shortened example)
# =====================================================

category_options = {

"Abd_surg_others": ["No", "Yes"],
"Appendectomy": ["No", "Yes"],
"Ascites": ["Absent", "Marked", "Mild"],
"BRCA1": ["Negative", "positive; germline", "positive; somatic", "unknown"],
"BRCA2": ["Negative", "positive; germline", "positive; somatic", "unknown"],
"Bladder": ["No", "Yes"],
"Bleomycin": ["No", "Yes"],
"Broad_ligament": ["No", "Yes"],
"Carboplatin": ["No", "Yes"],
"Chemo_others": ["No", "Yes"],
"Chemo_type": ["Adjuvant chemotherapy", "Neoadjuvant chemotherapy", "No"],
"Cisplatin": ["No", "Yes"],
"Cytology": ["Negative", "positive"],
"Distant_metastasis": ["No", "Yes"],
"Distant_site_LS": ["No", "Yes"],
"Distant_site_dia": ["No", "Yes"],
"Distant_site_lungP": ["No", "Yes"],
"Distant_site_retro": ["No", "Yes"],
"Distant_site_stomach": ["No", "Yes"],
"Douglas": ["No", "Yes"],
"Doxorubicin": ["No", "Yes"],
"ECOG": ["ECOG-0", "ECOG-1", "ECOG-2", "ECOG-3", "ECOG-4"],
"Ethnicity": ["Cacuasian", "Middle Eastern", "Others"],
"Etoposide": ["No", "Yes"],
"Exentration": ["No", "Total", "anterior", "posterior"],
"FH_BRCA": ["No", "Yes=first degree"],
"FH_breast": ["No", "Yes=first degree", "Yes=second degree"],
"FH_colon": ["No", "Yes=first degree", "Yes=second degree"],
"FH_ovarian": ["No", "Yes=first degree", "Yes=second degree"],
"FH_uterine": ["No", "Yes=first degree", "Yes=second degree"],
"Gemcitabine": ["No", "Yes"],
"Grade": ["1.0", "2.0", "3.0"],
"Histology": [
    "Adult Granulosa Cell Tumor", "Carcinosarcoma", "Dysgerminoma",
    "Endometrioid", "HGSOC", "Immature Teratomas", "LGSOC",
    "Others", "Squamous Cell Carcinoma", "Undifferentiated",
    "clear cell carcinoma", "mucinous carcinoma"
],
"Hysterectomy": ["No hysterectomy", "radical hysterectomy", "simple hysterectomy"],
"Ifosfamide": ["No", "Yes"],
"Inguinal_LN": ["No", "Yes"],
"Menopause": ["postmenopausal", "premenopausal"],
"Mismatch": ["Negative", "positive; germline", "positive; somatic", "unknown"],
"Omentectomy": ["No omentectomy", "infracolic omentectomy", "omental biopsy"],
"Oophororectomy": ["Unilateral SO", "bilateral SO"],
"Other_bowel": ["No", "Yes"],
"PH_breast": ["No", "Yes - Currently in remission", "Yes - currently treated"],
"PH_cancers_others": ["No", "Yes"],
"PH_colon": ["No", "Yes - Currently in remission", "Yes - currently treated"],
"PH_uterine": ["No", "Yes - currently treated"],
"PMH": ["Chronic renal disease", "Hypertension", "No", "Others", "Type 1 DM", "Type 2 DM"],
"Paclitaxel": ["No", "Yes"],
"Pelvic_NS": ["No", "Yes"],
"Pelvic_peritoneum": ["No", "Yes"],
"Pleura": ["Marked effusion", "Minimal effusion", "No effusion"],
"Pleura_cytology": ["Negative", "Not done", "Positive"],
"Previous_hysterectomy": ["No", "Yes"],
"Previous_ovarian_surgery": ["No", "Yes"],
"Recto_sigmoid": ["No", "Yes"],
"Site": ["Bilateral", "Left", "Right"],
"Smoking": ["Current smoker", "No smoker", "ex-smoker"],
"Symptom": [
    "AUB", "Abdominal discomfort and/or bloating", "Abdominal pain",
    "Asthenia and/or weight loss", "Constitutional syndrome", "Incidental",
    "Metastasis", "Others", "abdominal mass",
    "change of bowel movment", "dyspnea", "vaginal discharge"
],
"Target": ["No", "Yes"],
"Tubes": ["No", "Yes"],
"Uterus": ["No", "Yes"],
"Vagina": ["No", "Yes"],
"Vesicouterine": ["No", "Yes"],
"abdominal_invasion": [
    "No", "omental cake",
    "omental deposits < 2 cm",
    "omental deposits >= 2 cm",
    "peritoneal carcinomatosis"
],
"aortic_ln": ["No", "Yes"],
"aortic_lymphadenectomy": ["No", "lymph node sampling", "selective lymphadenectomy", "systematic lymphadenectomy"],
"diaphragmatic_stripping": ["No", "Yes"],
"ebrt": ["No", "Yes"],
"iliac_ln": ["No", "Yes"],
"iliac_lymphadenectomy": ["No", "lymph node sampling", "selective lymphadenectomy", "systematic lymphadenectomy"]

}
# =====================================================
# MODEL 1 SECTION
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

for col in model1_cat:
    if col in input_df1.columns:
        input_df1[col] = input_df1[col].astype(str)

input_df1 = input_df1.reindex(columns=model1_features)

if st.button("Calculate Baseline Survival"):

    prob1 = model1.predict_proba(input_df1)[0][1]
    st.session_state["prob1"] = prob1

# =====================================================
# Display Baseline Result if Exists
# =====================================================

if "prob1" in st.session_state:

    prob1 = st.session_state["prob1"]
    st.success(f"Baseline 5-Year Survival: {prob1*100:.2f}%")

    # =====================================================
    # MODEL 2 SECTION
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

    for col in model2_cat:
        if col in input_df2.columns:
            input_df2[col] = input_df2[col].astype(str)

    input_df2 = input_df2.reindex(columns=model2_features)

    if st.button("Calculate Treatment-Adjusted Survival"):

        prob2 = model2.predict_proba(input_df2)[0][1]

        st.success(f"Treatment-Adjusted 5-Year Survival: {prob2*100:.2f}%")

        delta = prob2 - prob1
        st.info(f"Absolute Survival Change: {delta*100:.2f}%")
