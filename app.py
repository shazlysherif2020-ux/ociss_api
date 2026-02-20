import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap

# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="OCISS",
    layout="wide",
)

st.title("OCISS: Ovarian Cancer Individualised Scoring System")
st.markdown("Clinical decision-support tool for predicting 5-year overall survival (OS5).")

# ==========================================
# Load Models & Metadata
# ==========================================

model1 = joblib.load("model1.pkl")
model2 = joblib.load("model2.pkl")

with open("metadata.json") as f:
    metadata = json.load(f)

model1_features = metadata["model1_features"]
model1_cat = metadata["model1_categorical"]

model2_features = metadata["model2_features"]
model2_cat = metadata["model2_categorical"]

# ==========================================
# Load Category Dictionary (PASTE YOUR FULL DICTIONARY HERE)
# ==========================================

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


# ==========================================
# Helper: Build Input DataFrame
# ==========================================

def build_input_df(features, categorical_vars):
    user_input = {}

    for feature in features:

        if feature in categorical_vars:
            options = category_options.get(feature, [])
            user_input[feature] = st.selectbox(feature, options)
        else:
            user_input[feature] = st.number_input(feature, value=0.0)

    df = pd.DataFrame([user_input])

    for col in categorical_vars:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


# ==========================================
# BASELINE SECTION (MODEL 1)
# ==========================================

with st.expander("Baseline Clinical & Tumour Variables", expanded=True):

    input_df1 = build_input_df(model1_features, model1_cat)
    input_df1 = input_df1.reindex(columns=model1_features)

    if st.button("Calculate Baseline Survival"):

        prob1 = model1.predict_proba(input_df1)[0][1]
        st.session_state["prob1"] = prob1
        st.session_state["input_df1"] = input_df1


# ==========================================
# DISPLAY BASELINE RESULT
# ==========================================

if "prob1" in st.session_state:

    prob1 = st.session_state["prob1"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Baseline 5-Year Survival",
            value=f"{prob1*100:.2f}%"
        )

    # ==========================================
    # SHAP FOR MODEL 1
    # ==========================================

  with st.expander("Explain Baseline Prediction (SHAP)"):

    explainer1 = shap.TreeExplainer(model1)
    shap_values1 = explainer1.shap_values(st.session_state["input_df1"])

    if isinstance(shap_values1, list):
        shap_values1 = shap_values1[1]

    import matplotlib.pyplot as plt

    fig = plt.figure()

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values1[0],
            base_values=explainer1.expected_value,
            data=st.session_state["input_df1"].iloc[0],
            feature_names=st.session_state["input_df1"].columns
        ),
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)
      
    # ==========================================
    # TREATMENT SECTION (MODEL 2)
    # ==========================================

    st.markdown("---")
    st.header("Treatment Variables")

    input_df2_partial = build_input_df(
        [f for f in model2_features if f != "Model1_Prob"],
        model2_cat
    )

    input_df2_partial["Model1_Prob"] = prob1

    input_df2 = input_df2_partial.reindex(columns=model2_features)

    if st.button("Calculate Treatment-Adjusted Survival"):

        prob2 = model2.predict_proba(input_df2)[0][1]
        st.session_state["prob2"] = prob2
        st.session_state["input_df2"] = input_df2


# ==========================================
# DISPLAY MODEL 2 RESULTS
# ==========================================

if "prob2" in st.session_state:

    prob1 = st.session_state["prob1"]
    prob2 = st.session_state["prob2"]

    delta = prob2 - prob1

    col1, col2, col3 = st.columns(3)

    col1.metric("Baseline Survival", f"{prob1*100:.2f}%")
    col2.metric("Treatment-Adjusted Survival", f"{prob2*100:.2f}%")
    col3.metric("Absolute Change", f"{delta*100:.2f}%")

    # ==========================================
    # SHAP FOR MODEL 2
    # ==========================================

    with st.expander("Explain Treatment-Adjusted Prediction (SHAP)"):

    explainer2 = shap.TreeExplainer(model2)
    shap_values2 = shap.TreeExplainer(model2).shap_values(st.session_state["input_df2"])

    if isinstance(shap_values2, list):
        shap_values2 = shap_values2[1]

    import matplotlib.pyplot as plt

    fig = plt.figure()

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values2[0],
            base_values=explainer2.expected_value,
            data=st.session_state["input_df2"].iloc[0],
            feature_names=st.session_state["input_df2"].columns
        ),
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)
# ==========================================
# Footer
# ==========================================

st.markdown("---")
st.caption("For research use only. Not intended to replace clinical judgment.")


