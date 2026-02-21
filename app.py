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
st.markdown(
    "Clinical decision-support tool for predicting **5-year overall survival (OS5)** and "
    "**cancer-specific survival duration (median months)**."
)

# ==========================================
# Helpers
# ==========================================

def probability_ci(p, n, z=1.96):
    """Approximate Wald CI for a probability. (Same method you used previously.)"""
    se = np.sqrt((p * (1 - p)) / n)
    lower = max(0, p - z * se)
    upper = min(1, p + z * se)
    return lower, upper

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def patient_survival_curve_from_baseline(lp: float, baseline: dict):
    """
    Cox: S(t|x) = S0(t)^(exp(lp))
    baseline dict includes: time, S0, S0_lower, S0_upper
    """
    t = np.array(baseline["time"], dtype=float)
    S0 = np.array(baseline["S0"], dtype=float)
    S0_lo = np.array(baseline["S0_lower"], dtype=float)
    S0_hi = np.array(baseline["S0_upper"], dtype=float)

    power = np.exp(lp)

    S = np.power(S0, power)
    S_lo = np.power(S0_lo, power)
    S_hi = np.power(S0_hi, power)

    return t, S, S_lo, S_hi

def median_time_from_curve(t: np.ndarray, S: np.ndarray):
    """First time where survival <= 0.5; if never, return np.nan."""
    idx = np.where(S <= 0.5)[0]
    if len(idx) == 0:
        return np.nan
    return float(t[idx[0]])

def build_input_df(features, categorical_vars, prefix: str):
    """
    Build a single-row DataFrame from Streamlit inputs.
    prefix is used to avoid Streamlit key collisions between sections.
    """
    user_input = {}

    for feature in features:
        if feature in categorical_vars:
            options = category_options.get(feature, [])
            user_input[feature] = st.selectbox(feature, options, key=f"{prefix}_cat_{feature}")
        else:
            user_input[feature] = st.number_input(feature, value=0.0, key=f"{prefix}_num_{feature}")

    df_out = pd.DataFrame([user_input])

    for col in categorical_vars:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str)

    return df_out

# Replace with your actual OS5 training size (as you used before)
TRAIN_SIZE = 560

# ==========================================
# Load OS5 Models & Metadata (existing files)
# ==========================================

model1 = joblib.load("model1.pkl")
model2 = joblib.load("model2.pkl")

metadata = load_json("metadata.json")
model1_features = metadata["model1_features"]
model1_cat = metadata["model1_categorical"]

model2_features = metadata["model2_features"]
model2_cat = metadata["model2_categorical"]

# ==========================================
# Load Survival (OS_duration / CSS) Models & Metadata (NEW files you will upload)
# ==========================================

# Expect these files to exist in the repo:
# css_model1.pkl, css_model2.pkl, css_metadata.json, css_baseline_m1.json, css_baseline_m2.json
css_model1 = joblib.load("css_model1.pkl")
css_model2 = joblib.load("css_model2.pkl")

css_metadata = load_json("css_metadata.json")
css_m1_features = css_metadata["css_model1_features"]
css_m1_cat = css_metadata["css_model1_categorical"]

css_m2_features = css_metadata["css_model2_features"]           # includes "Model1_Risk"
css_m2_cat = css_metadata["css_model2_categorical"]

css_baseline_m1 = load_json("css_baseline_m1.json")
css_baseline_m2 = load_json("css_baseline_m2.json")

# ==========================================
# Category Dictionary (your curated categories)
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
# BASELINE SECTION (OS5 Model 1 + CSS Model 1)
# ==========================================

with st.expander("Baseline Clinical & Tumour Variables", expanded=True):

    # Build baseline input ONCE (same baseline variables drive OS5 Model1 and CSS Model1)
    input_df1 = build_input_df(model1_features, model1_cat, prefix="baseline")
    input_df1 = input_df1.reindex(columns=model1_features)

    if st.button("Calculate Baseline Survival", key="btn_baseline"):
        # ---- OS5 probability (Model 1) ----
        prob1 = float(model1.predict_proba(input_df1)[0][1])

        # ---- CSS median months (Survival Model 1) ----
        # Build CSS model1 input using its feature list (should match baseline variables)
        css_input1 = input_df1.reindex(columns=css_m1_features)
        lp1 = float(css_model1.predict(css_input1)[0])

        t, S, Slo, Shi = patient_survival_curve_from_baseline(lp1, css_baseline_m1)
        med1 = median_time_from_curve(t, S)
        med1_lo = median_time_from_curve(t, Slo)
        med1_hi = median_time_from_curve(t, Shi)

        st.session_state["prob1"] = prob1
        st.session_state["input_df1"] = input_df1

        st.session_state["css_lp1"] = lp1
        st.session_state["css_med1"] = (med1, med1_lo, med1_hi)


# ==========================================
# DISPLAY BASELINE RESULTS
# ==========================================

if "prob1" in st.session_state and "css_med1" in st.session_state:

    prob1 = st.session_state["prob1"]
    lower1, upper1 = probability_ci(prob1, TRAIN_SIZE)

    med1, med1_lo, med1_hi = st.session_state["css_med1"]

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Baseline 5-Year Survival (OS5 Model 1)", f"{prob1*100:.2f}%")
        st.caption(f"95% CI: {lower1*100:.2f}% – {upper1*100:.2f}%")

    with c2:
        # Convert weeks → months
        if not np.isnan(med1):
            med1_m = med1 / 4.345
            med1_lo_m = med1_lo / 4.345 if not np.isnan(med1_lo) else np.nan
            med1_hi_m = med1_hi / 4.345 if not np.isnan(med1_hi) else np.nan
        
            st.metric(
                "Median Cancer-Specific Survival (CSS Model 1)",
                f"{med1_m:.1f} months"
            )
        
            if not np.isnan(med1_lo_m) and not np.isnan(med1_hi_m):
                st.caption(f"95% CI (approx): {med1_lo_m:.1f} – {med1_hi_m:.1f} months")
            else:
                st.caption("95% CI (approx): not reached within follow-up")
        else:
            st.metric(
                "Median Cancer-Specific Survival (CSS Model 1)",
                "Not reached"
            )
        if (not np.isnan(med1_lo)) and (not np.isnan(med1_hi)):
            st.caption(f"95% CI (approx): {med1_lo:.1f} – {med1_hi:.1f} months")
        else:
            st.caption("95% CI (approx): not reached within follow-up")

    # ==========================================
    # SHAP FOR OS5 MODEL 1
    # ==========================================

    with st.expander("Explain Baseline OS5 Prediction (SHAP)"):

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
# TREATMENT SECTION (single UI used for OS5 Model 2 + CSS Model 2)
# ==========================================

if "prob1" in st.session_state and "css_lp1" in st.session_state:

    st.markdown("---")
    st.header("Treatment Variables")

    prob1 = st.session_state["prob1"]      # OS5 baseline probability
    lp1   = st.session_state["css_lp1"]    # CSS baseline Cox linear predictor

    # --- Build ONE treatment form using the union of treatment variables from both model2 feature lists ---
    os5_treat_features = [f for f in model2_features if f != "Model1_Prob"]
    css_treat_features = [f for f in css_m2_features if f != "Model1_Risk"]

    # preserve order: OS5 list first, then append any missing from CSS list
    treat_union = os5_treat_features + [f for f in css_treat_features if f not in os5_treat_features]

    # determine which of those are categorical (union)
    treat_union_cat = list(dict.fromkeys(model2_cat + css_m2_cat))  # unique, stable order

    # Build single UI DataFrame
    treat_df = build_input_df(
        treat_union,
        treat_union_cat,
        prefix="treatment"
    )

    # --- Assemble OS5 Model 2 input ---
    input_df2 = treat_df.copy()
    input_df2["Model1_Prob"] = prob1
    input_df2 = input_df2.reindex(columns=model2_features)

    # --- Assemble CSS Model 2 input ---
    css_input2 = treat_df.copy()
    css_input2["Model1_Risk"] = lp1
    css_input2 = css_input2.reindex(columns=css_m2_features)

    if st.button("Calculate Treatment-Adjusted Survival", key="btn_treatment"):

        # OS5 probability (Model 2)
        prob2 = float(model2.predict_proba(input_df2)[0][1])

        # CSS median months (Survival Model 2)
        lp2 = float(css_model2.predict(css_input2)[0])
        t2, S2, S2lo, S2hi = patient_survival_curve_from_baseline(lp2, css_baseline_m2)

        med2 = median_time_from_curve(t2, S2)
        med2_lo = median_time_from_curve(t2, S2lo)
        med2_hi = median_time_from_curve(t2, S2hi)

        st.session_state["prob2"] = prob2
        st.session_state["input_df2"] = input_df2

        st.session_state["css_lp2"] = lp2
        st.session_state["css_med2"] = (med2, med2_lo, med2_hi)

# ==========================================
# DISPLAY TREATMENT-ADJUSTED RESULTS
# ==========================================

if "prob2" in st.session_state and "css_med2" in st.session_state:

    prob1 = st.session_state["prob1"]
    prob2 = st.session_state["prob2"]
    lower2, upper2 = probability_ci(prob2, TRAIN_SIZE)

    med1, _, _ = st.session_state["css_med1"]
    med2, med2_lo, med2_hi = st.session_state["css_med2"]

    delta_prob = prob2 - prob1
    delta_med = med2 - med1 if (not np.isnan(med1) and not np.isnan(med2)) else np.nan

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Baseline OS5 (Model 1)", f"{prob1*100:.2f}%")

    with c2:
        st.metric("Treatment OS5 (Model 2)", f"{prob2*100:.2f}%")
        st.caption(f"95% CI: {lower2*100:.2f}% – {upper2*100:.2f}%")

    with c3:
        if not np.isnan(med2):
            med2_m = med2 / 4.345
            med2_lo_m = med2_lo / 4.345 if not np.isnan(med2_lo) else np.nan
            med2_hi_m = med2_hi / 4.345 if not np.isnan(med2_hi) else np.nan
        
            st.metric("Median CSS (Model 2)", f"{med2_m:.1f} months")
        
            if not np.isnan(med2_lo_m) and not np.isnan(med2_hi_m):
                st.caption(f"95% CI (approx): {med2_lo_m:.1f} – {med2_hi_m:.1f} months")
            else:
                st.caption("95% CI (approx): not reached within follow-up")
        else:
            st.metric("Median CSS (Model 2)", "Not reached")

    with c4:
        st.metric("Absolute Change (OS5)", f"{delta_prob*100:.2f}%")
        if not np.isnan(delta_med):
            st.caption(f"Δ Median CSS: {delta_med:.1f} months")
        else:
            st.caption("Δ Median CSS: NA")

    # ==========================================
    # SHAP FOR OS5 MODEL 2
    # ==========================================

    with st.expander("Explain Treatment-Adjusted OS5 Prediction (SHAP)"):

        explainer2 = shap.TreeExplainer(model2)
        shap_values2 = explainer2.shap_values(st.session_state["input_df2"])

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
