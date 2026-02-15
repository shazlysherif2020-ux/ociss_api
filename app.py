import streamlit as st
import catboost as cb
import pandas as pd
import numpy as np

st.set_page_config(page_title="OCISS Survival Calculator")

st.title("OCISS – 5-Year Survival Prediction")
st.warning("Clinical decision support tool. Does not replace professional medical judgment.")

# -------------------------
# Load Model
# -------------------------

model = cb.CatBoostClassifier()
model.load_model("ociss_model.cbm")

# -------------------------
# Exact Category Dictionary
# -------------------------

category_dict = {

"Ethnicity": ['Cacuasian', 'Middle Eastern', 'Others'],

"abdominal_invasion": ['No', 'omental cake', 'omental deposits < 2 cm',
                       'omental deposits >= 2 cm', 'peritoneal carcinomatosis'],

"Grade": ['1.0', '2.0', '3.0'],

"Ascites": ['Absent', 'Marked', 'Mild'],

"BRCA1": ['Negative', 'positive; germline', 'positive; somatic', 'unknown'],

"BRCA2": ['Negative', 'Other', 'positive; germline',
          'positive; somatic', 'unknown'],

"MSH6": ['Negative', 'Other', 'positive; germline', 'unknown'],

"PMS2": ['Negative', 'positive; somatic', 'unknown'],

"MLH": ['Negative', 'Other', 'positive; germline',
        'positive; somatic', 'unknown'],

"MSH2": ['Negative', 'unknown'],

"Cytology": ['Negative', 'positive'],

"PMH": ['Chronic renal disease', 'Hypertension', 'No', 'Other',
        'Others', 'Type 1 DM', 'Type 2 DM'],

"Site": ['Bilateral', 'Left', 'Right'],

"Pelvic_invasions_others": [  # shortened display
    'NO','No','Other','yes'
],

"Abd_surg_others": ['No', 'Other', 'Yes'],

"FH_breast": ['No', 'Yes=first degree', 'Yes=second degree'],

"Histology": [
    'Adult Granulosa Cell Tumor','Carcinosarcoma','Dysgerminoma',
    'Endometrioid','HGSOC','Immature Teratomas',
    'Juvenile Granulosa Cell Tumor','LGSOC','Other',
    'Others; specify','Sertoli-Leydig Tumor',
    'Squamous Cell Carcinoma','Undifferentiated',
    'clear cell carcinoma','mucinous carcinoma'
],

"Symptom": [
    'AUB','Abdominal discomfort and/or bloating','Abdominal pain',
    'Asthenia and/or weight loss','Constitutional syndrome','DVT',
    'Incidental','Metastasis','Other','abdominal mass',
    'change of bowel movment','dyspnea','vaginal discharge'
],

"Pleura": ['Marked effusion','Minimal effusion','No effusion'],

"Distant_metastasis": ['No','Yes'],

"ECOG": ['ECOG-0','ECOG-1','ECOG-2','ECOG-3','ECOG-4'],

"Pleura_cytology": ['Negative','Not done','Positive'],

"Uterus": ['No','Yes'],

"Menopause": ['postmenopausal','premenopausal']
}

numeric_vars = [
    'BMI','Parity','Size_cm',
    'iliac_no','iliac_ln_size',
    'CA_125'
]

# -------------------------
# Input Form
# -------------------------

input_dict = {}

st.header("Categorical Variables")

for var, categories in category_dict.items():
    input_dict[var] = st.selectbox(var, categories)

st.header("Numeric Variables")

for var in numeric_vars:
    value = st.text_input(f"{var} (leave blank if unknown)")
    
    if value.strip() == "":
        input_dict[var] = np.nan
    else:
        try:
            input_dict[var] = float(value)
        except:
            input_dict[var] = np.nan

# -------------------------
# Prediction
# -------------------------

if st.button("Calculate 5-Year Survival Probability"):

    df = pd.DataFrame([input_dict])

    all_features = list(category_dict.keys()) + numeric_vars
    df = df[all_features]

    for col in category_dict.keys():
        df[col] = df[col].astype(str)

    from catboost import Pool

    prediction_pool = Pool(
        df,
        cat_features=list(category_dict.keys())
    )

    probability = model.predict_proba(prediction_pool)[0][1]

    st.success(f"Estimated 5-Year Survival Probability: {probability*100:.2f}%")
