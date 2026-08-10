"""
Streamlit demo: enter a patient's clinical values, get two predictions
that reuse exactly the same preprocessing and models built in
notebooks/02_classification.ipynb (disease Stage) and
notebooks/01_survival.ipynb (Cox PH survival/risk).
"""

import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Cirrhosis Risk Demo", page_icon="🩺")

FEATURE_COLS = [
    "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
    "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
    "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
]
CATEGORICAL_COLS = ["Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
COX_COLS = ["N_Days", "event", "Age", "Bilirubin", "Albumin", "Copper", "Prothrombin", "Stage", "Drug"]


@st.cache_resource
def load_models():
    df = pd.read_csv("data/cirrhosis.csv")

    # --- classification model (same as notebooks/02_classification.ipynb) ---
    model_df = df[FEATURE_COLS + ["Stage"]].dropna()
    X = pd.get_dummies(model_df[FEATURE_COLS], columns=CATEGORICAL_COLS, drop_first=True)
    y = model_df["Stage"]

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=5,
        class_weight="balanced", random_state=42,
    )
    rf.fit(X, y)

    # --- Cox PH model (same as notebooks/01_survival.ipynb) ---
    df["event"] = (df["Status"] == "D").astype(int)
    randomized = df[df["Drug"].notna()]
    cox_df = randomized[COX_COLS].dropna().copy()
    cox_df["Drug"] = (cox_df["Drug"] == "D-penicillamine").astype(int)

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="N_Days", event_col="event")

    return rf, X.columns, cph


rf_model, rf_columns, cox_model = load_models()

st.title("🩺 Liver Cirrhosis — Risk & Stage Demo")
st.caption(
    "Enter a patient's clinical values to get a predicted disease stage "
    "(Random Forest) and a survival/risk estimate (Cox Proportional Hazards). "
    "Both models are trained on the 418-patient Mayo Clinic PBC dataset — see "
    "`notebooks/` for the full analysis."
)

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & exam")
        age_years = st.number_input("Age (years)", min_value=1, max_value=100, value=50)
        sex = st.selectbox("Sex", ["F", "M"])
        ascites = st.selectbox("Ascites", ["N", "Y"])
        hepatomegaly = st.selectbox("Hepatomegaly", ["N", "Y"])
        spiders = st.selectbox("Spiders", ["N", "Y"])
        edema = st.selectbox("Edema", ["N", "S", "Y"])
        drug = st.selectbox("Treatment", ["Placebo", "D-penicillamine"])

    with col2:
        st.subheader("Lab values")
        bilirubin = st.number_input("Bilirubin (mg/dl)", min_value=0.0, value=1.0, step=0.1)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0.0, value=260.0)
        albumin = st.number_input("Albumin (gm/dl)", min_value=0.0, value=3.5, step=0.1)
        copper = st.number_input("Copper (ug/day)", min_value=0.0, value=60.0)
        alk_phos = st.number_input("Alk Phos (U/liter)", min_value=0.0, value=1200.0)
        sgot = st.number_input("SGOT (U/ml)", min_value=0.0, value=120.0)
        triglycerides = st.number_input("Triglycerides (mg/dl)", min_value=0.0, value=120.0)
        platelets = st.number_input("Platelets (per cubic ml/1000)", min_value=0.0, value=260.0)
        prothrombin = st.number_input("Prothrombin (seconds)", min_value=0.0, value=10.5, step=0.1)

    submitted = st.form_submit_button("Predict")

if submitted:
    raw_input = pd.DataFrame([{
        "Age": age_years * 365.25,
        "Sex": sex, "Ascites": ascites, "Hepatomegaly": hepatomegaly,
        "Spiders": spiders, "Edema": edema,
        "Bilirubin": bilirubin, "Cholesterol": cholesterol, "Albumin": albumin,
        "Copper": copper, "Alk_Phos": alk_phos, "SGOT": sgot,
        "Tryglicerides": triglycerides, "Platelets": platelets,
        "Prothrombin": prothrombin,
    }])

    # classification: encode exactly like training, align to training columns
    X_input = pd.get_dummies(raw_input, columns=CATEGORICAL_COLS, drop_first=True)
    X_input = X_input.reindex(columns=rf_columns, fill_value=0)

    predicted_stage = rf_model.predict(X_input)[0]
    stage_proba = rf_model.predict_proba(X_input)[0]

    st.subheader("Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted disease Stage", int(predicted_stage))
        proba_df = pd.DataFrame({"Stage": rf_model.classes_, "Probability": stage_proba})
        st.bar_chart(proba_df.set_index("Stage"))
        st.caption(
            "Model: Random Forest, 5-fold CV accuracy 48.5% ± 6.8% — treat this "
            "as a rough signal, not a diagnosis (see README for limitations)."
        )

    with col2:
        cox_input = pd.DataFrame([{
            "Age": age_years * 365.25,
            "Bilirubin": bilirubin, "Albumin": albumin, "Copper": copper,
            "Prothrombin": prothrombin, "Stage": predicted_stage,
            "Drug": 1 if drug == "D-penicillamine" else 0,
        }])
        partial_hazard = pd.Series(cox_model.predict_partial_hazard(cox_input)).iloc[0]
        median_survival = pd.Series(cox_model.predict_median(cox_input)).iloc[0]

        st.metric("Relative risk score", f"{partial_hazard:.2f}", help="1.0 = average risk in the cohort")
        if pd.notna(median_survival) and median_survival != float("inf"):
            st.metric("Estimated median survival", f"{median_survival:.0f} days (~{median_survival / 365.25:.1f} years)")
        else:
            st.metric("Estimated median survival", "not reached within follow-up")
            st.caption("Predicted risk is low enough that the model can't pin down a median within the observed data.")
        st.caption(
            "Model: Cox PH (concordance 0.84). Phase 1 found no significant "
            "difference between D-penicillamine and placebo (p = 0.97) — the "
            "treatment toggle is included for completeness, not because it "
            "meaningfully changes the estimate."
        )
