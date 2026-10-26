from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="CABG PMI Risk Calculator",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(15,118,110,0.12), rgba(37,99,235,0.08));
        border: 1px solid rgba(100,116,139,0.18);
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2.05rem;
        line-height: 1.18;
    }
    .hero p {
        margin: 0;
        color: #52606d;
        font-size: 1rem;
    }
    .result-card {
        padding: 1.4rem 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(100,116,139,0.20);
        background: rgba(248,250,252,0.75);
        margin-top: 0.7rem;
    }
    .risk-number {
        font-size: 3.0rem;
        line-height: 1.0;
        font-weight: 750;
        letter-spacing: -0.04em;
        margin: 0.2rem 0 0.45rem 0;
    }
    .risk-label {
        font-size: 1.08rem;
        font-weight: 650;
        margin-bottom: 0.4rem;
    }
    .small-note {
        font-size: 0.86rem;
        color: #64748b;
    }
    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }
    div[data-testid="stSelectbox"] > div {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Model
# -----------------------------
MODEL_PATH = Path(__file__).resolve().parent / "final_model_bundle.joblib"

FEATURE_ORDER = [
    "albumin",
    "neutrophils",
    "ck",
    "optime",
    "ldh",
    "cryoprecipitate",
    "_mchc",
    "copd",
    "asa_grade",
    "oxycodone",
]

DISPLAY_NAMES = {
    "albumin": "Albumin",
    "neutrophils": "Neutrophils",
    "ck": "Creatine kinase",
    "optime": "Operative time",
    "ldh": "Lactate dehydrogenase",
    "cryoprecipitate": "Cryoprecipitate transfusion",
    "_mchc": "Mean corpuscular hemoglobin concentration",
}

# Observed ranges in the development cohort.
REFERENCE_RANGES = {
    "albumin": (26.1, 51.4),
    "neutrophils": (30.5, 90.1),
    "ck": (13.0, 1603.0),
    "optime": (145.0, 525.0),
    "ldh": (86.0, 960.0),
    "cryoprecipitate": (0.0, 840.0),
    "_mchc": (284.0, 370.0),
}

@st.cache_resource
def load_model():
    # The model was trained and serialized with scikit-learn 1.8.0.
    # requirements.txt pins that version for deployment.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(MODEL_PATH)

    required = {"estimator", "features", "oof_threshold", "model_name"}
    missing = required.difference(bundle.keys())
    if missing:
        raise RuntimeError(f"Model bundle is missing required fields: {sorted(missing)}")

    if list(bundle["features"]) != FEATURE_ORDER:
        raise RuntimeError(
            "Feature order in the model bundle does not match the calculator configuration."
        )
    return bundle

try:
    bundle = load_model()
except Exception as exc:
    st.error(
        "The prediction model could not be loaded. "
        "Please confirm that final_model_bundle.joblib is present and "
        "that scikit-learn==1.8.0 is installed."
    )
    st.exception(exc)
    st.stop()

model = bundle["estimator"]
study_cutoff = float(bundle["oof_threshold"])

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Perioperative Myocardial Injury Risk Calculator</h1>
        <p>
            End-of-surgery risk estimation after isolated on-pump coronary artery bypass grafting (CABG)
            using a calibrated research SVM model with preliminary external validation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Model information")
    st.caption("Model version: 2026-10-26 · laboratory-verified outcomes")
    st.write("**Primary endpoint:** peak specimen-mean cTnI ≥70×URL (2.52 μg/L) during postoperative days 1–3.")
    st.write("**Development cohort:** n=1,140, 170 events")
    st.write("**External validation:** n=252, 37 events")
    st.write("**Selection-inclusive mean repeat-level AUROC:** 0.711 (95% CI 0.674–0.746)")
    st.write("**Fixed-predictor averaged out-of-fold AUROC:** 0.743 (95% CI 0.701–0.783; selection optimism possible)")
    st.write("**External AUROC:** 0.718 (95% CI 0.628–0.809)")
    st.write("**External calibration slope:** 1.018 (95% CI 0.603–1.514)")
    st.write("**External observed/expected event ratio:** 0.892")
    st.write(f"**Retained research cutoff:** {study_cutoff:.2%}")
    st.divider()
    st.caption(
        "This calculator is intended for research and risk-stratification purposes only. "
        "It does not diagnose myocardial injury and should not replace clinical judgment."
    )

st.subheader("Patient information")
st.caption("All ten fields are required. The initial values are illustrative, not recorded patient observations; replace them with the actual values. This interface does not accept missing values.")
st.info("CK, LDH and neutrophil percentage were recorded in the preoperative laboratory section of the source records, but their exact sampling times could not be confirmed. Availability before the end of surgery was therefore not verified for every measurement. This limits interpretation as a prospective end-of-surgery tool.")

left, right = st.columns(2, gap="large")

with left:
    albumin = st.number_input(
        "Albumin (g/L)",
        min_value=10.0, max_value=80.0, value=40.0, step=0.1,
        help="Preoperative serum albumin."
    )
    neutrophils = st.number_input(
        "Neutrophils (%)",
        min_value=0.0, max_value=100.0, value=62.6, step=0.1,
        help="Neutrophil percentage recorded in the preoperative laboratory section; exact sampling time was unavailable."
    )
    ck = st.number_input(
        "Creatine kinase, CK (U/L)",
        min_value=0.0, max_value=10000.0, value=60.0, step=1.0,
        help="CK recorded in the preoperative laboratory section; exact sampling time was unavailable."
    )
    optime = st.number_input(
        "Operative time (min)",
        min_value=30, max_value=1000, value=265, step=5,
        help="Time from recorded surgery start to surgery end, in minutes; not anesthesia duration."
    )
    ldh = st.number_input(
        "Lactate dehydrogenase, LDH (U/L)",
        min_value=0.0, max_value=5000.0, value=172.0, step=1.0,
        help="LDH recorded in the preoperative laboratory section; exact sampling time was unavailable."
    )

with right:
    cryoprecipitate = st.number_input(
        "Cryoprecipitate transfusion (mL)",
        min_value=0.0, max_value=3000.0, value=0.0, step=1.0,
        help="Total cryoprecipitate volume in mL; enter 0 if none was administered. Do not enter units or bags. The study-specific historical conversion of 25 mL per unit is not a universal conversion."
    )
    mchc = st.number_input(
        "MCHC (g/L)",
        min_value=200.0, max_value=450.0, value=331.0, step=1.0,
        help="Mean corpuscular hemoglobin concentration."
    )
    copd_label = st.selectbox(
        "Chronic obstructive pulmonary disease (COPD)",
        options=["No", "Yes"],
        index=0,
    )
    asa_label = st.selectbox(
        "ASA physical status",
        options=["ASA III", "ASA IV"],
        index=0,
        help="The locked model uses ASA III = 0 and ASA IV = 1; other grades are not represented."
    )
    oxycodone_label = st.selectbox(
        "Intraoperative oxycodone use",
        options=["No", "Yes"],
        index=0,
    )

copd = 1.0 if copd_label == "Yes" else 0.0
asa_grade = 1.0 if asa_label == "ASA IV" else 0.0
oxycodone = 1.0 if oxycodone_label == "Yes" else 0.0

input_values = {
    "albumin": float(albumin),
    "neutrophils": float(neutrophils),
    "ck": float(ck),
    "optime": float(optime),
    "ldh": float(ldh),
    "cryoprecipitate": float(cryoprecipitate),
    "_mchc": float(mchc),
    "copd": copd,
    "asa_grade": asa_grade,
    "oxycodone": oxycodone,
}

# Detect extrapolation beyond the observed development range.
outside = []
for name, (low, high) in REFERENCE_RANGES.items():
    value = input_values[name]
    if value < low or value > high:
        outside.append(f"{DISPLAY_NAMES[name]} ({value:g}; development range {low:g}–{high:g})")

if outside:
    st.warning(
        "One or more continuous inputs are outside the observed development-cohort range. "
        "The prediction therefore involves extrapolation:\n\n- " + "\n- ".join(outside)
    )

st.divider()

if st.button("Calculate PMI risk", type="primary", use_container_width=True):
    # The final estimator was fitted after imputation. Because the calculator requires
    # complete input for all ten predictors, the raw complete values are passed directly
    # in the exact feature order used by the fitted estimator.
    x = np.array(
        [[
            input_values["albumin"],
            input_values["neutrophils"],
            input_values["ck"],
            input_values["optime"],
            input_values["ldh"],
            input_values["cryoprecipitate"],
            input_values["_mchc"],
            input_values["copd"],
            input_values["asa_grade"],
            input_values["oxycodone"],
        ]],
        dtype=float,
    )

    probability = float(model.predict_proba(x)[0, 1])
    percent = probability * 100.0
    above_cutoff = probability >= study_cutoff

    result_left, result_right = st.columns([1.25, 1.0], gap="large")

    with result_left:
        label = (
            "Above the study-derived risk cutoff"
            if above_cutoff
            else "Below the study-derived risk cutoff"
        )
        st.markdown(
            f"""
            <div class="result-card">
                <div class="small-note">Estimated probability of the study-defined PMI outcome</div>
                <div class="risk-number">{percent:.1f}%</div>
                <div class="risk-label">{label}</div>
                <div class="small-note">
                    The retained research cutoff ({study_cutoff:.2%}) was originally derived using the Youden index
                    in the original development analysis and retained unchanged.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(int(round(percent)), 0), 100))

    with result_right:
        st.markdown("#### Interpretation")
        if above_cutoff:
            st.write(
                "The model places this patient in the higher predicted-risk group according "
                "to the study-derived cutoff. This is a research classification, "
                "not a validated threshold for clinical action or a diagnostic criterion."
            )
        else:
            st.write(
                "The model places this patient in the lower predicted-risk group according "
                "to the study-derived cutoff. A lower model-predicted risk does not exclude "
                "postoperative myocardial injury."
            )

        st.info(
            "The calculator predicts the probability learned for the primary study outcome. "
            "Clinical decisions should incorporate postoperative biomarkers, ECG findings, "
            "hemodynamics, symptoms, and other relevant clinical information."
        )

with st.expander("Predictors used by the model"):
    st.write(
        "Albumin, neutrophil percentage, creatine kinase, operative time, lactate "
        "dehydrogenase, cryoprecipitate transfusion, MCHC, COPD, ASA grade IV, "
        "and intraoperative oxycodone use."
    )

st.caption(
    "Research calculator based on a calibrated SVM model. "
    "Model predictions should not be interpreted as evidence that any individual predictor, "
    "including oxycodone exposure, has a causal treatment effect."
)

with st.expander("Locked-model parameters and calibration coefficients"):
    specification_path = Path(__file__).resolve().parent / "model-specification.json"
    specification_text = specification_path.read_text(encoding="utf-8")
    st.json(json.loads(specification_text))
    st.download_button("Download model specifications", data=specification_text, file_name="model-specification.json", mime="application/json")

with st.expander("Study outcome definition"):
    st.write("Valid repeated cTnI measurements from the same blood sample were averaged first. The highest specimen-level value during postoperative days 1–3 defined the patient peak. PMI was defined as peak cTnI ≥70×URL (2.52 μg/L; study URL 0.036 μg/L). This is a biomarker-defined outcome, not adjudicated myocardial infarction.")
