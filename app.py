import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "Disease_Prediction_Health_Dataset.xlsx"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "best disease_model.pkl"
)

NUM = [
    "Age",
    "BMI",
    "Blood_Pressure_mmHg",
    "Cholesterol_mg_dL",
    "Glucose_mg_dL",
    "Heart_Rate_bpm"
]

COLS = [
    "Age",
    "Gender",
    "BMI",
    "Blood_Pressure_mmHg",
    "Cholesterol_mg_dL",
    "Glucose_mg_dL",
    "Heart_Rate_bpm",
    "Smoking",
    "Exercise_Level"
]

FEATURE_ORDER = NUM + [
    "Gender",
    "Smoking",
    "Exercise_Level"
]

LABELS = {
    "Age": "Age",
    "BMI": "BMI",
    "Blood_Pressure_mmHg": "Blood pressure",
    "Cholesterol_mg_dL": "Cholesterol",
    "Glucose_mg_dL": "Glucose",
    "Heart_Rate_bpm": "Heart rate",
    "Gender": "Gender",
    "Smoking": "Smoking",
    "Exercise_Level": "Exercise level"
}

PRESETS = {
    "high": {
        "age": 55,
        "gender": "Male",
        "bmi": 28.0,
        "bp": 150,
        "chol": 230,
        "glu": 120,
        "hr": 75,
        "smoke": "Yes",
        "ex": "Low"
    },
    "low": {
        "age": 30,
        "gender": "Female",
        "bmi": 22.0,
        "bp": 110,
        "chol": 170,
        "glu": 90,
        "hr": 68,
        "smoke": "No",
        "ex": "High"
    },
    "mid": {
        "age": 50,
        "gender": "Male",
        "bmi": 25.0,
        "bp": 135,
        "chol": 205,
        "glu": 108,
        "hr": 72,
        "smoke": "No",
        "ex": "Moderate"
    }
}


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            saved = joblib.load(MODEL_FILE)

            if isinstance(saved, dict):
                return saved["model"], saved.get("threshold", 0.5)

            return saved, 0.5

        except Exception:
            pass

    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset not found: {DATA_FILE}")
        st.stop()

    df = pd.read_excel(
        DATA_FILE,
        sheet_name="Health_Data"
    )

    prep = ColumnTransformer([
        (
            "num",
            StandardScaler(),
            NUM
        ),
        (
            "bin",
            OrdinalEncoder(
                categories=[
                    ["Female", "Male"],
                    ["No", "Yes"]
                ]
            ),
            ["Gender", "Smoking"]
        ),
        (
            "ord",
            OrdinalEncoder(
                categories=[
                    ["Low", "Moderate", "High"]
                ]
            ),
            ["Exercise_Level"]
        )
    ])

    model = Pipeline([
        (
            "prep",
            prep
        ),
        (
            "clf",
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=2000
            )
        )
    ])

    model.fit(
        df[COLS],
        (df["Disease"] == "Disease").astype(int)
    )

    return model, 0.5


@st.cache_resource
def load_baseline(_model):
    if os.path.exists(DATA_FILE):
        df = pd.read_excel(
            DATA_FILE,
            sheet_name="Health_Data"
        )

        return _model.named_steps["prep"].transform(
            df[COLS]
        ).mean(axis=0)

    return np.zeros(len(FEATURE_ORDER))


def set_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


st.set_page_config(
    page_title="Early Disease Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

model, threshold = load_model()
baseline = load_baseline(model)

for k, v in PRESETS["mid"].items():
    st.session_state.setdefault(k, v)

st.title("🩺 Early Disease Risk Prediction")

st.caption(
    "Logistic Regression model trained on a synthetic "
    "health dataset. Educational demo, not medical advice."
)

with st.sidebar:
    st.header("👤 Patient Details")
    st.caption("Load an example:")

    b1, b2, b3 = st.columns(3)

    b1.button(
        "🔴 High",
        on_click=set_preset,
        args=("high",),
        use_container_width=True
    )

    b2.button(
        "🟡 Borderline",
        on_click=set_preset,
        args=("mid",),
        use_container_width=True
    )

    b3.button(
        "🟢 Low",
        on_click=set_preset,
        args=("low",),
        use_container_width=True
    )

    age = st.slider(
        "Age (years)",
        18,
        80,
        key="age"
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
        key="gender"
    )

    bmi = st.slider(
        "BMI",
        15.0,
        45.0,
        step=0.1,
        key="bmi"
    )

    bp = st.slider(
        "Systolic blood pressure (mmHg)",
        80,
        200,
        key="bp"
    )

    chol = st.slider(
        "Cholesterol (mg/dL)",
        100,
        350,
        key="chol"
    )

    glu = st.slider(
        "Glucose (mg/dL)",
        60,
        220,
        key="glu"
    )

    hr = st.slider(
        "Heart rate (bpm)",
        40,
        130,
        key="hr"
    )

    smoke = st.selectbox(
        "Smoking",
        ["No", "Yes"],
        key="smoke"
    )

    ex = st.selectbox(
        "Exercise level",
        ["Low", "Moderate", "High"],
        key="ex"
    )

row = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Blood_Pressure_mmHg": bp,
    "Cholesterol_mg_dL": chol,
    "Glucose_mg_dL": glu,
    "Heart_Rate_bpm": hr,
    "Smoking": smoke,
    "Exercise_Level": ex
}])

prob = float(
    model.predict_proba(row)[0, 1]
)

is_high = prob >= threshold

left, right = st.columns([1, 1.3])

with left:
    st.subheader("📊 Prediction")

    st.metric(
        "Estimated Risk of Disease",
        f"{prob:.0%}"
    )

    st.progress(
        min(max(prob, 0.0), 1.0)
    )

    if is_high:
        st.error(
            "🔴 Higher risk: a medical check-up is recommended."
        )
    else:
        st.success(
            "🟢 Lower risk: keep up regular check-ups."
        )

    st.caption(
        f"Decision threshold: {threshold:.2f}. "
        "Risk at or above this value is flagged."
    )

with right:
    st.subheader("📈 What Drives This Result")

    transformed_row = model.named_steps[
        "prep"
    ].transform(row)[0]

    coefficients = model.named_steps[
        "clf"
    ].coef_[0]

    contrib = (
        transformed_row - baseline
    ) * coefficients

    contrib = pd.Series(
        contrib,
        index=[
            LABELS[f]
            for f in FEATURE_ORDER
        ]
    ).sort_values()

    colors = [
        "#e74c3c" if value > 0 else "#2ecc71"
        for value in contrib.values
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.barh(
        contrib.index,
        contrib.values,
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_xlabel(
        "Risk contribution"
    )

    ax.set_title(
        "Factors Affecting Disease Risk"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.markdown(
        "🟢 **Green:** lowers estimated risk  |  "
        "🔴 **Red:** raises estimated risk"
    )

    up = (
        contrib[contrib > 0.1]
        .sort_values(ascending=False)
        .head(3)
    )

    if is_high and len(up):
        st.write(
            "🔴 Main risk-raising factors: "
            + ", ".join(up.index)
            + "."
        )

with st.expander("ℹ️ About the Model"):
    st.write(
        "Tuned Logistic Regression (C = 0.1, balanced "
        "class weights) trained on 3,000 synthetic records. "
        "On the test set it detected 85% of people with "
        "disease (recall 0.85) with a ROC-AUC of 0.912. "
        "Blood pressure, age and BMI are the strongest "
        "risk factors."
    )

    st.write(
        "Because the data is synthetic, the results must "
        "not be used for real medical decisions."
    )
