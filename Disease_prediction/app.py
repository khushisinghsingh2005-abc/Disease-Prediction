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
    "disease_model.joblib"
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

FEATURE_ORDER = [
    "Age",
    "BMI",
    "Blood_Pressure_mmHg",
    "Cholesterol_mg_dL",
    "Glucose_mg_dL",
    "Heart_Rate_bpm",
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
    }
}


@st.cache_resource
def load_model():

    if os.path.exists(MODEL_FILE):
        try:
            saved = joblib.load(MODEL_FILE)

            if isinstance(saved, dict) and "model" in saved:
                return saved["model"], saved.get("threshold", 0.5)

            if hasattr(saved, "named_steps"):
                return saved, 0.5

        except Exception:
            pass

    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    if not os.path.exists(DATA_FILE):
        st.error("Dataset file not found.")
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

    target = (
        df["Disease"] == "Disease"
    ).astype(int)

    model.fit(
        df[COLS],
        target
    )

    return model, 0.5


@st.cache_data
def load_baseline():

    if not os.path.exists(DATA_FILE):
        return np.zeros(len(FEATURE_ORDER))

    df = pd.read_excel(
        DATA_FILE,
        sheet_name="Health_Data"
    )

    transformed = model.named_steps[
        "prep"
    ].transform(df[COLS])

    return transformed.mean(axis=0)


def set_preset(name):

    for key, value in PRESETS[name].items():
        st.session_state[key] = value


st.set_page_config(
    page_title="Early Disease Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

model, threshold = load_model()

baseline = load_baseline()

for key, value in PRESETS["mid"].items():
    st.session_state.setdefault(
        key,
        value
    )

st.title(
    "🩺 Early Disease Risk Prediction"
)

st.caption(
    "Machine Learning based educational demo "
    "using a synthetic health dataset."
)

st.warning(
    "⚠️ This application is for educational purposes "
    "only and is not medical advice."
)


with st.sidebar:

    st.header("👤 Patient Details")

    st.caption(
        "Select an example:"
    )

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

    cholesterol = st.slider(
        "Cholesterol (mg/dL)",
        100,
        350,
        key="chol"
    )

    glucose = st.slider(
        "Glucose (mg/dL)",
        60,
        220,
        key="glu"
    )

    heart_rate = st.slider(
        "Heart rate (bpm)",
        40,
        130,
        key="hr"
    )

    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"],
        key="smoke"
    )

    exercise = st.selectbox(
        "Exercise level",
        ["Low", "Moderate", "High"],
        key="ex"
    )


input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Blood_Pressure_mmHg": bp,
    "Cholesterol_mg_dL": cholesterol,
    "Glucose_mg_dL": glucose,
    "Heart_Rate_bpm": heart_rate,
    "Smoking": smoking,
    "Exercise_Level": exercise
}])


probability = float(
    model.predict_proba(
        input_data
    )[0, 1]
)

is_high = probability >= threshold


left, right = st.columns(
    [1, 1.3]
)


with left:

    st.subheader(
        "📊 Prediction"
    )

    st.metric(
        "Estimated Disease Risk",
        f"{probability:.0%}"
    )

    st.progress(
        min(
            max(
                probability,
                0.0
            ),
            1.0
        )
    )

    if is_high:

        st.error(
            "🔴 Higher risk detected"
        )

        st.write(
            "A medical check-up may be appropriate."
        )

    else:

        st.success(
            "🟢 Lower risk detected"
        )

        st.write(
            "Continue maintaining healthy habits "
            "and regular check-ups."
        )


with right:

    st.subheader(
        "📈 What Drives This Result"
    )

    transformed_row = model.named_steps[
        "prep"
    ].transform(
        input_data
    )[0]

    coefficients = model.named_steps[
        "clf"
    ].coef_[0]

    contribution = (
        transformed_row - baseline
    ) * coefficients

    contribution = pd.Series(
        contribution,
        index=[
            LABELS[column]
            for column in FEATURE_ORDER
        ]
    ).sort_values()

    colors = [
        "#2ecc71" if value < 0 else "#e74c3c"
        for value in contribution.values
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        contribution.index,
        contribution.values,
        color=colors,
        edgecolor="black"
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
        "🟢 **Green = lowers estimated risk**  |  "
        "🔴 **Red = raises estimated risk**"
    )


with st.expander(
    "ℹ️ About the Model"
):

    st.write(
        "The application uses Logistic Regression "
        "with numerical standardization and categorical "
        "encoding."
    )

    st.write(
        "The dataset is synthetic, so predictions should "
        "not be used for real medical decisions."
    )