import streamlit as st
import pandas as pd
import joblib


# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .hero {
            background: linear-gradient(63deg, #b80817 0%, #dec198 100%);
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            font-size: 2.5rem;
            margin-bottom: 0.3rem;
        }
        .hero p {
            font-size: 1.05rem;
            opacity: 0.95;
            margin: 0;
        }
        .section-card {
            background: #d3d3d3;
            padding: 0.7rem 1.5rem;
            border-radius: 6px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            margin-bottom: 1.2rem;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.6rem;
            border-left: 4px solid #ff5f6d;
            padding-left: 0.6rem;
        }
        div.stButton > button {
            background: linear-gradient(135deg, #ff5f6d 0%, #ffc371 100%);
            color: white;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 0.7rem 1.5rem;
            border-radius: 10px;
            border: none;
            width: 100%;
        }
        div.stButton > button:hover {
            opacity: 0.9;
            color: white;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 14px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 700;
            margin-top: 1rem;
        }
        .risk-high {
            background-color: #ffe3e3;
            color: #c92a2a;
            border: 2px solid #ff8787;
        }
        .risk-low {
            background-color: #e6fcf5;
            color: #087f5b;
            border: 2px solid #63e6be;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load saved model, scaler, and expected columns

@st.cache_resource
def load_artifacts():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()


# Hero header

st.markdown(
    """
    <div class="hero">
        <h1>Heart Disease Risk Predictor</h1>
        <p>by Rimee &nbsp;|&nbsp; Enter your health details below to estimate your risk of heart disease.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Sidebar — about / info

with st.sidebar:
    st.header("ℹ️ About this tool")
    st.write(
        "This app uses a **K-Nearest Neighbors (KNN)** model trained on "
        "clinical heart health indicators to estimate heart disease risk."
    )
    st.write(
        "⚠️ **Disclaimer:** This is an educational demo, not a medical "
        "diagnostic tool. Always consult a qualified doctor for medical advice."
    )
    st.markdown("---")
    st.caption("Model: KNN Classifier")
    st.caption("Inputs are scaled before prediction.")


# Input form — grouped into logical sections

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧍 Demographics</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 40)
with col2:
    sex = st.selectbox("Sex", ["M", "F"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🩺 Vitals & Labs</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
with col2:
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
with col3:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💓 Heart & ECG</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
with col2:
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🏃 Exercise Response</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
with col2:
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st.markdown("</div>", unsafe_allow_html=True)


# Predict
predict_clicked = st.button("🔍 Predict My Risk")

if predict_clicked:
    # Create a raw input dictionary
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Try to get a probability/confidence score, if the model supports it
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(scaled_input)[0]
            confidence = proba[int(prediction)] * 100
        except Exception:
            confidence = None

    # Show result
    if prediction == 1:
        conf_text = f" (confidence: {confidence:.1f}%)" if confidence is not None else ""
        st.markdown(
            f'<div class="result-card risk-high">⚠️ High Risk of Heart Disease{conf_text}</div>',
            unsafe_allow_html=True,
        )
        st.progress(int(confidence) if confidence is not None else 70)
        st.caption("Consider consulting a cardiologist for a thorough evaluation.")
    else:
        conf_text = f" (confidence: {confidence:.1f}%)" if confidence is not None else ""
        st.markdown(
            f'<div class="result-card risk-low">✅ Low Risk of Heart Disease{conf_text}</div>',
            unsafe_allow_html=True,
        )
        st.progress(int(confidence) if confidence is not None else 80)
        st.caption("Keep up the healthy habits — regular checkups are still a good idea.")