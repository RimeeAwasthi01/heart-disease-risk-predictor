# ❤️ Heart Disease Risk Predictor

A simple Streamlit web app that estimates a person's risk of heart disease using a **K-Nearest Neighbors (KNN)** classification model trained on clinical health indicators.

---

## 🔍 Overview

Users enter basic health details — age, blood pressure, cholesterol, ECG results, exercise response, etc. — and the app returns a **High Risk** or **Low Risk** prediction, along with a confidence score.

---

## ✨ Features

- Clean, light-themed UI with inputs grouped in a dedicated container
- Organized sections: Demographics, Vitals & Labs, Heart & ECG, Exercise Response
- Prediction confidence score (when supported by the model)
- Sidebar with model info and a medical disclaimer
- Cached model loading for faster repeat predictions

---

## 🛠️ Tech Stack

- **Python 3**
- **Streamlit** – web app framework
- **scikit-learn** – KNN model
- **pandas** – data handling
- **joblib** – model/scaler serialization

---

## 📂 Project Structure

```
├── app.py     # Main Streamlit app
├── knn_heart_model.pkl      # Trained KNN model
├── heart_scaler.pkl         # Fitted StandardScaler
├── heart_columns.pkl        # Expected feature columns (post encoding)
├── heart.csv                # CSV file
├── requirements.txt
├── gitignore
└── README.md
```

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
    git clone https://github.com/RimeeAwasthi01/heart-disease-risk-predictor.git
    cd heart-disease-risk-predictor
   ```
 

2. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```

3. Make sure `knn_heart_model.pkl`, `heart_scaler.pkl`, and `heart_columns.pkl` are in the same folder as `app.py`.

---

## ▶️ Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`).

---

## 📊 Input Fields

| Field | Description |
|---|---|
| Age | Patient's age in years |
| Sex | M / F |
| Chest Pain Type | ATA, NAP, TA, ASY |
| Resting Blood Pressure | mm Hg |
| Cholesterol | mg/dL |
| Fasting Blood Sugar | 1 if > 120 mg/dL, else 0 |
| Resting ECG | Normal, ST, LVH |
| Max Heart Rate | Beats per minute |
| Exercise-Induced Angina | Y / N |
| Oldpeak | ST depression induced by exercise |
| ST Slope | Up, Flat, Down |

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is **not a certified medical diagnostic tool**. Predictions should not be used as a substitute for professional medical advice. Always consult a qualified healthcare provider for concerns about heart health.

---

## 👩‍💻 Author

Built by **Rimee Awasthi**.
