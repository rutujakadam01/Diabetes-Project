import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Prediction System")

# -----------------------------
# Show current directory
# -----------------------------
st.write("Current Folder:", os.getcwd())
st.write("Files:", os.listdir())

# -----------------------------
# Load Model
# -----------------------------
try:
    with open("best_model1.pkl", "rb") as f:
        model = pickle.load(f)

    st.success("✅ Model Loaded Successfully")

except Exception as e:
    st.error("❌ Model Loading Failed")
    st.exception(e)
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 100, 25)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])

smoking_history = st.sidebar.selectbox(
    "Smoking History",
    ["never", "former", "not current", "current", "ever"]
)

bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
hba1c = st.sidebar.slider("HbA1c Level", 3.0, 15.0, 5.5)
glucose = st.sidebar.slider("Blood Glucose Level", 50, 300, 100)

# -----------------------------
# Encoding
# -----------------------------
gender_map = {
    "Male": 1,
    "Female": 0
}

smoking_map = {
    "never": 0,
    "former": 1,
    "not current": 2,
    "current": 3,
    "ever": 2
}

gender = gender_map[gender]
smoking_history = smoking_map[smoking_history]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose]
    })

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
    else:
        probability = 0

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ No Diabetes Detected")

    st.write(f"Probability: {probability*100:.2f}%")
