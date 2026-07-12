# app.py

import streamlit as st
import pickle
import numpy as np
import os

print("Current folder:", os.getcwd())
print("Files:", os.listdir())

# =========================
# Load Model
# =========================

model = pickle.load(open("C:\\Users\\krush\\OneDrive\\Documents\\Downloads\\ITR\\diabetes project\\best_model1.pkl", "rb"))
# model = pickle.load(open("final_model.pkl", "rb"))
# =========================
# Page Config-
# =========================
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# =========================
# Title
# =========================
st.title("🩺 Diabetes Prediction System")
st.markdown("### Enter patient details to predict diabetes")

# =========================
# Sidebar Inputs
# =========================
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

# =========================
# Manual Encoding (same as training)
# =========================
gender_map = {"Male": 1, "Female": 0}
smoking_map = {
    'never': 0,
    'former': 1,
    'not current': 2,
    'current': 3,
    'ever': 2
}

gender = gender_map[gender]
smoking_history = smoking_map[smoking_history]

# =========================
# Prediction Button
# =========================
if st.button("🔍 Predict"):

    import pandas as pd

    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],   
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose]
    })

    prediction = model.predict(input_data)[0]
    # prediction = model.predict(input_data)[0]

    # Get probability
    probability = model.predict_proba(input_data)[0][1]
    # =========================
    # Output Section
    # =========================
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ No Diabetes Detected")

    st.write(f"🧠 Probability of Diabetes: {round(probability * 100, 2)}%")




# # =========================
# # IMPORT LIBRARIES
# # =========================
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt

# # =========================
# # PAGE CONFIG
# # =========================
# st.set_page_config(
#     page_title="Diabetes Prediction System",
#     page_icon="🧠",
#     layout="wide"
# )

# # =========================
# # LOAD MODEL
# # =========================
# model = joblib.load("best_model.pkl")   # make sure this file exists

# # =========================
# # TITLE
# # =========================
# st.title("🧠 Diabetes Risk Predictor")
# st.markdown("### Real-time health risk analysis using Machine Learning")

# # =========================
# # SIDEBAR INPUTS
# # =========================
# st.sidebar.header("📝 Patient Details")

# glucose = st.sidebar.slider("Blood Glucose Level", 70, 200, 100)
# hba1c = st.sidebar.slider("HbA1c Level", 3.5, 10.0, 5.5)
# age = st.sidebar.slider("Age", 10, 90, 30)
# bmi = st.sidebar.slider("BMI", 15.0, 40.0, 22.0)

# # Categorical inputs
# gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
# smoking = st.sidebar.selectbox("Smoking History", ["never", "former", "current"])

# # Binary inputs
# hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
# heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
# # =========================
# # CREATE INPUT DATAFRAME
# # =========================
# input_data = pd.DataFrame({
#     "gender": [gender],
#     "age": [age],
#     "hypertension": [hypertension],
#     "heart_disease": [heart_disease],
#     "smoking_history": [smoking],
#     "bmi": [bmi],
#     "HbA1c_level": [hba1c],
#     "blood_glucose_level": [glucose]
# })

# # 🔥 IMPORTANT: INTERACTION FEATURE
# input_data["glucose_hba1c_interaction"] = (
#     input_data["blood_glucose_level"] * input_data["HbA1c_level"]
# )

# # =========================
# # PREDICTION
# # =========================
# prediction = model.predict(input_data)[0]
# probability = model.predict_proba(input_data)[0][1]

# # =========================
# # DISPLAY RESULTS
# # =========================
# st.subheader("📊 Prediction Result")

# col1, col2 = st.columns(2)

# with col1:
#     if prediction == 1:
#         st.error("⚠️ High Risk of Diabetes")
#     else:
#         st.success("✅ Low Risk of Diabetes")

# with col2:
#     st.metric("🧠 Probability", f"{round(probability*100, 2)}%")

# # =========================
# # PROGRESS BAR
# # =========================
# st.progress(int(probability * 100))

# # =========================
# # RISK LEVEL
# # =========================
# if probability > 0.75:
#     st.error("🔴 Critical Risk")
# elif probability > 0.45:
#     st.warning("🟠 Moderate Risk")
# else:
#     st.success("🟢 Safe Zone")

# # =========================
# # INSIGHT TEXT
# # =========================
# st.markdown("### 🧾 Insight")
# st.write(
#     "Higher Blood Glucose and HbA1c levels together significantly increase diabetes risk."
# )

# # =========================
# # FEATURE IMPORTANCE
# # =========================
# st.markdown("### 📈 Feature Importance")

# try:
#     importance = model.feature_importances_
#     features = input_data.columns

#     fig, ax = plt.subplots()
#     ax.barh(features, importance)
#     ax.set_title("Feature Importance")

#     st.pyplot(fig)

# except:
#     st.info("Feature importance not available for this model.")

# # =========================
# # FOOTER
# # =========================
# st.markdown("---")
# st.caption("Built with ❤️ using Streamlit & Machine Learning")
