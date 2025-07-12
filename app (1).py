import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('xgboost_model.pkl')

# Page config
st.set_page_config(page_title="Calorie Burn Predictor", page_icon="🔥", layout="centered")

# Custom CSS for style
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
    }
    .stButton>button:hover {
        background-color: #ff7777;
    }
    .stNumberInput>div>input {
        background-color: #ffffff;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #d6336c;'>🔥 Calorie Burn Predictor 🔥</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Get an instant estimate of calories burned from your workout!</p>", unsafe_allow_html=True)
st.markdown("---")

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("🧍 Sex", ["Male", "Female"])
    age = st.slider("🎂 Age", 10, 100, 25)
    height = st.number_input("📏 Height (cm)", 100.0, 250.0, 170.0)
    weight = st.number_input("⚖️ Weight (kg)", 30.0, 200.0, 70.0)

with col2:
    duration = st.number_input("⏱️ Workout Duration (minutes)", 1.0, 300.0, 30.0)
    heart_rate = st.number_input("❤️ Heart Rate (bpm)", 40.0, 220.0, 120.0)
    body_temp = st.number_input("🌡️ Body Temperature (°C)", 35.0, 42.0, 37.0)

# Encoding
sex_encoded = 1 if sex == "Male" else 0

# Predict
if st.button("🔮 Predict Calories Burned"):
    features = np.array([[sex_encoded, age, height, weight, duration, heart_rate, body_temp]])
    prediction = model.predict(features)[0]

    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>🧾 Result</h3>", unsafe_allow_html=True)
    st.success(f"🔥 **Estimated Calories Burned:** {prediction:.2f} kcal")
    st.metric("💡 Tip", "Stay hydrated!", delta="💧")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8em;'>Made with ❤️ using XGBoost and Streamlit</p>", unsafe_allow_html=True)
