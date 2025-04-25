pip install scikit-learn

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

# ---- Load trained model and scaler ----
mlp = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")  # StandardScaler fitted on training data

# ---- Function to preprocess and predict ----
def predict_energy(input_dict):
    input_vector = np.zeros(100)  # Assuming atomic numbers from 1 to 100
    for atomic_num, count in input_dict.items():
        input_vector[int(atomic_num) - 1] = count
    input_scaled = scaler.transform([input_vector])
    prediction = mlp.predict(input_scaled)[0]
    return prediction

# ---- Streamlit UI ----
st.title("🔬 Catalyst Predictor from Relaxed Energy")
st.markdown("Enter a dictionary of atomic numbers and their counts.")

# Input field
user_input = st.text_input("Example: {1: 2, 6: 1, 8: 3, 13: 1, 29: 1}", "{}")

if st.button("Predict Relaxed Energy"):
    try:
        input_dict = eval(user_input)
        prediction = predict_energy(input_dict)
        st.write(f"🔮 **Predicted Relaxed Energy:** `{prediction:.4f}` eV")

        if prediction < -700:
            st.success("✅ This compound is likely a **good catalyst**.")
        else:
            st.warning("⚠️ This compound is **less likely** to be a good catalyst.")
    except Exception as e:
        st.error(f"Invalid input: {e}")
