import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title of the app
st.title('Diabetes Prediction App')

# Collect user input
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=85)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=140, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=85)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=30.1)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=0, max_value=120, value=25)

# Prediction button
if st.button('Predict'):
    # Create a DataFrame with the user input
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    # Scale the user input
    user_data_scaled = scaler.transform(user_data)

    # Make the prediction
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)

    # Display the prediction
    st.write(f'Prediction: {"Diabetes" if prediction[0] == 1 else "No Diabetes"}')
    st.write(f'Probability: {prediction_proba[0]}')
