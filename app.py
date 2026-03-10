import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.markdown(
    """
    <style>
    body{
         background-color: #f0f2f6;
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 10px;
        border-radius: 10px;
    }
    .stNumberInput>div>div>input {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Diabetes Prediction App")

preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    data = scaler.transform(data)
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.error("Person is likely to have Diabetes")
    else:
        st.success("Person is Not Diabetic")
