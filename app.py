import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Title
st.title("University Admission Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("/content/university_admission_data.csv")

university = load_data()

# Display dataset
if st.checkbox("Show Dataset"):
    st.dataframe(university.head())

# Check for nulls
if st.checkbox("Show Missing Values"):
    st.write(university.isnull().sum())

# Train model
a = university[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
b = university['Admitted']
mo = LinearRegression()
mo.fit(a, b)

# Save the model
with open('mo.pkl', 'wb') as files:
    pickle.dump(mo, files)

st.success("Model trained and saved as 'mo.pkl'")

# User input
st.header("Enter Student Information")

gre = st.number_input("GRE Score", min_value=0, max_value=340, value=320)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
sop = st.slider("SOP Strength", 1.0, 5.0, 3.5)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.5)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)
research = st.selectbox("Research Experience", [0, 1])

# Predict
if st.button("Predict Admission Chance"):
    input_data = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
    prediction = mo.predict(input_data)
    st.write(f"Predicted Admission Chance: {'Admitted' if prediction[0] >= 0.5 else 'Not Admitted'} ({prediction[0]:.2f})")
