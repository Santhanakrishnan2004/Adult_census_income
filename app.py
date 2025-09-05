import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
clf = joblib.load("clf.pkl")

st.set_page_config(page_title="Adult Income Predictor", layout="centered")

st.title("💰 Adult Census Income Prediction")
st.write("Predict whether a person earns **>50K** or **<=50K** per year.")

# Sidebar inputs
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider("Age", 17, 90, 30)
    workclass = st.sidebar.selectbox("Workclass", 
        ['Private','Self-emp-not-inc','Self-emp-inc','Federal-gov',
         'Local-gov','State-gov','Without-pay','Never-worked'])
    education = st.sidebar.selectbox("Education", 
        ['Bachelors','Some-college','11th','HS-grad','Prof-school',
         'Assoc-acdm','Assoc-voc','9th','7th-8th','12th','Masters',
         '1st-4th','10th','Doctorate','5th-6th','Preschool'])
    marital_status = st.sidebar.selectbox("Marital Status", 
        ['Married-civ-spouse','Divorced','Never-married','Separated',
         'Widowed','Married-spouse-absent','Married-AF-spouse'])
    occupation = st.sidebar.selectbox("Occupation", 
        ['Tech-support','Craft-repair','Other-service','Sales','Exec-managerial',
         'Prof-specialty','Handlers-cleaners','Machine-op-inspct','Adm-clerical',
         'Farming-fishing','Transport-moving','Priv-house-serv','Protective-serv','Armed-Forces'])
    relationship = st.sidebar.selectbox("Relationship", 
        ['Wife','Own-child','Husband','Not-in-family','Other-relative','Unmarried'])
    race = st.sidebar.selectbox("Race", 
        ['White','Asian-Pac-Islander','Amer-Indian-Eskimo','Other','Black'])
    sex = st.sidebar.selectbox("Sex", ['Male','Female'])
    capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.sidebar.number_input("Capital Loss", 0, 4356, 0)
    hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
    native_country = st.sidebar.selectbox("Native Country", 
        ['United-States','Mexico','Philippines','Germany','Canada',
         'India','England','Puerto-Rico','Honduras','Jamaica','Others'])

    data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours_per_week': hours_per_week,
        'native_country': native_country
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
if st.button("🔍 Predict Income"):
    prediction = clf.predict(input_df)[0]
    proba = clf.predict_proba(input_df)[0][1]

    if prediction == ">50K":
        st.success(f"✅ This person is predicted to earn >50K/year (probability: {proba:.2f})")
    else:
        st.error(f"❌ This person is predicted to earn <=50K/year (probability: {1-proba:.2f})")
