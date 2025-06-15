# Gender -> 1 Female   0 NMale
# Churn  -> 1 Yes   0 No
# Scaler is exported as Scaler.pkl
# Model is exported as model.pkl
# Order of the X ->  'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("CHURN PREDICTION APP")
st.write("THIS APP PREDICTS WHETHER A CUSTOMER WILL CHURN OR NOT BASED ON THEIR DEMOGRAPHIC INFORMATION.")


st.divider()

st.write("Please enter the value and hit the predict button for getting a prediction :)")

st.divider()

age = st.number_input("Enter Age", min_value=10, max_value=100, value=25)

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=0)

monthlycharges =  st.number_input("Enter Monthly Charges", min_value=30, max_value=150)

gender = st.selectbox("Select Gender", ["Male", "Female"])

st.divider()

predictbuton = st.button("PREDICT!")

st.divider()

if predictbuton:

    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthlycharges]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "YES" if prediction == 1 else "NO"

    st.snow()

    st.write(f"Predicted: {predicted}")

else:
    st.write("Please press the predict button to get the prediction result.")
    




