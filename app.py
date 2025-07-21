import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Predictor")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0)
Parch = st.number_input("Parents/Children Aboard", 0)
Fare = st.number_input("Fare Paid", 0.0)
Embarked = st.selectbox("Embarked", ['S', 'C', 'Q'])

sex_encoded = 1 if Sex == "male" else 0
embarked_map = {'S': 2, 'C': 0, 'Q': 1}
embarked_encoded = embarked_map[Embarked]

input_data = np.array([[Pclass, sex_encoded, Age, SibSp, Parch, Fare, embarked_encoded]])
prediction = model.predict(input_data)

if st.button("Predict"):
    if prediction[0] == 1:
        st.success("🎉 The passenger would have survived.")
    else:
        st.error("💀 The passenger would not have survived.")
