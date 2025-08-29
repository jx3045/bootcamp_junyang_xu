import streamlit as st
import pickle
import numpy as np

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Model Prediction Dashboard")

x1 = st.number_input("Feature 1", value=0.0)
x2 = st.number_input("Feature 2", value=0.0)

if st.button("Predict"):
    pred = model.predict(np.array([[x1, x2]])).tolist()
    st.write("Prediction:", pred)
