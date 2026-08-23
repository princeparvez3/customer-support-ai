import streamlit as st
import joblib
from pathlib import Path

# Load trained model
model_path = Path("models/customer_support_model.pkl")
model = joblib.load(model_path)

# Page configuration
st.set_page_config(
    page_title="Customer Support AI",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 Customer Support AI")
st.write("Enter a customer message and the AI will predict its support intent.")

# Input
text = st.text_area(
    "Customer Message",
    placeholder="Example: My card hasn't arrived yet",
    height=120
)

# Prediction
if st.button("Predict Intent"):
    if not text.strip():
        st.warning("Please enter a customer message.")
    else:
        prediction = model.predict([text])[0]

        st.success("Prediction complete!")
        st.subheader("Predicted Intent")
        st.code(prediction)