import streamlit as st
import pickle

# Load the trained model and text vectorizer
robot = pickle.load(open("review_robot.pkl", "rb"))
translator = pickle.load(open("word_translator.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="E-Commerce Review Analyzer",
    page_icon="🛍️",
    layout="centered"
)

# Title
st.title("🛍️ E-Commerce Review Analyzer")
st.write(
    "Enter a customer review below and let the AI analyze the sentiment."
)

# Review input
review = st.text_area(
    "Customer Review",
    placeholder="Example: The product quality is excellent and delivery was very fast!"
)

# Analyze button
if st.button("🔍 Analyze Review"):

    if review.strip():

        # Convert text into the format expected by the model
        review_vector = translator.transform([review])

        # Make prediction
        prediction = robot.predict(review_vector)[0]

        st.subheader("🤖 AI Prediction")

        # Display the model's prediction
        st.success(f"Prediction: {prediction}")

    else:
        st.warning("Please enter a customer review first.")
