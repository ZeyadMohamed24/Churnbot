import streamlit as st
import requests

st.title("Churn Prediction and Chat")

user_text = st.text_area("Enter your message")


def get_response(text):
    response = requests.post("http://localhost:8000/chat", json={"text": text})
    return response.json()


if st.button("Send"):
    if user_text:
        result = get_response(user_text)
        st.write(f"Response: {result['response']}")
    else:
        st.write("Please enter a message.")
