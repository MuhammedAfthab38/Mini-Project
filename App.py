# -*- coding: utf-8 -*-
"""app.ipynb

import nltk
nltk.download('stopwords')

# app.py
import streamlit as st
import pickle
import string

# -------------------------
# Load saved model and vectorizer
# -------------------------
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------
# Stopwords for preprocessing
# -------------------------
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# -------------------------
# Text cleaning function
# -------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# -------------------------
# Streamlit UI
# -------------------------
st.title("📰 Fake News Detection")
st.write("Enter a news article below to check if it is **Real** or **Fake**.")

# Text input
user_input = st.text_area("Paste your news article here:")

# Prediction button
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocess input
        cleaned_text = clean_text(user_input)

        # Transform with TF-IDF
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]

        # Display result
        if prediction == 1:
            st.success("✅ This news is likely **REAL**.")
        else:
            st.error("❌ This news is likely **FAKE**.")
