import streamlit as st
import pickle
import re
import numpy as np
import nltk
import pandas as pd
import requests
from io import StringIO
from nltk.corpus import stopwords
from scipy.sparse import hstack

# ------------------------
# NLTK & Setup
# ------------------------
nltk.download("stopwords")
stop_words = stopwords.words("english")

st.set_page_config(page_title="Fake News Detector AI", layout="wide")

# ------------------------
# Text cleaning function
# ------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ------------------------
# Load preprocessing objects
# ------------------------
@st.cache_resource
def load_models():
    with open("model/hybrid_svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/hybrid_tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model/hybrid_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, vectorizer, scaler

model, vectorizer, scaler = load_models()

# ------------------------
# Streamlit UI - Main App
# ------------------------
st.title("🛡️ ADVANCED FAKE NEWS DETECTOR")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyze Headline")
    user_input = st.text_area("Enter news headline here:", placeholder="e.g., Scientist found a way to live forever...")

    if st.button("Check Authenticity"):
        if user_input.strip() == "":
            st.warning("Please enter a headline.")
        else:
            # Preprocessing
            cleaned_text = clean_text(user_input)
            
            # Metadata features (matching the hybrid model training)
            title_length = len(user_input)
            exclamation_count = user_input.count("!")
            tweet_log = np.log1p(10) # Default
            source_encoded = 0 # Default for manual input

            # Combine Numeric Features
            numeric_features = np.array([[title_length, exclamation_count, tweet_log, source_encoded]])
            numeric_features_scaled = scaler.transform(numeric_features)

            # TF-IDF Features
            text_features = vectorizer.transform([cleaned_text])

            # Final Stack
            final_features = hstack([text_features, numeric_features_scaled])

            # Prediction
            prediction = model.predict(final_features)[0]
            
            if prediction == 1:
                st.success("✅ **PREDICTION: THIS NEWS IS REAL**")
            else:
                st.error("🚨 **PREDICTION: THIS NEWS IS LIKELY FAKE**")

# ------------------------
# Live News Feed Section
# ------------------------
with col2:
    st.subheader("📰 Live News Feed (n8n)")
    st.write("Latest news analyzed by AI:")
    
    SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1rp9WfOluqxlVcOaDkOwf4EQy4RSZbi-rGKwNINZ3804/export?format=csv"
    
    try:
        response = requests.get(SHEET_CSV_URL)
        df = pd.read_csv(StringIO(response.text))
        
        # Latest 5 news dikhayen (reverse order)
        latest_news = df.tail(5)[['Headline', 'Verdict']].iloc[::-1]
        
        for index, row in latest_news.iterrows():
            with st.expander(f"{row['Headline'][:50]}..."):
                st.write(f"**Full Headline:** {row['Headline']}")
                st.write(f"**AI Analysis:** {row['Verdict']}")
                
        if st.button("🔄 Refresh Feed"):
            st.rerun()
            
    except Exception as e:
        st.info("Waiting for live feed data...")

st.markdown("---")
st.caption("Model is trained on Live News via n8n Automation.")