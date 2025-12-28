import streamlit as st
import pickle
import re
import numpy as np
import nltk
import pandas as pd
import requests
import google.generativeai as genai
from io import StringIO
from nltk.corpus import stopwords
from scipy.sparse import hstack

# ------------------------
# Gemini AI Setup
# ------------------------
# Streamlit Secrets se API Key uthayenge
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel('gemini-pro')

# ------------------------
# NLTK & Setup
# ------------------------
nltk.download("stopwords")
stop_words = stopwords.words("english")

st.set_page_config(page_title="Hybrid Fake News Detector", layout="wide")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load preprocessing objects
@st.cache_resource
def load_models():
    # model/ folder path ensure karein GitHub par sahi ho
    with open("model/hybrid_svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/hybrid_tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model/hybrid_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, vectorizer, scaler

# ------------------------
# Streamlit UI
# ------------------------
st.title("🛡️ HYBRID FAKE NEWS DETECTOR")
st.write("Machine Learning Analysis + Real-time AI Fact-Checking")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Headline Analysis")
    user_input = st.text_area("Enter news headline here:", placeholder="e.g., Scientist found a way to live forever...")

    if st.button("Check Authenticity"):
        if user_input.strip() == "":
            st.warning("Please enter a headline.")
        else:
            # Layout for dual results
            res_col1, res_col2 = st.columns(2)
            
            # --- 1. Machine Learning Prediction (SVM) ---
            with res_col1:
                st.info("🤖 SVM Model Result")
                try:
                    model, vectorizer, scaler = load_models()
                    cleaned_text = clean_text(user_input)
                    
                    # Numeric features
                    title_length = len(user_input)
                    exclamation_count = user_input.count("!")
                    tweet_log = np.log1p(10)
                    source_encoded = 0

                    numeric_features = np.array([[title_length, exclamation_count, tweet_log, source_encoded]])
                    numeric_features_scaled = scaler.transform(numeric_features)
                    text_features = vectorizer.transform([cleaned_text])
                    final_features = hstack([text_features, numeric_features_scaled])

                    prediction = model.predict(final_features)[0]
                    
                    if prediction == 1:
                        st.success("✅ **REAL (Pattern Match)**")
                    else:
                        st.error("🚨 **FAKE (Pattern Match)**")
                    st.caption("Based on linguistic patterns.")
                except Exception as e:
                    st.error(f"ML Error: {e}")

            # --- 2. Gemini AI Fact-Check ---
            with res_col2:
                st.info("🌐 AI Fact-Check")
                with st.spinner('Checking facts...'):
                    try:
                        prompt = f"Fact-check this news headline: '{user_input}'. Is it true or false based on recent news? Give a clear 'Verdict: TRUE/FALSE' and a short reason."
                        response = ai_model.generate_content(prompt)
                        st.write(response.text)
                        st.caption("Based on real-time factual data.")
                    except Exception as e:
                        st.error(f"AI Error: {e}")

# ------------------------
# Live News Feed (Google Sheets)
# ------------------------
with col2:
    st.subheader("📰 Live n8n Feed")
    st.write("Latest results from your automated sheet:")
    
    SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1rp9WfOluqxlVcOaDkOwf4EQy4RSZbi-rGKwNINZ3804/export?format=csv"
    
    try:
        response = requests.get(SHEET_CSV_URL)
        df = pd.read_csv(StringIO(response.text))
        latest_news = df.tail(5)[['Headline', 'Verdict']].iloc[::-1]
        
        for index, row in latest_news.iterrows():
            with st.expander(f"{str(row['Headline'])[:40]}..."):
                st.write(f"**News:** {row['Headline']}")
                st.write(f"**Analysis:** {row['Verdict']}")
                
        if st.button("🔄 Refresh"):
            st.rerun()
    except:
        st.info("Waiting for sheet data...")

st.markdown("---")
st.caption("Developed with n8n Automation & Streamlit Cloud.")

