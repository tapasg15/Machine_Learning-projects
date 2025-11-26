# ================================
# app.py  (WITH PREPROCESSING + PDF + GRAPHS)
# ================================

import os
import pickle
import re

import streamlit as st
from PyPDF2 import PdfReader

# [NEW] imports for graphs & tables
import pandas as pd
import matplotlib.pyplot as plt

# ---------- NLP imports ----------
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources (only first time)
nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ================================
# 1. Text Preprocessing Function (SAME AS train_model.py)
# ================================
def preprocess_text(text: str) -> str:
    """
    Clean and normalize resume text:
    - convert to lowercase
    - remove non-letters
    - remove stopwords
    - lemmatize words
    """
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(cleaned_words)


# ================================
# 2. Load saved model & vectorizer
# ================================
MODEL_PATH = "models/resume_model.pkl"
VEC_PATH = "models/tfidf_vectorizer.pkl"

if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
    st.error("Model files not found. Please run train_model.py first to create the model.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VEC_PATH, "rb") as f:
    tfidf = pickle.load(f)


# ================================
# 3. Skill extraction helper
# ================================
COMMON_SKILLS = [
    "python", "java", "c++", "c#", "sql", "mysql", "postgresql", "mongodb",
    "html", "css", "javascript", "react", "node", "django", "flask",
    "power bi", "excel", "tableau", "pandas", "numpy", "machine learning",
    "deep learning", "tensorflow", "keras", "nlp", "git", "github"
]

def extract_skills(text: str):
    text_lower = text.lower()
    found = [skill for skill in COMMON_SKILLS if skill in text_lower]
    return sorted(set(found))


# ================================
# 4. PDF text extraction helper
# ================================
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""


# ================================
# 5. Streamlit page layout
# ================================
st.set_page_config(page_title="Resume Classifier", page_icon="📄")

st.title("📄 Resume Classifier using Machine Learning")
st.write(
    """
    This app classifies a resume into categories such as 
    **Software Developer, Web Developer, Data Analyst, Machine Learning Engineer**
    based on the text content.
    """
)

st.subheader("Step 1: Paste your resume text")
resume_text = st.text_area(
    "Paste the resume content here:",
    height=250,
    placeholder="Copy and paste the full resume text..."
)

st.write("or")

st.subheader("Step 2: (Optional) Upload a .txt or .pdf file")
uploaded_file = st.file_uploader(
    "Upload a resume file (.txt or .pdf)",
    type=["txt", "pdf"]
)

# Load text from uploaded file if any
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        file_text = extract_text_from_pdf(uploaded_file)
    else:
        file_bytes = uploaded_file.read()
        file_text = file_bytes.decode("utf-8", errors="ignore")

    if file_text:
        if not resume_text.strip():
            resume_text = file_text
        st.info("Text loaded from uploaded file. You can edit it above if needed.")
    else:
        st.warning("Could not extract text from the uploaded file.")


# ================================
# 6. Prediction + Graphs
# ================================
if st.button("🔍 Classify Resume"):
    if not resume_text.strip():
        st.error("Please paste resume text or upload a file first.")
    else:
        # --- Preprocess before sending to TF-IDF ---
        cleaned_text = preprocess_text(resume_text)

        # transform and predict
        features = tfidf.transform([cleaned_text])
        prediction = model.predict(features)[0]

        st.success(f"Predicted Category: **{prediction}**")

        # --------- 6A. Probabilities (Table + Bar Chart) ---------
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            classes = model.classes_

            # Create DataFrame for nicer display
            prob_df = pd.DataFrame({
                "Category": classes,
                "Probability": probs
            }).sort_values("Probability", ascending=False)

            st.subheader("Prediction Probabilities (Table)")
            st.table(prob_df)

            st.subheader("Prediction Probabilities (Bar Chart)")

            # Create bar chart using matplotlib
            fig, ax = plt.subplots()
            ax.bar(prob_df["Category"], prob_df["Probability"])
            ax.set_ylabel("Probability")
            ax.set_xlabel("Category")
            ax.set_ylim(0, 1)
            ax.set_title("Model Confidence by Category")

            # Rotate x-labels for readability
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")

            st.pyplot(fig)

        # --------- 6B. Skills section ---------
        skills = extract_skills(resume_text)
        st.subheader("Detected Skills (simple keyword-based):")
        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No common skills from the predefined list were detected.")
