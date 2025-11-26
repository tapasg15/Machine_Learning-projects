# ================================
# train_model.py  (UPDATED WITH PREPROCESSING)
# ================================

import os
import pickle
import re

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ---------- NLP imports ----------
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources (only first time; later they are cached)
nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ================================
# 1. Text Preprocessing Function
# ================================
def preprocess_text(text: str) -> str:
    """
    Clean and normalize resume text:
    - convert to lowercase
    - remove non-letters
    - remove stopwords
    - lemmatize words
    """
    # convert to string just in case
    text = str(text)

    # lowercase
    text = text.lower()

    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # tokenize by spaces
    words = text.split()

    # remove stopwords and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # join back to a single string
    return " ".join(cleaned_words)


# ================================
# 2. Paths & setup
# ================================
DATA_PATH = "data/resumes.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "resume_model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# 3. Load dataset
# ================================
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print("\nFirst 5 rows (raw):")
print(df.head())

print("\nCategory counts:")
print(df["category"].value_counts())

# Apply preprocessing to the text column
print("\nApplying text preprocessing...")
df["text_clean"] = df["text"].astype(str).apply(preprocess_text)

print("\nFirst 5 rows (after preprocessing):")
print(df[["text", "text_clean"]].head())

X = df["text_clean"]
y = df["category"]

# ================================
# 4. Train–test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ================================
# 5. Text → TF-IDF features
# ================================
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),     # use unigrams + bigrams
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ================================
# 6. Train model (Naive Bayes)
# ================================
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
# ================================
# 7. Evaluate model
# ================================
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================================
# 8. Save model and vectorizer
# ================================
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VEC_PATH, "wb") as f:
    pickle.dump(tfidf, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Vectorizer saved to: {VEC_PATH}")
print("\n✅ Training complete. You can now run:  python3 -m streamlit run app.py")
