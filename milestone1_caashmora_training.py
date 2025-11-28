"""
🏦 Milestone 1 — Intent & Entity Recognition Engine
-----------------------------------------------------------
Features:
1️⃣ Safely loads multiple possible dataset files
2️⃣ Handles CSV quoting, bad lines, and renames
3️⃣ Builds & trains TF-IDF + Logistic Regression classifier
4️⃣ Extracts key banking entities (slots)
5️⃣ Evaluates accuracy and saves all artifacts
6️⃣ Tests predictions interactively
-----------------------------------------------------------
Outputs:
 • models/intent_pipeline.joblib
 • models/intent_responses.json
 • models/metrics.json
"""

import os
import json
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ----------------------------------------------------------
# 🔧 Configuration
# ----------------------------------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "intent_pipeline.joblib")
RESPONSES_PATH = os.path.join(MODEL_DIR, "intent_responses.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

CANDIDATE_FILES = [
    "bankbot_finial_expanded.csv",  # Corrected typo in filename
    "bankbot_final_expanded_v2.csv",
    "bankbot_final_expanded.csv",
    "bank_chatbot_dataset_large.csv",
    "bank_chatbot_dataset_large (2).csv",
    "training_data.csv"
]

# ----------------------------------------------------------
# 📥 1️⃣ Load Dataset Safely
# ----------------------------------------------------------
def load_dataset():
    df, loaded_path = None, None
    for candidate in CANDIDATE_FILES:
        if os.path.exists(candidate):
            try:
                df = pd.read_csv(candidate, encoding="utf-8",
                                 on_bad_lines="skip", quotechar='"', escapechar='\\')
                loaded_path = candidate
                break
            except Exception:
                try:
                    df = pd.read_csv(candidate, encoding="utf-8", on_bad_lines="skip")
                    loaded_path = candidate
                    break
                except Exception:
                    pass

    if df is None:
        raise SystemExit(f"❌ No valid dataset found. Tried: {CANDIDATE_FILES}")

    # Clean column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Map common column name variations
    if "text" not in df.columns:
        if "query" in df.columns:
            df["text"] = df["query"]
        elif "question" in df.columns:
            df["text"] = df["question"]
    
    if "response" not in df.columns and "answer" in df.columns:
        df["response"] = df["answer"]
    elif "response" not in df.columns:
        df["response"] = ""

    if not all(col in df.columns for col in ["text", "intent"]):
        print("❌ Available columns:", list(df.columns))
        raise ValueError("Dataset must contain columns mapped to ['text','intent']")

    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)
    df["intent"] = df["intent"].astype(str).str.strip().str.lower()
    print(f"✅ Loaded {loaded_path} — {len(df)} rows, {df['intent'].nunique()} intents")
    return df

# ----------------------------------------------------------
# 🧠 2️⃣ Build Model Pipeline
# ----------------------------------------------------------
def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

# ----------------------------------------------------------
# 💬 3️⃣ Build Intent → Response Map
# ----------------------------------------------------------
def make_intent_response_map(df):
    resp_map = {}
    if "response" in df.columns:
        for _, r in df.iterrows():
            intent = str(r.get("intent", "")).strip().lower()
            resp = str(r.get("response", "")).strip()
            if intent and resp:
                resp_map.setdefault(intent, []).append(resp)
    return resp_map

# ----------------------------------------------------------
# 🧩 4️⃣ Entity Extraction (Slot Filling)
# ----------------------------------------------------------
SLOT_PATTERNS = {
    "account_number": r"\b\d{6,16}\b",
    "amount": r"\b\d{2,8}\b",
    "mobile_number": r"\b\d{10}\b",
    "city_name": r"\b(?:chennai|puducherry|mumbai|delhi|bangalore)\b",
    "payment_method": r"\b(?:upi|bank transfer|neft|imps|rtgs)\b",
    "card_type": r"\b(?:debit|credit)\b",
    "account_type": r"\b(?:savings|current)\b"
}

def extract_entities(text):
    entities = {}
    lower = text.lower()
    for slot, pattern in SLOT_PATTERNS.items():
        m = re.search(pattern, lower)
        if m:
            entities[slot] = m.group()
    return entities

# ----------------------------------------------------------
# 🚀 5️⃣ Train, Evaluate, Save
# ----------------------------------------------------------
def train_and_save(df):
    X, y = df["text"], df["intent"]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = build_pipeline()
    print("🧠 Training model...")
    model.fit(X_train, y_train)

    print("🔍 Evaluating model...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    resp_map = make_intent_response_map(df)
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(resp_map, f, ensure_ascii=False, indent=2)
    print(f"💬 Responses saved → {RESPONSES_PATH}")

    metrics = {
        "n_rows": len(df),
        "n_intents": df["intent"].nunique(),
        "report": report
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"📊 Metrics saved → {METRICS_PATH}")

    return model

# ----------------------------------------------------------
# 🧪 6️⃣ Test Predictions
# ----------------------------------------------------------
def test_sample(model, query):
    pred = model.predict([query])[0]
    prob = max(model.predict_proba([query])[0])
    ents = extract_entities(query)
    print(f"\n🗣 {query}")
    print(f"🎯 Intent: {pred} (Confidence: {prob:.2f})")
    print(f"📎 Entities: {ents}")

# ----------------------------------------------------------
# ▶️ Main Entry
# ----------------------------------------------------------
def main():
    df = load_dataset()
    model = train_and_save(df)

    print("\n✅ Testing few sample queries...")
    samples = [
        "Check my account balance",
        "Transfer 5000 to account 9876543210 via UPI",
        "Block my debit card",
        "Increase my credit card limit",
        "Nearest ATM in Puducherry",
        "Apply for a new savings account"
    ]
    for s in samples:
        test_sample(model, s)

    print("\n🎯 Milestone 1 completed successfully — model ready for Milestone 2.")

if __name__ == "__main__":
    main()

