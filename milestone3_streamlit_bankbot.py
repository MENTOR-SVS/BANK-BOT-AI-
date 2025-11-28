# ------------------------------------------------------------
# 🏦 BANKBOT — Milestone 3 (Streamlit + ML + Rule-based Hybrid)
# ------------------------------------------------------------
# ✅ Fixes greeting issue (handles hello/hi/thanks manually)
# ✅ Uses Milestone 1 trained model + responses
# ✅ Displays intent, confidence & entities in Streamlit UI
# ✅ Graceful fallback + multi-turn context handling
# ------------------------------------------------------------

import os
import re
import json
import joblib
import streamlit as st

# ------------------------------------------------------------
# 1️⃣ Load Model and Responses
# ------------------------------------------------------------
MODEL_PATH = "models/intent_pipeline.joblib"
RESPONSES_PATH = "models/intent_responses.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(RESPONSES_PATH):
    st.error("❌ Model or response mapping not found. Run Milestone 1 first.")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
    RESPONSES = json.load(f)

# ------------------------------------------------------------
# 2️⃣ Entity Extraction
# ------------------------------------------------------------
ENTITY_PATTERNS = {
    "account_number": r"\b\d{6,16}\b",
    "amount": r"\b\d{2,8}\b",
    "city": r"\b(?:chennai|puducherry|mumbai|delhi|bangalore)\b",
    "card_type": r"\b(?:debit|credit)\b",
}

def extract_entities(text):
    ent = {}
    lower = text.lower()
    for k, p in ENTITY_PATTERNS.items():
        m = re.search(p, lower)
        if m:
            ent[k] = m.group()
    return ent

# ------------------------------------------------------------
# 3️⃣ Intent Prediction + Rule Fallbacks
# ------------------------------------------------------------
def predict_intent(text):
    t = text.lower().strip()

    # --- Rule-based shortcuts ---
    if any(w in t for w in ["hi", "hello", "hey"]):
        return "greet", 1.0
    if "thank" in t:
        return "thanks", 1.0
    if "bye" in t or "goodbye" in t:
        return "goodbye", 1.0

    # --- ML Prediction ---
    try:
        probs = model.predict_proba([text])[0]
        idx = probs.argmax()
        intent = model.classes_[idx]
        conf = float(probs[idx])
        return (intent, conf)
    except Exception as e:
        return ("unknown", 0.0)

# ------------------------------------------------------------
# 4️⃣ Response Generation
# ------------------------------------------------------------
def get_response(intent, entities):
    # From dataset if available
    if intent in RESPONSES and RESPONSES[intent]:
        return RESPONSES[intent][0]

    # Otherwise rule-based defaults
    defaults = {
        "greet": "Hello 👋 How can I assist you today?",
        "thanks": "You're welcome! 😊",
        "goodbye": "Goodbye 👋 Have a great day!",
        "account_statement": "Your account balance is ₹2,50,000.",
        "loan_info": "We offer personal, car, and home loans. Which would you like to know about?",
        "block_card": "Would you like to block a debit or credit card?",
        "unknown": "I'm sorry, I didn’t understand that. Could you rephrase?",
    }
    return defaults.get(intent, defaults["unknown"])

# ------------------------------------------------------------
# 5️⃣ Streamlit Chat UI
# ------------------------------------------------------------
st.set_page_config(page_title="BankBot", page_icon="🏦", layout="centered")

st.title("🏦 BankBot Assistant — Milestone 3")
st.caption("Conversational Banking Assistant (Streamlit UI + ML)")

# Persistent session storage
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug" not in st.session_state:
    st.session_state.debug = []

# Display chat history
for sender, msg in st.session_state.messages:
    st.chat_message("user" if sender == "user" else "assistant").write(msg)

# User input
user_input = st.chat_input("Type your message here...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(("user", user_input))

    entities = extract_entities(user_input)
    intent, conf = predict_intent(user_input)
    response = get_response(intent, entities)

    st.session_state.messages.append(("bot", response))
    st.chat_message("assistant").write(response)

    # Log for debugging
    st.session_state.debug.append(f"Intent: {intent} | Confidence: {conf:.2f}")
    if entities:
        st.session_state.debug.append(f"Entities: {entities}")

# Debug info
with st.expander("🧩 Debug Info"):
    st.write("\n".join(st.session_state.debug))
