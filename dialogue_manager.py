# dialogue_manager.py
# -----------------------------------------
# Milestone 2 – CAASHMORA Bank Virtual Assistant (Terminal Chatbot)
# -----------------------------------------
# Uses:
#   - Trained model: models/intent_pipeline.joblib
#   - Dataset: 6e06fad5-fc7d-41b0-ac26-0c63b25a6d22.csv
# Handles:
#   - Context memory
#   - Intent-based responses
#   - Flow control for card, loan, ATM, money transfer, etc.

import pandas as pd
import random
import string
import joblib
import re
import os
from sklearn.pipeline import Pipeline

# -----------------------------
# Configuration
# -----------------------------
CSV_FILE = "bankbot_final_expanded.csv"  # Updated to use your actual dataset
MODEL_FILE = "models/intent_pipeline.joblib"

# Keyword mapping for short/ambiguous queries
KEYWORD_INTENT_CANDIDATES = {
    "debit card": ["debit_card_block", "debit_card_replacement", "debit_card_replacement2"],
    "credit card": ["credit_card_block", "credit_card_limit", "credit_card_payment"],
    "card": ["debit_card_block", "card_inquiry", "card_request", "credit_card_block"],
    "card block": ["block_card", "debit_card_block", "credit_card_block"],
    "block card": ["block_card", "debit_card_block", "credit_card_block"],
}

# -----------------------------
# Load model & dataset
# -----------------------------
model = None
try:
    if os.path.exists(MODEL_FILE):
        model: Pipeline = joblib.load(MODEL_FILE)
        print("✅ Loaded intent classification model")
    else:
        print("⚠️ Model not found - will use rule-based classification only")

    # Use permissive CSV reading to handle JSON in entities
    df = pd.read_csv(CSV_FILE, encoding='utf-8', quoting=1, quotechar='"', escapechar='\\')
    print(f"✅ Loaded dataset: {len(df)} rows")
except Exception as e:
    # Fallback to more permissive reading
    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8', on_bad_lines='skip')
        print(f"✅ Loaded dataset (skipped bad lines): {len(df)} rows")
    except Exception as e2:
        raise SystemExit(f"Failed to load dataset: {str(e2)}")

df.columns = [c.strip().lower() for c in df.columns]

if "query" not in df.columns and "text" in df.columns:
    df["query"] = df["text"]

intent_responses = {}
for _, row in df.iterrows():
    intent = str(row.get("intent", "")).strip().lower()
    resp = str(row.get("response", "")).strip()
    if intent and resp:
        intent_responses.setdefault(intent, []).append(resp)

# -----------------------------
# Memory and Utility Functions
# -----------------------------
memory = {}

def random_txn_id():
    return "TXN" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def random_balance():
    return f"₹{random.randint(1000, 500000):,}"

def extract_entities(text):
    entities = {}
    acc = re.search(r'\b\d{6,16}\b', text)
    if acc:
        entities["account_number"] = acc.group()
    amt = re.search(r'(?:₹|rs\.?|inr)?\s?(\d{2,9})', text, re.I)
    if amt:
        entities["amount"] = amt.group(1)
    if re.search(r'\bupi\b', text, re.I):
        entities["payment_method"] = "UPI"
    elif re.search(r'\b(bank transfer|neft|imps|rtgs)\b', text, re.I):
        entities["payment_method"] = "Bank Transfer"
    if re.search(r'\b(debit)\b', text, re.I):
        entities["card_type"] = "debit"
    elif re.search(r'\b(credit)\b', text, re.I):
        entities["card_type"] = "credit"
    return entities

def get_response(intent):
    responses = intent_responses.get(intent, [])
    if responses:
        return random.choice(responses)
    return "I'm sorry, I didn't understand that. Could you rephrase?"

# -----------------------------
# Dialogue Management Logic
# -----------------------------
def handle_input(user_input):
    text = user_input.strip().lower()
    entities = extract_entities(user_input)

    # Greeting
    if re.search(r'\b(hi|hello|hey)\b', text):
        return "greet", entities, get_response("greet")

    # Goodbye
    if re.search(r'\b(bye|goodbye|exit)\b', text):
        return "goodbye", entities, get_response("goodbye")

    # Check Balance Flow
    if re.search(r'\b(balance|check balance|account balance)\b', text):
        memory["last_intent"] = "balance_check"
        return "check_balance", entities, "Please provide your account number to check balance."

    if re.fullmatch(r'\d{6,16}', text) and memory.get("last_intent") == "balance_check":
        memory.pop("last_intent", None)
        return "balance_result", {"account_number": text}, f"Your account balance is {random_balance()}."

    # Card Flow
    if re.search(r'\b(card)\b', text):
        if "card_type" not in entities:
            return "card_menu", entities, "Would you like Debit Card or Credit Card services?"
        if entities["card_type"] == "debit":
            return "debit_menu", entities, (
                "Debit Card Options:\n1️⃣ Block Debit Card\n2️⃣ Unblock Debit Card\n3️⃣ Check Status\n4️⃣ Apply New Card\n5️⃣ Report Lost Card"
            )
        if entities["card_type"] == "credit":
            return "credit_menu", entities, (
                "Credit Card Options:\n1️⃣ Block Credit Card\n2️⃣ Unblock Credit Card\n3️⃣ View Bill\n4️⃣ Apply New Card\n5️⃣ Pay Bill"
            )

    if re.search(r'\b(block|lost|stolen)\b', text) and re.search(r'\bdebit\b', text):
        return "debit_card_block", entities, get_response("debit_card_block")
    if re.search(r'\b(block|lost|stolen)\b', text) and re.search(r'\bcredit\b', text):
        return "credit_card_block", entities, get_response("credit_card_block")
    if re.search(r'\b(apply|new card)\b', text) and re.search(r'\bcredit\b', text):
        return "credit_card_apply", entities, get_response("credit_card_apply")
    if re.search(r'\b(apply|new card)\b', text) and re.search(r'\bdebit\b', text):
        return "debit_card_apply", entities, get_response("debit_card_apply")
    if re.search(r'\b(bill|pay)\b', text) and re.search(r'\bcredit\b', text):
        return "credit_card_bill", entities, get_response("credit_card_bill")

    # Loan Flow
    if re.search(r'\b(loan|apply loan|personal loan)\b', text):
        memory["loan_flow"] = True
        return "loan_menu", entities, (
            "Loan Services:\n1️⃣ Apply for Loan\n2️⃣ Check Eligibility\n3️⃣ Loan Balance\n4️⃣ EMI Calculator"
        )

    if re.search(r'\b(emi|emi calculator)\b', text):
        memory["emi_flow"] = True
        return "emi_start", entities, "Please provide loan amount and tenure (e.g., 100000 24 months)."

    if memory.get("emi_flow"):
        nums = re.findall(r'\d+', text)
        if len(nums) >= 2:
            P = float(nums[0])
            n = int(nums[1])
            r = 0.085 / 12
            emi = (P * r * (1 + r)**n) / ((1 + r)**n - 1)
            memory.pop("emi_flow", None)
            return "emi_result", {"amount": P, "tenure": n}, f"Your estimated EMI is ₹{int(emi):,}/month."
        return "emi_error", {}, "Please provide amount and months (e.g., 50000 12)."

    # Money Transfer
    if re.search(r'\b(transfer|send|pay)\b', text):
        if "amount" in entities and "account_number" in entities:
            txn = random_txn_id()
            return "money_transfer", entities, f"₹{entities['amount']} transferred to A/C {entities['account_number']}. Transaction ID: {txn}."
        return "money_transfer_start", entities, "Please provide amount and receiver account number."

    # ATM and Netbanking
    if re.search(r'\batm\b', text):
        return "atm_info", entities, get_response("atm_location")
    if re.search(r'\bnetbanking|net banking|online banking\b', text):
        return "netbanking", entities, get_response("netbanking_register")

    # Try keyword matching for short inputs
    if len(text.split()) <= 2:
        print(f"[DEBUG] Checking keyword matches for: '{text}'")
        for kw, candidates in KEYWORD_INTENT_CANDIDATES.items():
            if text == kw.lower():
                for cand in candidates:
                    if cand in df['intent'].values:
                        resp = get_response(cand)
                        if resp:
                            print(f"[DEBUG] Keyword match: '{kw}' -> {cand}")
                            return cand, entities, resp

    # Model Prediction (if available)
    if model is not None:
        try:
            probs = model.predict_proba([user_input])[0]
            intent = model.classes_[probs.argmax()].lower()
            conf = probs.max()
            print(f"[DEBUG] ML prediction: intent={intent} confidence={conf:.2f}")

            # Use ML prediction if confident enough
            if conf >= 0.4:  # Lower threshold since we have fallbacks
                response = get_response(intent)
                if response:
                    return intent, entities, response
        except Exception as e:
            print(f"[DEBUG] Prediction error: {str(e)}")

    return "unknown", entities, "Sorry, I didn't understand that. Could you provide more details?"

# -----------------------------
# Main Chat Loop
# -----------------------------
def main():
    print("\n💬 CAASHMORA Bank Virtual Assistant (Milestone 2)")
    print("Type 'exit' to quit.\n")
    print("🤖 Bot: Hello! How can I assist you with banking today?\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\n🤖 Bot: Thank you for banking with us. Have a great day!")
            break

        intent, entities, response = handle_input(user_input)
        print(f"\n🎯 Intent: {intent}")
        if entities:
            print(f"📎 Entities: {entities}")
        print(f"🤖 Bot: {response}\n")

if __name__ == "__main__":
    main()
