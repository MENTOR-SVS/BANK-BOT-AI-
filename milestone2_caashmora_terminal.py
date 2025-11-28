import re, sys, csv, os, joblib, pandas as pd
from datetime import datetime

WELCOME = "💬 Bank Assistant (Milestone 2). Type 'exit' to quit."
DATA_FILENAME = "bankbot_finial_expanded.csv"  # Corrected typo in filename
MODEL_FILE = "models/intent_pipeline.joblib"
LOG_PATH = "./chat_logs.csv"

# Load trained model and dataset
model = None
df = None

try:
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        print("✅ Loaded trained intent model")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")

try:
    df = pd.read_csv(DATA_FILENAME, encoding='utf-8', on_bad_lines='skip')
    df.columns = [c.strip().lower() for c in df.columns]
    if "question" in df.columns and "text" not in df.columns:
        df["text"] = df["question"]
    if "answer" in df.columns and "response" not in df.columns:
        df["response"] = df["answer"]
    print(f"✅ Loaded dataset: {len(df)} rows")
except Exception as e:
    print(f"⚠️ Could not load dataset: {e}")
    df = None

# Keyword mapping for short/ambiguous queries
KEYWORD_INTENT_CANDIDATES = {
    "card": ["block_card", "card_reissue", "lost_card"],
    "debit card": ["block_card", "card_reissue"],
    "credit card": ["block_card", "card_reissue"],
    "block card": ["block_card"],
    "lost card": ["lost_card"],
    "new card": ["card_reissue"],
}

# ---------- State ----------
state = {
    "balance": 40295702,
    "context": None,
    "ctx_data": {},
    "last_topic": None,
    "last_intent": None,
}
logs = []

# ---------- Helpers ----------
def say(text): print(f"🤖 Bot: {text}")
def set_topic(topic): state["last_topic"] = topic
def print_intent_line(intent): state["last_intent"] = intent; print(f"🎯 Intent: {intent}")
def add_log(user, intent): logs.append({"time": datetime.now().isoformat(timespec="seconds"), "user": user, "intent": intent})
def in_(t,*w): t=t.lower();return any(x in t for x in w)
def sanitize_num(t): m=re.findall(r"\d+",t.replace(",",""));return int(m[0]) if m else None
def is_confirmation(t): return t.strip().lower() in {"yes","ok","okay","sure","confirm","go ahead"}

# ---------- Intents ----------
def detect_intent(t):
    t_lower = t.lower().strip()
    
    # Try keyword matching for short inputs first
    if len(t_lower.split()) <= 2 and df is not None:
        for kw, candidates in KEYWORD_INTENT_CANDIDATES.items():
            if t_lower == kw:
                for cand in candidates:
                    if cand in df['intent'].values:
                        print(f"[DEBUG] Keyword match: '{kw}' -> {cand}")
                        return cand
    
    # Try trained model if available
    if model is not None:
        try:
            probs = model.predict_proba([t])[0]
            intent = model.classes_[probs.argmax()].lower()
            conf = probs.max()
            if conf >= 0.4:
                print(f"[DEBUG] ML prediction: {intent} (conf={conf:.2f})")
                return intent
        except Exception as e:
            print(f"[DEBUG] Model prediction error: {e}")
    
    # Fallback to rule-based detection
    if in_(t_lower,"balance"): return "account_statement"
    if in_(t_lower,"loan"): return "loan_info"
    if in_(t_lower,"transfer money","send money"): return "upi_setup"
    if in_(t_lower,"card"): return "block_card"
    if in_(t_lower,"ifsc"): return "ifsc_search"
    if in_(t_lower,"manager"): return "who_is_manager"
    if in_(t_lower,"branch","nearest branch","atm"): return "atm_location"
    if in_(t_lower,"cheque","stop payment"): return "cheque_deposit"
    if in_(t_lower,"hello","hi","hey"): return "greet"
    return "fallback"

# ---------- Handlers ----------
def get_response(intent):
    """Get response from dataset or provide a default"""
    if df is not None:
        try:
            matches = df[df['intent'] == intent]
            if not matches.empty:
                response = matches['response'].values[0]
                if isinstance(response, str) and response.strip():
                    return response
        except:
            pass
    # Fallback responses
    fallback_responses = {
        "account_statement": "Your current balance is ₹40,295,702.",
        "loan_info": "Loan services: 1) Home 2) Car 3) Personal 4) Education",
        "upi_setup": "Sure, please provide recipient details.",
        "block_card": "I can help you block your card. What type - debit or credit?",
        "ifsc_search": "Please provide your Bank, Branch and City to find IFSC code.",
        "who_is_manager": "Please provide the Bank and City to find the branch manager details.",
        "atm_location": "Please provide your City to find the nearest ATM/branch.",
        "cheque_deposit": "For lost cheque book: 1️⃣ Stop payment 2️⃣ Request new book 3️⃣ Visit branch.",
        "greet": "Hello! How can I assist you today?",
        "fallback": "I'm sorry, I didn't understand that. Could you rephrase?"
    }
    return fallback_responses.get(intent, "I'm sorry, I didn't understand that.")

def handle_greet(_): say(get_response("greet")); set_topic("greet")
def handle_check_balance(_): say(get_response("account_statement")); set_topic("account_statement")
def handle_money_transfer(_): say(get_response("upi_setup")); set_topic("transfer")
def handle_card_menu(_): say(get_response("block_card")); set_topic("card_menu")
def handle_loan_menu(_): say(get_response("loan_info")); set_topic("loan_menu")
def handle_ifsc_query(_): say(get_response("ifsc_search")); set_topic("ifsc")
def handle_manager_lookup(_): say(get_response("who_is_manager")); set_topic("manager_lookup")
def handle_branch_lookup(_): say(get_response("atm_location")); set_topic("branch_lookup")
def handle_cheque_help(_): say(get_response("cheque_deposit")); set_topic("cheque_help")
def handle_fallback(_): say("I'm sorry, I didn’t understand that. Could you rephrase?"); set_topic("fallback")

# ---------- Router ----------
def route(user_text):
    intent = detect_intent(user_text)
    add_log(user_text,intent)
    print_intent_line(intent)
    if intent=="account_statement": handle_check_balance(user_text)
    elif intent=="loan_info": handle_loan_menu(user_text)
    elif intent=="upi_setup": handle_money_transfer(user_text)
    elif intent=="block_card": handle_card_menu(user_text)
    elif intent=="ifsc_search": handle_ifsc_query(user_text)
    elif intent=="who_is_manager": handle_manager_lookup(user_text)
    elif intent=="atm_location": handle_branch_lookup(user_text)
    elif intent=="cheque_deposit": handle_cheque_help(user_text)
    elif intent=="greet": handle_greet(user_text)
    else: handle_fallback(user_text)

# ---------- Main ----------
def main():
    print(WELCOME)
    while True:
        try: 
            user = input("👤 You: ")
            if user.lower() in {"exit","quit"}:
                say("Goodbye!"); break
            route(user)
        except (KeyboardInterrupt,EOFError):
            say("Session Ended."); break

if __name__ == "__main__":
    main()
