# ------------------------------------------------------------
# 🏦 BANK PORTAL (Final Integrated Project)
# ------------------------------------------------------------
# Features:
# ✅ Login System (Customer / Admin)
# ✅ SQLite Database for users & accounts
# ✅ Customer Dashboard with chatbot (Milestone 3)
# ✅ Admin Dashboard with analytics (Milestone 4)
# ------------------------------------------------------------
# Run:  streamlit run bank_portal_app.py
# ------------------------------------------------------------

import streamlit as st
import sqlite3
import os
import pandas as pd
import json
import joblib
from datetime import datetime

# ============================================================
# DATABASE SETUP
# ============================================================
DB_PATH = "bank.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT,
            name TEXT,
            account_no TEXT,
            balance REAL
        )
    """)
    conn.commit()

    # Sample data
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        users = [
            ("admin", "admin123", "admin", "Bank Manager", "-", 0.0),
            ("suriya", "1234", "customer", "Suriya Varshan", "ACC9876543210", 125000.50),
            ("pooja", "1234", "customer", "Pooja Sree", "ACC5432167890", 98200.75)
        ]
        c.executemany("INSERT INTO users (username,password,role,name,account_no,balance) VALUES (?,?,?,?,?,?)", users)
        conn.commit()
    conn.close()

init_db()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def load_chat_model():
    model_path = "models/intent_pipeline.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_responses():
    path = "models/intent_responses.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_bot_response(user_input, model, responses, df_data=None):
    """Enhanced intent detection with keyword fallback"""
    
    # Keyword mapping for short queries
    KEYWORD_INTENT_MAP = {
        "hello": "greet", "hi": "greet", "hey": "greet",
        "card": "block_card", "block": "block_card",
        "balance": "account_statement", "check": "account_statement",
        "loan": "loan_info", "atm": "atm_location", "branch": "atm_location",
        "transfer": "upi_setup", "pay": "upi_setup", "ifsc": "ifsc_search"
    }
    
    intent = "unknown"
    conf = 0.0
    reply = "I'm sorry, I didn't understand that. Could you provide more details?"
    
    text_lower = user_input.lower().strip()
    
    # Step 1: Keyword matching for short queries
    if len(text_lower.split()) <= 3:
        for keyword, mapped_intent in KEYWORD_INTENT_MAP.items():
            if keyword in text_lower:
                intent = mapped_intent
                conf = 0.95
                break
    
    # Step 2: ML model prediction if no keyword match
    if intent == "unknown" and model:
        try:
            probs = model.predict_proba([user_input])[0]
            pred_intent = model.classes_[probs.argmax()]
            pred_conf = probs.max()
            if pred_conf > 0.4:
                intent = pred_intent
                conf = pred_conf
        except:
            pass
    
    # Step 3: Get response
    if intent != "unknown":
        # Try dataset first
        if df_data is not None:
            try:
                intent_col = "intent" if "intent" in df_data.columns else "Intent"
                matches = df_data[df_data[intent_col].astype(str).str.lower() == intent.lower()]
                if not matches.empty:
                    response = str(matches["response"].values[0])
                    if response and response != "nan":
                        reply = response
            except:
                pass
        
        # Fallback to responses map
        if reply.startswith("I'm sorry"):
            if intent in responses:
                resp_list = responses[intent]
                reply = resp_list[0] if isinstance(resp_list, list) else str(resp_list)
    
    return intent, conf, reply

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="🏦 Bank Portal", page_icon="💰", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None

# ============================================================
# LOGIN PAGE
# ============================================================
def login_page():
    st.title("🏦 Welcome to SmartBank Portal")
    st.subheader("🔐 Login to Continue")

    role = st.radio("Select Role", ["Customer", "Admin"])
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        user = get_user(username, password)
        if user:
            if user[3].lower() == "customer" and role == "Customer":
                st.session_state.user = user
                st.session_state.role = "customer"
                st.success("✅ Customer Login Successful!")
                st.rerun()
            elif user[3].lower() == "bank manager" and role == "Admin":
                st.session_state.user = user
                st.session_state.role = "admin"
                st.success("✅ Admin Login Successful!")
                st.rerun()
            else:
                st.error("❌ Role mismatch. Check role selection.")
        else:
            st.error("❌ Invalid credentials.")

# ============================================================
# CUSTOMER DASHBOARD (Milestone 3 Chatbot + Details)
# ============================================================
def customer_dashboard():
    user = st.session_state.user
    name = user[4]
    account_no = user[5]
    balance = user[6]

    st.sidebar.title("🏦 Customer Panel")
    st.sidebar.success(f"Welcome, {name}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    st.title(f"👋 Hello {name} — Your Banking Dashboard")
    st.subheader("💳 Account Summary")

    col1, col2 = st.columns(2)
    col1.metric("Account Number", account_no)
    col2.metric("Balance", f"₹{balance:,.2f}")

    st.divider()
    st.subheader("💬 Chat with BankBot Assistant")

    # Load ML Model
    model = load_chat_model()
    responses = load_responses()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for sender, msg in st.session_state.messages:
        if sender == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.chat_message("user").write(user_input)

        # Get bot response with enhanced intent detection
        intent, conf, reply = get_bot_response(user_input, model, responses, None)

        st.chat_message("assistant").write(reply)
        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("bot", reply))

# ============================================================
# ADMIN DASHBOARD (Milestone 4 Panel)
# ============================================================
def admin_dashboard():
    st.sidebar.title("⚙️ Admin Panel")
    st.sidebar.success("Admin: Bank Manager")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    st.title("🏦 Admin Dashboard — BankBot Analytics")

    log_file = "chat_logs.csv"
    data_file = "bankbot_finial_expanded.csv"

    if os.path.exists(data_file):
        df_train = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        df_train = pd.DataFrame()

    if os.path.exists(log_file):
        df_logs = pd.read_csv(log_file)
    else:
        df_logs = pd.DataFrame(columns=["time", "user", "intent"])

    tabs = st.tabs(["📊 Dashboard", "📂 Training Data", "💬 FAQs", "⚙️ Settings"])

    with tabs[0]:
        st.subheader("📈 Query Analytics")
        col1, col2, col3 = st.columns(3)
        total_queries = len(df_logs)
        success_rate = (
            (df_logs["intent"] != "fallback").sum() / len(df_logs) * 100
            if len(df_logs) > 0 else 0
        )
        intents = df_logs["intent"].nunique()
        col1.metric("Total Queries", total_queries)
        col2.metric("Success Rate", f"{success_rate:.1f}%")
        col3.metric("Unique Intents", intents)
        if not df_logs.empty:
            st.bar_chart(df_logs["intent"].value_counts())

    with tabs[1]:
        st.subheader("🧠 Training Data")
        if not df_train.empty:
            st.dataframe(df_train.head(20))
            st.download_button(
                "⬇️ Export Training Data",
                data=df_train.to_csv(index=False).encode("utf-8"),
                file_name="training_data_export.csv",
                mime="text/csv"
            )
        else:
            st.warning("No training dataset found.")

    with tabs[2]:
        st.subheader("💬 FAQs & Response Map")
        path = "models/intent_responses.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                responses = json.load(f)
            intent = st.selectbox("Select Intent", list(responses.keys()))
            st.write(responses[intent][:3])
        else:
            st.warning("Response file missing.")

    with tabs[3]:
        st.subheader("⚙️ Admin Settings")
        if st.button("🗑️ Clear Logs"):
            if os.path.exists("chat_logs.csv"):
                os.remove("chat_logs.csv")
                st.success("Logs cleared.")
        if st.button("🔁 Retrain Model"):
            st.info("Model retraining feature will be added in the next phase.")

# ============================================================
# MAIN LOGIC
# ============================================================
if st.session_state.user is None:
    login_page()
else:
    if st.session_state.role == "customer":
        customer_dashboard()
    elif st.session_state.role == "admin":
        admin_dashboard()
