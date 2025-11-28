# ------------------------------------------------------------
# 🏦 BankBot Milestone 4 — Admin Panel & Knowledge Base
# Weeks 7–8: Fully Working Application
# ------------------------------------------------------------
# Features:
# ✅ Dashboard with analytics
# ✅ Training data viewer
# ✅ User query monitor (from chat_logs.csv)
# ✅ Placeholder for FAQ & settings
# ✅ Uses models & data from Milestones 1–3
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import os
import joblib
import json
from datetime import datetime

# ------------------------------------------------------------
# 🧭 PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="🏦 BankBot Admin Panel", page_icon="⚙️", layout="wide")

st.title("🏦 BankBot Assistant — Milestone 4")
st.caption("Admin Panel & Knowledge Base (Weeks 7–8)")

# ------------------------------------------------------------
# 📂 PATHS & GLOBALS
# ------------------------------------------------------------
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "intent_pipeline.joblib")
RESPONSES_FILE = os.path.join(MODEL_DIR, "intent_responses.json")
LOG_FILE = "chat_logs.csv"
DATA_FILE = "bankbot_finial_expanded.csv"

# ------------------------------------------------------------
# 🧠 LOAD MODEL + RESPONSES
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

@st.cache_data
def load_responses():
    if os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, on_bad_lines='skip')
    return pd.DataFrame()

@st.cache_data
def load_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["time", "user", "intent"])

model = load_model()
responses = load_responses()
df_data = load_data()
df_logs = load_logs()

# ------------------------------------------------------------
# 🧭 SIDEBAR NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("⚙️ Admin Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "📂 Training Data", "💬 FAQs", "📊 Analytics", "⚙️ Settings"]
)

# ------------------------------------------------------------
# 🏠 DASHBOARD
# ------------------------------------------------------------
if page == "🏠 Dashboard":
    st.subheader("📊 Chatbot Dashboard Overview")

    col1, col2, col3 = st.columns(3)
    total_queries = len(df_logs)
    unique_intents = df_logs["intent"].nunique() if not df_logs.empty else 0
    success_rate = (
        (df_logs["intent"] != "fallback").sum() / len(df_logs) * 100
        if len(df_logs) > 0 else 0
    )

    col1.metric("Total Queries", total_queries)
    col2.metric("Success Rate", f"{success_rate:.1f}%")
    col3.metric("Unique Intents", unique_intents)

    st.divider()
    st.write("### Recent Queries")
    if not df_logs.empty:
        st.dataframe(df_logs.tail(10).reset_index(drop=True))
    else:
        st.info("No chat logs found. Try chatting in Milestone 3 and refresh.")

# ------------------------------------------------------------
# 📂 TRAINING DATA PAGE
# ------------------------------------------------------------
elif page == "📂 Training Data":
    st.subheader("🧠 Training Data Viewer")
    if not df_data.empty:
        st.write(f"✅ Loaded {len(df_data)} training samples")
        st.dataframe(df_data.head(15))
    else:
        st.warning("⚠️ No dataset found. Ensure 'bankbot_finial_expanded.csv' exists.")

    st.download_button(
        "⬇️ Export Training Data (CSV)",
        data=df_data.to_csv(index=False).encode("utf-8"),
        file_name="training_data_export.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------
# 💬 FAQ PAGE
# ------------------------------------------------------------
elif page == "💬 FAQs":
    st.subheader("💬 Manage Frequently Asked Questions")
    st.info("This section allows admins to add or edit question-answer pairs.")

    if responses:
        intents_list = list(responses.keys())
        selected_intent = st.selectbox("Select an intent", intents_list)
        st.write("### Sample Responses")
        for resp in responses[selected_intent][:3]:
            st.text(f"- {resp}")
    else:
        st.warning("⚠️ No intent-response mapping loaded.")

    st.text_input("Add New FAQ Question", "")
    st.text_area("Add FAQ Answer", "")
    st.button("➕ Add FAQ (Feature under development)")

# ------------------------------------------------------------
# 📊 ANALYTICS PAGE
# ------------------------------------------------------------
elif page == "📊 Analytics":
    st.subheader("📈 Intent & Confidence Analytics")

    if not df_logs.empty:
        st.write("### Intent Distribution")
        intent_counts = df_logs["intent"].value_counts()
        st.bar_chart(intent_counts)

        st.write("### Log Table")
        st.dataframe(df_logs.tail(15))
    else:
        st.warning("⚠️ No analytics available — chat logs not found.")

# ------------------------------------------------------------
# ⚙️ SETTINGS PAGE
# ------------------------------------------------------------
elif page == "⚙️ Settings":
    st.subheader("⚙️ System Settings")
    st.write("Here you can retrain models or reset data (future feature).")

    if st.button("🔄 Retrain Model"):
        st.info("Retraining will be added later in Milestone 5.")

    if st.button("🗑️ Clear Logs"):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            st.success("Chat logs cleared successfully.")
        else:
            st.warning("No chat logs found to delete.")

# ------------------------------------------------------------
# ✅ SIDEBAR FOOTER
# ------------------------------------------------------------
st.sidebar.success("✅ Admin Panel Loaded Successfully")
