"""
test_ap.py
------
Simple student travel planner chat.
Just OpenAI + Streamlit. No agents, no RAG, no MCP yet.

Run: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Wander 🎒", page_icon="🎒", layout="centered")

st.title("🎒 Wander")
st.caption("Student travel planner — tell me where you want to go")

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly student travel planner. "
                "Help college students plan affordable trips. "
                "Ask about their destination, budget, number of days, and vibe. "
                "Give practical, budget-conscious advice. Keep responses concise."
            )
        }
    ]

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Where do you want to go?")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner(""):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                max_tokens=500
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})