import os
import random
import time
import re
import json
import logging
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wander")

# Import LangGraph agent
from agents.agent import build_graph, get_initial_state

# ── Name persistence ───────────────────────────────────────────────────────────
USER_FILE = os.path.join(os.path.dirname(__file__), "data", "user.json")

def load_saved_name() -> str | None:
    """Load saved user name from disk."""
    try:
        os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
        if os.path.exists(USER_FILE):
            with open(USER_FILE, "r") as f:
                data = json.load(f)
                return data.get("name")
    except Exception:
        pass
    return None

def save_name(name: str):
    """Persist user name to disk so it survives restarts."""
    try:
        os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
        with open(USER_FILE, "w") as f:
            json.dump({"name": name}, f)
    except Exception:
        pass

st.set_page_config(
    page_title="TripBuddy",
    page_icon="assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* ── TripBuddy Brand Kit ─────────────────────────────────────────────────
     Primary orange : #F97316
     Navy           : #1E3A5F
     Light orange   : #FEF3E2
     Font           : Poppins
  ── */

  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
  }

  #MainMenu, footer { visibility: hidden; }

  [data-testid="stHeader"],
  [data-testid="stDecoration"] {
    display: none !important;
    height: 0 !important;
  }

  /* ── Layout ── */
  .block-container {
    max-width: 860px;
    padding-top: 1.5rem;
    padding-bottom: 5rem;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.12);
    min-width: 240px !important;
    display: block !important;
  }
  [data-testid="stSidebarCollapsedControl"] {
    display: block !important;
  }

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
    align-items: flex-start !important;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] li,
  [data-testid="stChatMessage"] h1,
  [data-testid="stChatMessage"] h2,
  [data-testid="stChatMessage"] h3 {
    font-size: 16px !important;
    line-height: 1.7 !important;
    text-align: left !important;
    font-family: 'Poppins', sans-serif !important;
  }

  /* ── Chat input ── */
  [data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    font-size: 15px !important;
    font-family: 'Poppins', sans-serif !important;
    border-color: rgba(249,115,22,0.3) !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    border-color: #F97316 !important;
    box-shadow: 0 0 0 2px rgba(249,115,22,0.15) !important;
  }

  /* ── All buttons ── */
  div[data-testid="stButton"] > button {
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    font-family: 'Poppins', sans-serif !important;
    text-align: left !important;
    padding: 10px 16px !important;
    height: auto !important;
    white-space: normal !important;
    line-height: 1.5 !important;
    border: 1px solid rgba(249,115,22,0.25) !important;
    background: transparent !important;
    width: 100% !important;
    margin-bottom: 6px !important;
    transition: all 0.15s !important;
    color: inherit !important;
  }
  div[data-testid="stButton"] > button:hover {
    border-color: #F97316 !important;
    background: rgba(249,115,22,0.06) !important;
  }

  /* ── New trip button — orange filled ── */
  div[data-testid="stButton"] > button[kind="primary"],
  .primary-btn > div[data-testid="stButton"] > button {
    background: #F97316 !important;
    color: white !important;
    border-color: #F97316 !important;
    font-weight: 600 !important;
  }
  .primary-btn > div[data-testid="stButton"] > button:hover {
    background: #ea6c0a !important;
  }

  /* Left-align sidebar image */
  [data-testid="stSidebar"] [data-testid="stImage"] {
    display: flex !important;
    justify-content: flex-start !important;
  }

  /* Remove image border/background boxes */
  [data-testid="stImage"] > img {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }
  [data-testid="stImage"] {
    background: transparent !important;
  }

  /* Trip card text left aligned */
  .trip-card-btn > div[data-testid="stButton"] > button {
    border-radius: 14px !important;
    border: 1px solid rgba(249,115,22,0.2) !important;
    padding: 18px 20px !important;
    font-size: 15px !important;
    margin-bottom: 10px !important;
    text-align: left !important;
  }
  .trip-card-btn > div[data-testid="stButton"] > button:hover {
    border-color: #F97316 !important;
    background: rgba(249,115,22,0.05) !important;
  }

  /* ── Hero ── */
  .tb-hero {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
  }
  .tb-hero img {
    height: 80px;
    margin-bottom: 0.5rem;
  }
  .tb-hero p {
    font-size: 1rem;
    opacity: 0.5;
    margin: 0;
  }

  /* ── Divider accent ── */
  hr {
    border-color: rgba(249,115,22,0.15) !important;
  }
</style>
""", unsafe_allow_html=True)
# ── Helpers ────────────────────────────────────────────────────────────────────
def get_default_messages():
    return []
def time_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning ☀️"
    elif hour < 18:
        return "Good afternoon 🌤️"
    else:
        return "Good evening 🌙"
def _partial_title(agent):
    """Build progressive title from whatever is known so far."""
    destination = agent.get("destination", "").split(",")[0].strip()
    origin = (agent.get("origin") or "").split(",")[0].strip()
    transport = agent.get("transport", "")
    vibe = agent.get("vibe", "")
    budget = agent.get("budget")
    budget_type = agent.get("budget_type", "total")
    num_travelers = agent.get("num_travelers", 1)

    parts = []
    if origin:
        parts.append(f"{origin} to {destination}")
    else:
        parts.append(destination)
    if transport:
        parts.append(transport.lower())
    if vibe:
        vibes = [v.strip().lower() for v in vibe.split(",")]
        parts.append("for " + ", ".join(vibes))
    if budget:
        total = int(budget * num_travelers if budget_type == "per_person" else budget)
        parts.append(f"with a ${total:,} budget")
    return ", ".join(parts) if parts else destination


def generate_chat_title(messages):
    """Use LLM to summarize the full conversation into a short natural sentence."""
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    asst_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    if not user_msgs:
        return "New trip"

    # Build a short transcript for the LLM
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'TripBuddy'}: {m['content'][:120]}"
        for m in messages[-10:]
        if m["role"] in ("user", "assistant")
    )

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Summarize this travel planning conversation in ONE short natural sentence.
Include: origin → destination, transport, vibe/activities, budget.
Keep it under 15 words. Be specific. No quotes.
Example: "Tulsa to Nashville, driving, food and nightlife, $500 budget"
Example: "Stillwater to Jackson Heights, flying, all vibes, $800 budget"
Example: "Tulsa to Miami, flying, beach and culture, $700 budget" """
                },
                {"role": "user", "content": transcript}
            ],
            max_tokens=30,
            temperature=0
        )
        return r.choices[0].message.content.strip().strip('"').strip("'")
    except Exception:
        return _partial_title(st.session_state.get("agent_state", {}))
def extract_trip_info(messages):
    """
    Extract trip details from the full conversation.
    Uses LLM to read the whole conversation so it catches
    corrections like 'dallas instead of las vegas'.
    Only includes values that have been explicitly confirmed.
    """
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    asst_msgs = [m["content"] for m in messages if m["role"] == "assistant"]

    if not user_msgs:
        return {"destination": None, "origin": None, "days": None, "budget": None, "vibe": None}

    # Build a readable transcript for the LLM to analyze
    transcript_parts = []
    for msg in messages:
        if msg["role"] == "user":
            transcript_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            transcript_parts.append(f"Assistant: {msg['content']}")
    transcript = "\n".join(transcript_parts[-12:])  # last 12 messages

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Read this travel planning conversation and extract the CURRENT confirmed trip details.
Important: if the user corrected something (e.g. 'dallas instead of las vegas'), use the LATEST value.
Only include a value if it was clearly stated. If unsure, use null.

Return ONLY a JSON object:
{
  "destination": "most recent US city/destination mentioned, or null",
  "origin": "where they are traveling FROM (their home city), or null",
  "days": "number as integer or null",
  "budget": "dollar amount as string like '$600' or null",
  "vibe": "one of: culture, food, nightlife, outdoors, road trip, beach, mix — or null"
}"""
                },
                {"role": "user", "content": transcript}
            ],
            max_tokens=100, temperature=0
        )
        raw = r.choices[0].message.content.strip().replace("```json","").replace("```","")
        extracted = json.loads(raw)

        info = {
            "destination": extracted.get("destination", {}) or None,
            "origin":      extracted.get("origin") or None,
            "days":        f"{extracted['days']} days" if extracted.get("days") else None,
            "budget":      extracted.get("budget") or None,
            "vibe":        extracted.get("vibe", "").capitalize() or None,
        }
        # Clean up title case for city names
        if info["destination"]:
            info["destination"] = str(info["destination"]).title()
        if info["origin"]:
            info["origin"] = str(info["origin"]).title()
        return info

    except Exception:
        return {"destination": None, "origin": None, "days": None, "budget": None, "vibe": None}
# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = get_default_messages()
if "user_name" not in st.session_state:
    st.session_state.user_name = load_saved_name()
if "trip_info" not in st.session_state:
    st.session_state.trip_info = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
# LangGraph
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "agent_state" not in st.session_state:
    st.session_state.agent_state = get_initial_state()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "wander_001"
def save_current_chat():
    """Save current conversation to chat history."""
    msgs = st.session_state.messages
    if len(msgs) == 0:
        return

    agent = st.session_state.agent_state
    if not agent.get("destination"):
        return

    has_all = all([
        agent.get("destination"),
        agent.get("origin"),
        agent.get("transport"),
        agent.get("num_days"),
        agent.get("budget"),
        agent.get("vibe"),
    ])

    # Only show in sidebar once budget is confirmed
    has_budget = agent.get("destination") and agent.get("budget")
    if not has_budget:
        return

    chat_id = st.session_state.active_chat_id
    title = generate_chat_title(msgs) if has_all else _partial_title(agent)
    timestamp = datetime.now().strftime("%b %d, %I:%M %p")

    trip_info = {
        "destination": agent.get("destination"),
        "origin":      agent.get("origin"),
        "days":        f"{agent['num_days']} days" if agent.get("num_days") else None,
        "budget":      f"${int(agent['budget'])}" if agent.get("budget") else None,
        "vibe":        agent.get("vibe", "").capitalize() if agent.get("vibe") else None,
    }
    st.session_state.trip_info = trip_info

    if chat_id:
        # Update existing
        for chat in st.session_state.chat_history:
            if chat["id"] == chat_id:
                chat["title"] = title
                chat["messages"] = msgs.copy()
                chat["trip_info"] = trip_info
                chat["timestamp"] = timestamp
                break
    else:
        # Create new
        new_id = str(int(time.time()))
        st.session_state.active_chat_id = new_id
        st.session_state.chat_history.insert(0, {
            "id": new_id,
            "title": title,
            "messages": msgs.copy(),
            "trip_info": trip_info,
            "timestamp": timestamp
        })
def load_chat(chat_id):
    """Load a past chat into the current session."""
    for chat in st.session_state.chat_history:
        if chat["id"] == chat_id:
            st.session_state.messages = chat["messages"].copy()
            st.session_state.trip_info = chat["trip_info"]
            st.session_state.active_chat_id = chat_id
            break
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/icon.png", width=200)

    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    if st.button("＋  New trip", width="stretch", key="new_trip_btn"):
        save_current_chat()
        st.session_state.messages = get_default_messages()
        st.session_state.agent_state = get_initial_state()
        st.session_state.thread_id = f"wander_{int(time.time())}"
        st.session_state.trip_info = {}
        st.session_state.active_chat_id = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        current = [c for c in st.session_state.chat_history if c["id"] == st.session_state.active_chat_id]
        past = [c for c in st.session_state.chat_history if c["id"] != st.session_state.active_chat_id]

        if current:
            st.markdown("<div style='font-size:11px;font-weight:600;opacity:0.4;letter-spacing:0.8px;margin:8px 0 4px'>CURRENT TRIP</div>", unsafe_allow_html=True)
            chat = current[0]
            st.button(f"📍 {chat['title']}", key=f"chat_{chat['id']}", width="stretch")

        if past:
            st.markdown("<div style='font-size:11px;font-weight:600;opacity:0.4;letter-spacing:0.8px;margin:12px 0 4px'>PAST TRIPS</div>", unsafe_allow_html=True)
            for chat in past[:9]:
                if st.button(chat["title"], key=f"chat_{chat['id']}", width="stretch"):
                    save_current_chat()
                    load_chat(chat["id"])
                    st.rerun()

# ── Hero ───────────────────────────────────────────────────────────────────────
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.image("assets/logo.png", width="stretch")

st.divider()

# ── Name gate — blocks everything until name is entered ───────────────────────
if not st.session_state.user_name:
    with st.chat_message("assistant", avatar="assets/icon.png"):
        st.markdown(f"{time_greeting()}! Before we start — **what's your name?**")

    name_input = st.chat_input("Tell me your name to get started 👋")
    if name_input:
        raw = name_input.strip()
        # Handle "My name is X", "I'm X", "It's X", "Call me X"
        for pattern in [
            r"(?:my name is|i'm|i am|it's|its|call me|hi i'm|hey i'm)\s+([a-zA-Z]+)",
            r"^([a-zA-Z]+)$",           # just a name
            r"^([a-zA-Z]+)[,!\s]",      # name followed by punctuation
        ]:
            match = re.search(pattern, raw.lower())
            if match:
                name = match.group(1).capitalize()
                break
        else:
            name = raw.split()[0].capitalize()
        st.session_state.user_name = name
        save_name(name)
        st.rerun()
    st.stop()

# ── Suggested prompts (fresh session only) ─────────────────────────────────────
is_fresh = len(st.session_state.messages) == 0

if is_fresh:
    st.markdown("**✈️ Quick start — pick a trip or type your own:**")
    st.write("")

    prompts = [
        ("🎸", "Nashville, TN", "4 days", "$500", "I want to go to Nashville for 4 days with a $500 budget."),
        ("🌊", "Miami, FL",     "5 days", "$700", "I want to go to Miami for 5 days with a $700 budget."),
        ("🏜️", "Las Vegas, NV", "3 days", "$600", "I want to go to Las Vegas for 3 days with a $600 budget."),
    ]

    for flag, city, days, budget, prompt_text in prompts:
        st.markdown('<div class="trip-card-btn">', unsafe_allow_html=True)
        if st.button(
            f"{flag}  **{city}** — {days} · {budget}",
            key=f"p_{city}",
            width="stretch"
        ):
            st.session_state.messages.append({"role": "user", "content": prompt_text})
            st.session_state.agent_state["messages"].append({"role": "user", "content": prompt_text})

            with st.spinner("Planning your trip..."):
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = st.session_state.graph.invoke(
                        st.session_state.agent_state,
                        config=config
                    )
                    st.session_state.agent_state = result
                    asst_msgs = [m for m in result.get("messages", []) if m["role"] == "assistant"]
                    full_reply = asst_msgs[-1]["content"] if asst_msgs else "Let me help you plan that trip!"
                except Exception as e:
                    logger.error("Graph error: %s", e)
                    full_reply = "Something went wrong. Please try again."

            st.session_state.messages.append({"role": "assistant", "content": full_reply})
            st.session_state.trip_info = {
                "destination": st.session_state.agent_state.get("destination"),
                "origin":      st.session_state.agent_state.get("origin"),
                "days":        f"{st.session_state.agent_state['num_days']} days" if st.session_state.agent_state.get("num_days") else None,
                "budget":      f"${int(st.session_state.agent_state['budget'])}" if st.session_state.agent_state.get("budget") else None,
                "vibe":        st.session_state.agent_state.get("vibe", "").capitalize() if st.session_state.agent_state.get("vibe") else None,
            }
            save_current_chat()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    avatar = "assets/icon.png" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Opening message ────────────────────────────────────────────────────────────
if is_fresh:
    with st.chat_message("assistant", avatar="assets/icon.png"):
        name = st.session_state.user_name
        st.markdown(random.choice([
            f"{time_greeting()}, {name}! Where do you want to go?",
            f"Hey {name}! 🌍 What US trip is on your mind?",
            f"Ready for another adventure, {name}? 🎒",
            f"{time_greeting()}! ✈️ Where are we headed, {name}?",
        ]))

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your trip...")

if user_input:
    # Add to display messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Add to agent state and run graph
    st.session_state.agent_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="assets/icon.png"):
        placeholder = st.empty()
        placeholder.markdown("▌")

        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            prev_stage = st.session_state.agent_state.get("stage")
            result = st.session_state.graph.invoke(
                st.session_state.agent_state,
                config=config
            )
            st.session_state.agent_state = result

            # Pull latest assistant message
            asst_msgs = [m for m in result.get("messages", []) if m["role"] == "assistant"]
            full_reply = asst_msgs[-1]["content"] if asst_msgs else "Sorry, something went wrong."

            placeholder.markdown(full_reply)

            # If agent reset state (new destination mid-conv or post-itinerary)
            was_done = prev_stage == "done"
            is_resetting = (
                result.get("stage") == "clarifying" and
                result.get("destination") is None and
                result.get("itinerary") is None
            )
            if is_resetting:
                save_current_chat()
                # Reset to fresh state but run graph once more to process the new destination
                new_state = get_initial_state()
                new_state["messages"].append({"role": "user", "content": user_input})
                st.session_state.thread_id = f"trip_{int(time.time())}"
                config2 = {"configurable": {"thread_id": st.session_state.thread_id}}
                try:
                    result2 = st.session_state.graph.invoke(new_state, config=config2)
                    st.session_state.agent_state = result2
                    asst2 = [m for m in result2.get("messages", []) if m["role"] == "assistant"]
                    reply2 = asst2[-1]["content"] if asst2 else "Sure! Where do you want to go? 🌍"
                except Exception:
                    reply2 = "Sure! Where do you want to go? 🌍"
                    st.session_state.agent_state = get_initial_state()
                placeholder.markdown(reply2)
                full_reply = reply2
                st.session_state.messages = [{"role": "user", "content": user_input}, {"role": "assistant", "content": reply2}]
                st.session_state.trip_info = {}
                st.session_state.active_chat_id = None
                st.rerun()

            # Sync trip info from agent state to sidebar
            st.session_state.trip_info = {
                "destination": result.get("destination"),
                "origin":      result.get("origin"),
                "days":        f"{result['num_days']} days" if result.get("num_days") else None,
                "budget":      f"${int(result['budget'])}" if result.get("budget") else None,
                "vibe":        result.get("vibe", "").capitalize() if result.get("vibe") else None,
            }

        except Exception as e:
            import traceback
            logger.error("Graph error: %s\n%s", e, traceback.format_exc())
            full_reply = f"⚠️ Error: {str(e)}"
            placeholder.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
    save_current_chat()
    st.rerun()