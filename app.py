"""
app.py — TripBuddy Streamlit frontend
"""

import os
import random
import time
import re
import json
import logging
import requests
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from agents.agent_new import build_graph, get_initial_state

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tripbuddy")

HEADERS = {"User-Agent": "TripBuddy/1.0 (student travel planner; educational use)"}


# ══════════════════════════════════════════════════════════════════════════════
# TRANSPORT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def geocode(city: str):
    try:
        time.sleep(1)
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{city}, USA", "format": "json", "limit": 1, "countrycodes": "us"},
            headers=HEADERS, timeout=8,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.warning("Geocode failed for %s: %s", city, e)
    return None


def get_route(origin: str, destination: str):
    o = geocode(origin)
    d = geocode(destination)
    if not o or not d:
        return None
    try:
        resp = requests.get(
            f"http://router.project-osrm.org/route/v1/driving/{o[1]},{o[0]};{d[1]},{d[0]}?overview=false",
            headers=HEADERS, timeout=10,
        )
        data = resp.json()
        if data.get("code") == "Ok":
            r = data["routes"][0]
            return {
                "miles": round(r["distance"] * 0.000621371, 1),
                "hours": round(r["duration"] / 3600, 1),
            }
    except Exception as e:
        logger.warning("OSRM failed: %s", e)
    return None


def get_gas_price():
    try:
        resp = requests.get(
            "https://api.eia.gov/v2/petroleum/pri/gnd/data/",
            params={
                "frequency": "weekly",
                "data[0]": "value",
                "facets[series][]": "EMM_EPM0_PTE_NUS_DPG",
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "length": 1,
                "api_key": "DEMO",
            },
            headers=HEADERS, timeout=8,
        )
        return float(resp.json()["response"]["data"][0]["value"])
    except Exception:
        return 3.50


def get_vehicle_efficiency(vehicle: str) -> dict:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"""What is the real-world fuel efficiency of a {vehicle}?
Return ONLY JSON: {{"type":"gas"|"electric"|"hybrid","mpg":number_or_null,"kwh_per_mile":number_or_null,"notes":"brief note"}}"""}],
            max_tokens=80, temperature=0,
        )
        raw = r.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        return json.loads(raw)
    except Exception:
        return {"type": "gas", "mpg": 28, "kwh_per_mile": None, "notes": "estimated"}


def estimate_parking(destination: str, nights: int = 0) -> dict:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            messages=[{"role": "user", "content":
                f"What is the average overnight parking garage cost per night in {destination}, USA? Give a realistic low and high price range."}],
            max_tokens=200, temperature=0,
        )
        content = r.choices[0].message.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        content = str(content or "")
        parse = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"Based on: '{content[:500]}'\nReturn ONLY JSON for {nights} nights: "
                "{\"per_night_low\":number,\"per_night_high\":number,\"estimated_total\":number,\"note\":\"one sentence source\"}"}],
            max_tokens=80, temperature=0,
        )
        raw = parse.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(raw)
        data["per_night"] = f"${data['per_night_low']}–${data['per_night_high']}"
        return data
    except Exception as e:
        logger.warning("Parking estimate failed: %s", e)
        return {"per_night": "$10–$30", "estimated_total": round(15 * nights), "note": "Estimate — verify locally"}


def estimate_tolls(origin: str, destination: str, miles: float) -> dict:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            messages=[{"role": "user", "content":
                f"What are the approximate road toll costs driving from {origin} to {destination}, USA ({miles} miles)?"}],
            max_tokens=200, temperature=0,
        )
        content = r.choices[0].message.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        content = str(content or "")
        parse = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"Based on: '{content[:500]}'\nReturn ONLY JSON: {{\"estimated_tolls\":number,\"note\":\"one sentence\"}}"}],
            max_tokens=60, temperature=0,
        )
        raw = parse.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        return json.loads(raw)
    except Exception as e:
        logger.warning("Tolls estimate failed: %s", e)
        return {"estimated_tolls": 0, "note": "Unable to estimate — verify with TollGuru"}


def calculate_road_trip(origin, destination, vehicle, num_travelers, num_nights=0):
    route = get_route(origin, destination)
    if not route:
        return None

    miles_rt   = route["miles"] * 2
    efficiency = get_vehicle_efficiency(vehicle)
    gas_price  = get_gas_price()

    if efficiency["type"] == "electric":
        kwh        = miles_rt * (efficiency.get("kwh_per_mile") or 0.30)
        fuel_total = round(kwh * 0.35, 2)
        fuel_info  = {"type": "electric", "kwh": round(kwh, 1), "cost_per_kwh": 0.35, "total": fuel_total}
    else:
        mpg        = efficiency.get("mpg") or 28
        gallons    = miles_rt / mpg
        fuel_total = round(gallons * gas_price, 2)
        fuel_info  = {"type": efficiency["type"], "mpg": mpg, "gallons": round(gallons, 1),
                      "gas_price": gas_price, "total": fuel_total}

    parking    = estimate_parking(destination, nights=num_nights)
    tolls      = estimate_tolls(origin, destination, route["miles"])
    toll_total = (tolls.get("estimated_tolls") or 0) * 2
    total      = round(fuel_total + parking["estimated_total"] + toll_total, 2)
    per_person = round(total / num_travelers, 2)

    return {
        "route":         route,
        "vehicle":       vehicle,
        "efficiency":    efficiency,
        "fuel":          fuel_info,
        "parking":       parking,
        "tolls":         {"total": toll_total, "note": tolls.get("note", "")},
        "total":         total,
        "per_person":    per_person,
        "num_travelers": num_travelers,
    }


def render_transport_card(calc: dict):
    if not calc:
        return
    route = calc["route"]
    fuel  = calc["fuel"]
    is_ev = fuel["type"] == "electric"
    st.markdown(f"""
<div style="border:1px solid rgba(249,115,22,0.3);border-radius:14px;padding:16px 20px;margin:12px 0;background:rgba(249,115,22,0.04)">
<div style="font-weight:600;font-size:15px;margin-bottom:10px">🚗 Road Trip Cost Breakdown</div>
<div style="font-size:14px;line-height:2">
  📍 <b>{route['miles']} miles</b> each way &nbsp;·&nbsp; ~{route['hours']} hrs drive<br>
  {'⚡' if is_ev else '⛽'} {'<b>EV charging:</b> ' + str(fuel['kwh']) + ' kWh × $0.35 = <b>$' + str(fuel['total']) + '</b>' if is_ev else '<b>Gas:</b> ' + str(fuel['gallons']) + ' gal × $' + str(fuel['gas_price']) + '/gal = <b>$' + str(fuel['total']) + '</b>'}<br>
  🅿️ <b>Parking:</b> {calc['parking']['per_night']}/night · est. <b>${calc['parking']['estimated_total']}</b><br>
  {'🛣️ <b>Tolls:</b> est. $' + str(calc['tolls']['total']) + ' round trip<br>' if calc['tolls']['total'] > 0 else ''}
  <hr style="margin:8px 0;border-color:rgba(249,115,22,0.15)">
  💰 <b>Total transport: ${calc['total']}</b> &nbsp;·&nbsp; <b>${calc['per_person']}/person</b> (split {calc['num_travelers']} ways)
</div>
<div style="font-size:11px;opacity:0.45;margin-top:6px">Gas price from US EIA weekly average. Parking & tolls are estimates.</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# USER PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
USER_FILE  = os.path.join(os.path.dirname(__file__), "data", "user.json")
CHATS_FILE  = os.path.join(os.path.dirname(__file__), "data", "chats.json")
DRAFT_FILE  = os.path.join(os.path.dirname(__file__), "data", "draft.json")

def load_saved_name() -> str | None:
    try:
        os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
        if os.path.exists(USER_FILE):
            with open(USER_FILE, "r") as f:
                return json.load(f).get("name")
    except Exception:
        pass
    return None

def save_name(name: str):
    try:
        os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
        with open(USER_FILE, "w") as f:
            json.dump({"name": name}, f)
    except Exception:
        pass


def load_chats() -> list:
    """Load persisted chat history from disk."""
    try:
        os.makedirs(os.path.dirname(CHATS_FILE), exist_ok=True)
        if os.path.exists(CHATS_FILE):
            with open(CHATS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_chats(history: list):
    """Persist chat history to disk so it survives page refreshes and server restarts."""
    try:
        os.makedirs(os.path.dirname(CHATS_FILE), exist_ok=True)
        # Keep at most 20 chats to avoid unbounded growth
        with open(CHATS_FILE, "w") as f:
            json.dump(history[:20], f, default=str)
    except Exception:
        pass


def _load_draft() -> dict | None:
    """Load the in-progress draft session from disk, if one exists."""
    try:
        if os.path.exists(DRAFT_FILE):
            with open(DRAFT_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _clear_draft():
    """Delete the draft file (called when user starts a fresh trip)."""
    try:
        if os.path.exists(DRAFT_FILE):
            os.remove(DRAFT_FILE)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TripBuddy",
    page_icon="assets/icon.png",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
  #MainMenu, footer { visibility: hidden; }
  [data-testid="stDecoration"] { display: none !important; }
  .block-container { max-width: 860px; padding-top: 1.5rem; padding-bottom: 5rem; }
  [data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 0.5rem 0 !important; align-items: flex-start !important;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] li,
  [data-testid="stChatMessage"] h1,
  [data-testid="stChatMessage"] h2,
  [data-testid="stChatMessage"] h3 {
    font-size: 16px !important; line-height: 1.7 !important;
    text-align: left !important; font-family: 'Poppins', sans-serif !important;
  }
  [data-testid="stChatInput"] textarea {
    border-radius: 14px !important; font-size: 15px !important;
    font-family: 'Poppins', sans-serif !important;
    border-color: rgba(249,115,22,0.3) !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    border-color: #F97316 !important;
    box-shadow: 0 0 0 2px rgba(249,115,22,0.15) !important;
  }
  div[data-testid="stButton"] > button {
    border-radius: 12px !important; font-size: 14px !important;
    font-weight: 500 !important; font-family: 'Poppins', sans-serif !important;
    text-align: left !important; padding: 10px 16px !important;
    height: auto !important; white-space: normal !important;
    line-height: 1.5 !important; border: 1px solid rgba(249,115,22,0.25) !important;
    background: transparent !important; width: 100% !important;
    margin-bottom: 6px !important; transition: all 0.15s !important;
    color: inherit !important;
  }
  div[data-testid="stButton"] > button:hover {
    border-color: #F97316 !important; background: rgba(249,115,22,0.06) !important;
  }
  div[data-testid="stButton"] > button[kind="primary"],
  .primary-btn > div[data-testid="stButton"] > button {
    background: #F97316 !important; color: white !important;
    border-color: #F97316 !important; font-weight: 600 !important;
  }
  .primary-btn > div[data-testid="stButton"] > button:hover { background: #ea6c0a !important; }
  [data-testid="stSidebar"] [data-testid="stImage"] {
    display: flex !important; justify-content: flex-start !important;
  }
  [data-testid="stImage"] > img { background: transparent !important; border: none !important; box-shadow: none !important; }
  [data-testid="stImage"] { background: transparent !important; }
  .trip-card-btn > div[data-testid="stButton"] > button {
    border-radius: 14px !important; border: 1px solid rgba(249,115,22,0.2) !important;
    padding: 18px 20px !important; font-size: 15px !important;
    margin-bottom: 10px !important; text-align: left !important;
  }
  .trip-card-btn > div[data-testid="stButton"] > button:hover {
    border-color: #F97316 !important; background: rgba(249,115,22,0.05) !important;
  }
  hr { border-color: rgba(249,115,22,0.15) !important; }

  /* ── Checklist checkboxes ── */
  [data-testid="stCheckbox"] { padding: 2px 0 !important; }
  [data-testid="stCheckbox"] label {
    font-size: 14px !important;
    font-family: 'Poppins', sans-serif !important;
    cursor: pointer !important;
  }
  [data-testid="stCheckbox"] input:checked + div { color: #F97316 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_default_messages():
    return []

def safe_markdown(text: str):
    """Escape bare dollar signs to prevent Streamlit LaTeX rendering."""
    escaped = re.sub(r'(?<!\\)\$', r'\\$', text)
    st.markdown(escaped)

def time_greeting():
    hour = datetime.now().hour
    if hour < 12:   return "Good morning ☀️"
    elif hour < 18: return "Good afternoon 🌤️"
    else:           return "Good evening 🌙"

def _null_guard(val: str | None) -> str:
    """Return empty string if value is None or the literal string 'null'/'none'."""
    if not val or str(val).strip().lower() in ("null", "none", "n/a"):
        return ""
    return val

def _partial_title(agent: dict) -> str:
    destination   = _null_guard(agent.get("destination", "")).split(",")[0].strip()
    origin        = _null_guard(agent.get("origin") or "").split(",")[0].strip()
    transport     = _null_guard(agent.get("transport", ""))
    vibe          = _null_guard(agent.get("vibe", ""))
    budget        = agent.get("budget")
    budget_type   = agent.get("budget_type", "total")
    num_travelers = agent.get("num_travelers", 1)
    parts = []
    if origin: parts.append(f"{origin} to {destination}")
    else:      parts.append(destination)
    if transport: parts.append(transport.lower())
    if vibe:
        vibes = [v.strip().lower() for v in vibe.split(",")]
        parts.append("for " + ", ".join(vibes))
    if budget:
        total = int(budget * num_travelers if budget_type == "per_person" else budget)
        parts.append(f"with a ${total:,} budget")
    return ", ".join(parts) if parts else destination

def generate_chat_title(messages: list) -> str:
    agent         = st.session_state.get("agent_state", {})
    destination   = _null_guard(agent.get("destination", ""))
    origin        = _null_guard(agent.get("origin", ""))
    transport     = _null_guard(agent.get("transport", ""))
    vibe          = _null_guard(agent.get("vibe", ""))
    budget        = agent.get("budget")
    budget_type   = agent.get("budget_type", "total")
    num_travelers = agent.get("num_travelers", 1)
    if not destination:
        return "New trip"
    total = int(budget * num_travelers if budget_type == "per_person" else budget) if budget else None
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""Write a short natural summary of this trip in under 12 words.
Use ONLY these exact values — do not change or replace any names:
- Origin: {origin if origin else "unknown"}
- Destination: {destination}
- Transport: {transport if transport else "unknown"}
- Vibe: {vibe if vibe else "mix"}
- Budget: ${total} total for {num_travelers} people
If origin is unknown, just start with the destination.
Example: "Stillwater to Las Vegas, flying, culture & nightlife, $600"
Return only the summary, no quotes."""}],
            max_tokens=25, temperature=0,
        )
        return r.choices[0].message.content.strip().strip('"').strip("'")
    except Exception:
        if origin: return f"{origin.split(',')[0]} to {destination.split(',')[0]}"
        return destination.split(",")[0]

def extract_trip_info(messages: list) -> dict:
    empty = {"destination": None, "origin": None, "days": None, "budget": None, "vibe": None}
    if not any(m["role"] == "user" for m in messages):
        return empty
    transcript_parts = []
    for msg in messages:
        if msg["role"] == "user":        transcript_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant": transcript_parts.append(f"Assistant: {msg['content']}")
    transcript = "\n".join(transcript_parts[-12:])
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Read this travel planning conversation and extract the CURRENT confirmed trip details.
If the user corrected something, use the LATEST value. Only include if clearly stated.
Return ONLY JSON:
{
  "destination": "most recent US city/destination or null",
  "origin": "where they travel FROM or null",
  "days": "number as integer or null",
  "budget": "dollar amount like '$600' or null",
  "vibe": "culture, food, nightlife, outdoors, road trip, beach, mix — or null"
}"""},
                {"role": "user", "content": transcript},
            ],
            max_tokens=100, temperature=0,
        )
        raw = r.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        extracted = json.loads(raw)
        info = {
            "destination": extracted.get("destination") or None,
            "origin":      extracted.get("origin") or None,
            "days":        f"{extracted['days']} days" if extracted.get("days") else None,
            "budget":      extracted.get("budget") or None,
            "vibe":        extracted.get("vibe", "").capitalize() or None,
        }
        if info["destination"]: info["destination"] = str(info["destination"]).title()
        if info["origin"]:      info["origin"]      = str(info["origin"]).title()
        return info
    except Exception:
        return empty


# ══════════════════════════════════════════════════════════════════════════════
# GROCERY & CHECKLIST RENDERERS
# ══════════════════════════════════════════════════════════════════════════════
def render_grocery_form(agent: dict):
    """Inline form to collect traveler names + dietary info before generating grocery list."""
    num_travelers = agent.get("num_travelers", 1)

    st.markdown("""
<div style="border:1px solid rgba(249,115,22,0.3);border-radius:14px;padding:18px 20px;margin:12px 0">
<div style="font-weight:600;font-size:15px;margin-bottom:4px">🛒 Customize Your Grocery List</div>
<div style="font-size:13px;opacity:0.6;margin-bottom:14px">Tell us about your group so we can personalize the list.</div>
""", unsafe_allow_html=True)

    budget_mode = st.radio(
        "Budget type",
        ["Whole group", "Per person"],
        key="grocery_budget_mode_radio",
        horizontal=True,
    )

    st.markdown(f"**Traveler names & dietary needs** *(optional but helpful)*")
    travelers_info = []
    for i in range(num_travelers):
        c1, c2 = st.columns([1, 2])
        with c1:
            name = st.text_input(
                f"Traveler {i+1} name",
                key=f"g_name_{i}",
                placeholder=f"Traveler {i+1}",
                label_visibility="collapsed",
            )
        with c2:
            diet = st.text_input(
                f"Dietary restrictions",
                key=f"g_diet_{i}",
                placeholder="e.g. vegetarian, gluten-free, no nuts",
                label_visibility="collapsed",
            )
        travelers_info.append({
            "name": name.strip() if name.strip() else f"Traveler {i+1}",
            "diet": diet.strip() or None,
        })

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🛒 Build My List", key="grocery_submit_btn", use_container_width=True):
        st.session_state.grocery_travelers_info = travelers_info
        st.session_state.grocery_budget_mode    = "per_person" if budget_mode == "Per person" else "group"
        st.session_state.grocery_form_done      = True
        st.rerun()

def render_grocery_list(grocery_data: dict, agent: dict):
    """Render the generated grocery list with prices and prep tips."""
    budget_mode = grocery_data.get("budget_mode", "group")
    total       = grocery_data.get("total_cost", "?")
    per_person  = grocery_data.get("cost_per_person", "?")

    st.markdown(
        f'<div style="border:1px solid rgba(249,115,22,0.3);border-radius:14px;padding:16px 20px;margin:12px 0">'
        f'<div style="font-weight:600;font-size:15px;margin-bottom:6px">🛒 Grocery & Snacks List</div>'
        f'<div style="font-size:13px;opacity:0.6;margin-bottom:12px">'
        f'Estimated total: <b>${total}</b>'
        f'{"&nbsp;·&nbsp;$" + str(per_person) + "/person" if budget_mode == "per_person" else ""}'
        f'&nbsp;·&nbsp;<i>Prices approximate — verify at store</i></div>',
        unsafe_allow_html=True,
    )
    for cat in grocery_data.get("categories", []):
        st.markdown(f"**{cat['name']}**")
        for item in cat.get("items", []):
            src   = "" if item.get("source") == "walmart" else " *(est.)*"
            for_  = f" — *for {item['for']}*" if item.get("for") and item["for"] != "everyone" else ""
            st.markdown(f"- {item['name']} ({item['quantity']}){for_} — **${item['price']:.2f}**{src}")
    if grocery_data.get("prep_tips"):
        st.markdown("**💡 Prep tips:**")
        for tip in grocery_data["prep_tips"]:
            st.markdown(f"- {tip}")
    st.markdown("</div>", unsafe_allow_html=True)

def render_checklist(checklist_data: dict):
    """Render interactive checklist — clean card with progress bar, section counts, no emoji clutter."""
    sections      = checklist_data.get("sections", [])
    total_items   = sum(len(s.get("items", [])) for s in sections)
    checked_count = sum(
        1 for s in sections
        for j in range(len(s.get("items", [])))
        if st.session_state.get(f"chk_{s['name']}_{j}", False)
    )
    pct = int(checked_count / total_items * 100) if total_items > 0 else 0

    # Header + progress bar
    st.markdown(f"""
<div style="border:1px solid rgba(249,115,22,0.3);border-radius:16px;
            padding:18px 22px 6px;margin:14px 0 2px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <span style="font-weight:600;font-size:15px">Travel Checklist</span>
    <span style="font-size:13px;font-weight:600;color:#F97316">{checked_count} / {total_items} packed</span>
  </div>
  <div style="background:rgba(249,115,22,0.12);border-radius:99px;height:5px;margin-bottom:6px">
    <div style="background:#F97316;height:5px;border-radius:99px;
                width:{pct}%;transition:width 0.3s ease"></div>
  </div>
</div>""", unsafe_allow_html=True)

    for sec in sections:
        sec_name = sec["name"]
        items    = sec.get("items", [])
        sec_done = sum(1 for j in range(len(items)) if st.session_state.get(f"chk_{sec_name}_{j}", False))

        # Section label — small caps with count
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:0.9px;opacity:0.4;'
            f'text-transform:uppercase;margin:18px 0 4px 2px">'
            f'{sec_name}&nbsp;&nbsp;'
            f'<span style="font-weight:400">{sec_done}/{len(items)}</span></div>',
            unsafe_allow_html=True,
        )

        for j, item in enumerate(items):
            key       = f"chk_{sec_name}_{j}"
            note      = item.get("note", "")
            # Fallback: if item name is missing use the note, else "Item {j+1}"
            item_name = (item.get("item") or "").strip() or note or f"Item {j + 1}"
            st.checkbox(item_name, key=key)
            # Only show note separately if it adds info beyond what's in the label
            if note and note != item_name:
                st.markdown(
                    f'<div style="font-size:12px;opacity:0.45;margin:-10px 0 4px 28px">{note}</div>',
                    unsafe_allow_html=True,
                )

    tips = checklist_data.get("trip_specific_tips", [])
    if tips:
        st.markdown(
            '<div style="border-top:1px solid rgba(249,115,22,0.12);margin-top:14px;'
            'padding:12px 0 4px;font-size:11px;font-weight:700;letter-spacing:0.7px;'
            'opacity:0.4;text-transform:uppercase;margin-bottom:6px">Trip Tips</div>',
            unsafe_allow_html=True,
        )
        for tip in tips:
            st.markdown(
                f'<div style="font-size:13px;opacity:0.75;margin-bottom:5px">• {tip}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
# ── Restore in-progress draft from disk (survives page refresh) ──────────────
_draft = _load_draft()
if _draft and _draft.get("agent_state", {}).get("destination"):
    if "messages"    not in st.session_state:
        st.session_state.messages    = _draft.get("messages", [])
    if "agent_state" not in st.session_state:
        _restored = get_initial_state()
        _restored.update(_draft["agent_state"])
        _restored["messages"] = st.session_state.get("messages", [])
        st.session_state.agent_state = _restored
# ─────────────────────────────────────────────────────────────────────────────

if "messages"               not in st.session_state: st.session_state.messages               = get_default_messages()
if "user_name"              not in st.session_state: st.session_state.user_name              = load_saved_name()
if "trip_info"              not in st.session_state: st.session_state.trip_info              = {}
if "chat_history"           not in st.session_state: st.session_state.chat_history           = []
if "active_chat_id"         not in st.session_state: st.session_state.active_chat_id         = None
if "graph"                  not in st.session_state: st.session_state.graph                  = build_graph()
if "agent_state"            not in st.session_state: st.session_state.agent_state            = get_initial_state()
if "thread_id"              not in st.session_state: st.session_state.thread_id              = "tripbuddy_001"
if "transport_calc"         not in st.session_state: st.session_state.transport_calc         = None
if "transport_calc_key"     not in st.session_state: st.session_state.transport_calc_key     = None
# Grocery state
if "show_grocery"           not in st.session_state: st.session_state.show_grocery           = False
if "grocery_form_done"      not in st.session_state: st.session_state.grocery_form_done      = False
if "grocery_data"           not in st.session_state: st.session_state.grocery_data           = None
if "grocery_travelers_info" not in st.session_state: st.session_state.grocery_travelers_info = None
if "grocery_budget_mode"    not in st.session_state: st.session_state.grocery_budget_mode    = "group"
# Checklist state
if "show_checklist"         not in st.session_state: st.session_state.show_checklist         = False
if "checklist_data"         not in st.session_state: st.session_state.checklist_data         = None
if "budget_guardrail"       not in st.session_state: st.session_state.budget_guardrail       = None
if "_last_itinerary"        not in st.session_state: st.session_state._last_itinerary        = None
if "greeting"               not in st.session_state: st.session_state.greeting               = None


# ══════════════════════════════════════════════════════════════════════════════
# CHAT PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
def _serialise_agent_state(agent: dict) -> dict:
    """Return a JSON-safe copy of agent_state (drop non-serialisable values)."""
    safe = {}
    for k, v in agent.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            pass
    return safe

def _save_draft(agent: dict, msgs: list):
    """Persist the in-progress session to disk so state survives a page refresh,
    even before budget is set. Cleared when a new trip starts."""
    if not agent.get("destination"):
        return
    try:
        os.makedirs(os.path.dirname(DRAFT_FILE), exist_ok=True)
        with open(DRAFT_FILE, "w") as f:
            json.dump(
                {"agent_state": _serialise_agent_state(agent), "messages": msgs},
                f, default=str,
            )
    except Exception:
        pass

def save_current_chat():
    msgs  = st.session_state.messages
    agent = st.session_state.agent_state
    # Sidebar entries require destination + budget (this is a budget planner).
    # Draft state is saved separately below to handle mid-clarification refreshes.
    if not msgs or not agent.get("destination") or not agent.get("budget"):
        _save_draft(agent, msgs)   # always persist draft when destination known
        return
    _save_draft(agent, msgs)
    has_all = all([
        agent.get("destination"), agent.get("origin"), agent.get("transport"),
        agent.get("num_days"), agent.get("budget"), agent.get("vibe"),
    ])
    # Use LLM title only when all fields known; partial title otherwise
    # (avoids LLM call mid-clarification and handles no-budget gracefully)
    title = generate_chat_title(msgs) if has_all else _partial_title(agent)
    if not title or title == "New trip":
        title = _partial_title(agent) or agent.get("destination", "New trip")
    timestamp = datetime.now().strftime("%b %d, %I:%M %p")
    trip_info = {
        "destination": agent.get("destination"),
        "origin":      agent.get("origin"),
        "days":        f"{agent['num_days']} days" if agent.get("num_days") else None,
        "budget":      f"${int(agent['budget'])}"  if agent.get("budget")   else None,
        "vibe":        agent.get("vibe", "").capitalize() if agent.get("vibe") else None,
    }
    st.session_state.trip_info = trip_info
    chat_id = st.session_state.active_chat_id
    agent_snapshot = _serialise_agent_state(agent)
    if chat_id:
        for chat in st.session_state.chat_history:
            if chat["id"] == chat_id:
                chat.update({"title": title, "messages": msgs.copy(),
                             "trip_info": trip_info, "timestamp": timestamp,
                             "agent_state": agent_snapshot})
                break
    else:
        new_id = str(int(time.time()))
        st.session_state.active_chat_id = new_id
        st.session_state.chat_history.insert(0, {
            "id": new_id, "title": title, "messages": msgs.copy(),
            "trip_info": trip_info, "timestamp": timestamp,
            "agent_state": agent_snapshot,
        })
    save_chats(st.session_state.chat_history)

def load_chat(chat_id: str):
    for chat in st.session_state.chat_history:
        if chat["id"] == chat_id:
            st.session_state.messages       = chat["messages"].copy()
            st.session_state.trip_info      = chat["trip_info"]
            st.session_state.active_chat_id = chat_id
            # Restore agent state so the graph has full context on the next message
            if chat.get("agent_state"):
                restored = get_initial_state()
                restored.update(chat["agent_state"])
                st.session_state.agent_state = restored
                # Re-inject messages into agent state so clarify/revise nodes see history
                restored["messages"] = chat["messages"].copy()
            break


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("assets/icon.png", width=80)
    st.divider()
    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    if st.button("＋  New trip", use_container_width=True, key="new_trip_btn"):
        save_current_chat()
        _clear_draft()
        st.session_state.messages               = get_default_messages()
        st.session_state.agent_state            = get_initial_state()
        st.session_state.thread_id              = f"tripbuddy_{int(time.time())}"
        st.session_state.trip_info              = {}
        st.session_state.active_chat_id         = None
        st.session_state.show_grocery           = False
        st.session_state.grocery_form_done      = False
        st.session_state.grocery_data           = None
        st.session_state.grocery_travelers_info = None
        st.session_state.grocery_budget_mode    = "group"
        st.session_state.show_checklist         = False
        st.session_state.checklist_data         = None
        st.session_state.budget_guardrail       = None
        st.session_state.greeting               = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        current = [c for c in st.session_state.chat_history if c["id"] == st.session_state.active_chat_id]
        past    = [c for c in st.session_state.chat_history if c["id"] != st.session_state.active_chat_id]
        if current:
            st.markdown("<div style='font-size:11px;font-weight:600;opacity:0.4;letter-spacing:0.8px;margin:8px 0 4px'>CURRENT TRIP</div>", unsafe_allow_html=True)
            st.button(f"📍 {current[0]['title']}", key=f"chat_{current[0]['id']}", use_container_width=True)
        if past:
            st.markdown("<div style='font-size:11px;font-weight:600;opacity:0.4;letter-spacing:0.8px;margin:12px 0 4px'>PAST TRIPS</div>", unsafe_allow_html=True)
            for chat in past[:9]:
                if st.button(chat["title"], key=f"chat_{chat['id']}", use_container_width=True):
                    save_current_chat()
                    load_chat(chat["id"])
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.image("assets/logo.png", use_container_width=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# NAME GATE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.user_name:
    with st.chat_message("assistant", avatar="assets/icon.png"):
        st.markdown(f"{time_greeting()}! Before we start — **what's your name?**")
    name_input = st.chat_input("Tell me your name to get started 👋")
    if name_input:
        raw = name_input.strip()
        for pattern in [
            r"(?:my name is|i'm|i am|it's|its|call me|hi i'm|hey i'm)\s+([a-zA-Z]+)",
            r"^([a-zA-Z]+)$",
            r"^([a-zA-Z]+)[,!\s]",
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


# ══════════════════════════════════════════════════════════════════════════════
# QUICK-START CARDS (fresh session only)
# ══════════════════════════════════════════════════════════════════════════════
is_fresh = len(st.session_state.messages) == 0

if is_fresh:
    with st.chat_message("assistant", avatar="assets/icon.png"):
        # name is loaded from data/user.json via load_saved_name() → st.session_state.user_name
        name = st.session_state.user_name
        if not st.session_state.greeting:
            has_past_trips = len(st.session_state.chat_history) > 0
            if name and has_past_trips:
                options = [
                    f"Welcome back, {name}! Ready for the next adventure? 🌍",
                    f"{time_greeting()}, {name}! Where are we headed this time? ✈️",
                    f"Good to see you again, {name}! What's the next trip? 🗺️",
                ]
            elif name:
                options = [
                    f"{time_greeting()}, {name}! Where do you want to go?",
                    f"Hey {name}! 🌍 What US trip is on your mind?",
                    f"{time_greeting()}, {name}! ✈️ Tell me where you're headed.",
                ]
            else:
                options = [
                    f"{time_greeting()}! Where do you want to go? 🌍",
                    f"Hey there! ✈️ What US trip are you planning?",
                ]
            st.session_state.greeting = random.choice(options)
        st.markdown(st.session_state.greeting)
    st.write("")

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
        if st.button(f"{flag}  **{city}** — {days} · {budget}", key=f"p_{city}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": prompt_text})
            st.session_state.agent_state["messages"].append({"role": "user", "content": prompt_text})
            with st.spinner("Planning your trip..."):
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = st.session_state.graph.invoke(st.session_state.agent_state, config=config)
                    st.session_state.agent_state = result
                    asst_msgs  = [m for m in result.get("messages", []) if m["role"] == "assistant"]
                    full_reply = asst_msgs[-1]["content"] if asst_msgs else "Let me help you plan that trip!"
                except Exception as e:
                    logger.error("Graph error: %s", e)
                    full_reply = "Something went wrong. Please try again."
            st.session_state.messages.append({"role": "assistant", "content": full_reply})
            st.session_state.trip_info = {
                "destination": st.session_state.agent_state.get("destination"),
                "origin":      st.session_state.agent_state.get("origin"),
                "days":        f"{st.session_state.agent_state['num_days']} days" if st.session_state.agent_state.get("num_days") else None,
                "budget":      f"${int(st.session_state.agent_state['budget'])}"  if st.session_state.agent_state.get("budget")   else None,
                "vibe":        st.session_state.agent_state.get("vibe", "").capitalize() if st.session_state.agent_state.get("vibe") else None,
            }
            save_current_chat()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "system":
        # Render trip dividers; skip all other system messages
        if msg.get("content") == "__TRIP_DIVIDER__":
            st.markdown("""
<div style="display:flex;align-items:center;margin:24px 0 18px;gap:12px">
  <div style="flex:1;height:1px;background:rgba(249,115,22,0.25)"></div>
  <div style="font-size:12px;font-weight:600;color:#F97316;letter-spacing:0.6px;white-space:nowrap">✈️ NEW TRIP</div>
  <div style="flex:1;height:1px;background:rgba(249,115,22,0.25)"></div>
</div>""", unsafe_allow_html=True)
        continue
    avatar = "assets/icon.png" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        safe_markdown(msg["content"])

    is_last_msg = i == len(st.session_state.messages) - 1
    is_done     = st.session_state.agent_state.get("stage") == "done"
    _agent      = st.session_state.agent_state

    # Transport card
    if (is_last_msg and msg["role"] == "assistant" and is_done
            and _agent.get("transport") == "driving"
            and st.session_state.get("transport_calc")):
        render_transport_card(st.session_state.transport_calc)

    # Budget guardrail card removed — warnings shown in-chat before the itinerary.
    # ── Grocery & Checklist section ─────────────────────────────────────────
    if is_last_msg and msg["role"] == "assistant" and is_done:

        grocery_done   = st.session_state.grocery_data is not None
        checklist_done = st.session_state.checklist_data is not None

        # Initial buttons — only show buttons for features not yet open/generated
        _show_grocery_btn   = not grocery_done and not st.session_state.show_grocery
        _show_checklist_btn = not checklist_done and not st.session_state.show_checklist

        if _show_grocery_btn or _show_checklist_btn:
            _c1, _c2 = st.columns(2)
            if _show_grocery_btn:
                with _c1:
                    if st.button("🛒 Grocery & Snacks List", key="grocery_btn", use_container_width=True):
                        st.session_state.show_grocery   = True
                        st.session_state.show_checklist = False
                        st.rerun()
            if _show_checklist_btn:
                with _c2:
                    if st.button("✅ Travel Checklist", key="checklist_btn", use_container_width=True):
                        st.session_state.show_checklist = True
                        st.session_state.show_grocery   = False
                        st.rerun()

        # ── Grocery flow ───────────────────────────────────────────────────
        if st.session_state.show_grocery and not grocery_done:
            if not st.session_state.grocery_form_done:
                render_grocery_form(_agent)
            else:
                with st.spinner("Building your personalized grocery list..."):
                    try:
                        from agents.agent import generate_grocery_list
                        _budget      = _agent.get("budget", 0)
                        _btype       = _agent.get("budget_type", "total")
                        _ntrav       = _agent.get("num_travelers", 1)
                        _total       = _budget * _ntrav if _btype == "per_person" else _budget
                        st.session_state.grocery_data = generate_grocery_list(
                            destination      = _agent.get("destination", ""),
                            num_days         = _agent.get("num_days", 3),
                            num_travelers    = _ntrav,
                            transport        = _agent.get("transport", "flying"),
                            vibe             = _agent.get("vibe", "mix"),
                            accommodation    = _agent.get("accommodation_pref", "budget hotel"),
                            budget_remaining = _total * 0.15,
                            travelers_info   = st.session_state.grocery_travelers_info,
                            budget_mode      = st.session_state.grocery_budget_mode,
                        )
                    except Exception as _ge:
                        st.error(f"Could not generate grocery list: {_ge}")
                st.rerun()

        if grocery_done:
            render_grocery_list(st.session_state.grocery_data, _agent)
            # Show checklist button below grocery list if not yet generated
            if not checklist_done:
                st.write("")
                if st.button("✅ Travel Checklist", key="checklist_after_grocery", use_container_width=True):
                    st.session_state.show_checklist = True
                    st.rerun()

        # ── Checklist flow ─────────────────────────────────────────────────
        if st.session_state.show_checklist and not checklist_done:
            with st.spinner("Building your checklist..."):
                try:
                    from agents.agent import generate_travel_checklist
                    st.session_state.checklist_data = generate_travel_checklist(
                        destination   = _agent.get("destination", ""),
                        num_days      = _agent.get("num_days", 3),
                        transport     = _agent.get("transport", "flying"),
                        vibe          = _agent.get("vibe", "mix"),
                        accommodation = _agent.get("accommodation_pref", "budget hotel"),
                        num_travelers = _agent.get("num_travelers", 1),
                    )
                except Exception as _ce:
                    st.error(f"Could not generate checklist: {_ce}")
            st.rerun()

        if checklist_done:
            render_checklist(st.session_state.checklist_data)
            # Show grocery button below checklist if not yet generated
            if not grocery_done:
                st.write("")
                if st.button("🛒 Grocery & Snacks List", key="grocery_after_checklist", use_container_width=True):
                    st.session_state.show_grocery = True
                    st.rerun()





# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT & AGENT INVOCATION
# ══════════════════════════════════════════════════════════════════════════════
user_input = st.chat_input("Ask anything about your trip...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Pre-calculate transport cost before graph runs
    agent = st.session_state.agent_state
    if (
        agent.get("transport") == "driving"
        and agent.get("vehicle") and agent.get("origin")
        and agent.get("destination") and agent.get("num_travelers")
        and not agent.get("transport_cost")
    ):
        calc_key = f"{agent['origin']}_{agent['destination']}_{agent.get('num_travelers',1)}"
        if st.session_state.get("transport_calc_key") != calc_key:
            with st.spinner("Calculating road trip costs..."):
                calc = calculate_road_trip(
                    origin        = agent["origin"],
                    destination   = agent["destination"],
                    vehicle       = agent["vehicle"],
                    num_travelers = agent["num_travelers"],
                    num_nights    = (agent.get("num_days") or 1) - 1,
                )
            st.session_state.transport_calc                = calc
            st.session_state.transport_calc_key            = calc_key
            st.session_state.agent_state["transport_cost"] = calc

    with st.chat_message("assistant", avatar="assets/icon.png"):
        placeholder = st.empty()
        placeholder.markdown("▌")
        new_asst   = []   # defined before try so it's always accessible after except
        full_reply = "Sorry, something went wrong. Please try again."
        try:
            config            = {"configurable": {"thread_id": st.session_state.thread_id}}
            _prev_destination = st.session_state.agent_state.get("destination")
            _prev_stage       = st.session_state.agent_state.get("stage")
            # Count messages in previous state (BEFORE the new user msg).
            # We then pass only the new user msg to invoke — MemorySaver checkpointer
            # already holds the full history, so passing accumulated messages would
            # double them via operator.add.
            _total_msgs_before = len(st.session_state.agent_state.get("messages", []))
            _invoke_state = {k: v for k, v in st.session_state.agent_state.items() if k != "messages"}
            _invoke_state["messages"] = [{"role": "user", "content": user_input}]
            result            = st.session_state.graph.invoke(_invoke_state, config=config)
            st.session_state.agent_state = result

            all_msgs_after = result.get("messages", [])
            new_msgs  = all_msgs_after[_total_msgs_before:]  # only what the graph added
            new_asst  = [m for m in new_msgs if m["role"] == "assistant"]
            full_reply = new_asst[-1]["content"] if new_asst else "Sorry, something went wrong."
            placeholder.markdown(re.sub(r"(?<!\\)\$", r"\\$", full_reply))

            # Post-invoke transport calc if itinerary just generated
            if (result.get("stage") == "done" and result.get("transport") == "driving"
                    and result.get("vehicle") and result.get("origin") and result.get("destination")):
                calc_key = f"{result.get('origin')}_{result.get('destination')}_{result.get('num_travelers',1)}"
                if st.session_state.get("transport_calc_key") != calc_key:
                    with st.spinner("Calculating road trip costs..."):
                        calc = calculate_road_trip(
                            origin        = result["origin"],
                            destination   = result["destination"],
                            vehicle       = result["vehicle"],
                            num_travelers = result.get("num_travelers", 1),
                            num_nights    = (result.get("num_days") or 1) - 1,
                        )
                    st.session_state.transport_calc     = calc
                    st.session_state.transport_calc_key = calc_key

            # Detect when the user is starting a new trip after a completed one.
            # Two cases:
            #   1. New US destination: destination → None, itinerary → None
            #   2. International redirect: user typed a new international place after finishing a trip
            # Both should save the old chat and open a new one.
            _from_done_trip     = _prev_stage == "done" and bool(_prev_destination)
            _new_us_trip        = (
                result.get("stage") == "clarifying"
                and result.get("destination") is None
                and result.get("itinerary") is None
                and bool(_prev_destination)
            )
            _new_intl_from_done = (
                _from_done_trip
                and result.get("stage") == "clarifying"
                and bool(result.get("_intl_destination"))
            )
            is_resetting = _new_us_trip or _new_intl_from_done
            if is_resetting:
                # ── New destination = new chat ─────────────────────────────────────────
                # Save old chat, then open a brand-new chat for the new trip.

                # 1. Append notice to old chat before saving
                _old_dest = _prev_destination or "your previous trip"
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content":
                    f"\u2708\ufe0f Looks like you're heading somewhere new! "
                    f"I've saved your **{_old_dest}** trip and I'm opening a fresh chat."})
                save_current_chat()

                # 2. Run fresh agent on the new input
                new_state = get_initial_state()
                new_state["messages"].append({"role": "user", "content": user_input})
                st.session_state.thread_id = f"trip_{int(time.time())}"
                config2 = {"configurable": {"thread_id": st.session_state.thread_id}}
                try:
                    result2    = st.session_state.graph.invoke(new_state, config=config2)
                    st.session_state.agent_state = result2
                    new_asst2  = [m for m in result2.get("messages", []) if m["role"] == "assistant"]
                    reply2     = new_asst2[-1]["content"] if new_asst2 else "Sure! Where do you want to go? \U0001f30d"
                except Exception:
                    reply2 = "Sure! Where do you want to go? \U0001f30d"
                    st.session_state.agent_state = get_initial_state()

                placeholder.markdown(re.sub(r"(?<!\\)\$", r"\\$", reply2))
                full_reply = reply2

                # 3. Fresh messages for the new chat
                st.session_state.messages = [
                    {"role": "user",      "content": user_input},
                    {"role": "assistant", "content": reply2},
                ]
                st.session_state.active_chat_id         = None
                st.session_state.trip_info              = {}
                st.session_state.transport_calc         = None
                st.session_state.transport_calc_key     = None
                st.session_state.show_grocery           = False
                st.session_state.grocery_form_done      = False
                st.session_state.grocery_data           = None
                st.session_state.grocery_travelers_info = None
                st.session_state.grocery_budget_mode    = "group"
                st.session_state.show_checklist         = False
                st.session_state.checklist_data         = None
                st.session_state.budget_guardrail       = None
                st.session_state._last_itinerary        = None
                st.session_state.greeting               = None
                _clear_draft()
                st.rerun()
            st.session_state.trip_info = {
                "destination": result.get("destination"),
                "origin":      result.get("origin"),
                "days":        f"{result['num_days']} days" if result.get("num_days") else None,
                "budget":      f"${int(result['budget'])}"  if result.get("budget")   else None,
                "vibe":        result.get("vibe", "").capitalize() if result.get("vibe") else None,
            }
            if result.get("budget_guardrail"):
                st.session_state.budget_guardrail = result["budget_guardrail"]

            # Reset grocery & checklist whenever the itinerary is revised
            # so the user can regenerate them with the updated trip details.
            new_itinerary = result.get("itinerary")
            if (result.get("stage") == "done"
                    and new_itinerary
                    and new_itinerary != st.session_state._last_itinerary):
                st.session_state._last_itinerary    = new_itinerary
                st.session_state.grocery_data       = None
                st.session_state.checklist_data     = None
                st.session_state.show_grocery       = False
                st.session_state.show_checklist     = False
                st.session_state.grocery_form_done  = False
                st.session_state.grocery_travelers_info = None

        except Exception as e:
            import traceback
            logger.error("Graph error: %s\n%s", e, traceback.format_exc())
            full_reply = f"⚠️ Error: {str(e)}"
            placeholder.markdown(full_reply)

    # Append all new assistant messages as separate chat bubbles
    if new_asst:
        for _new_m in new_asst:
            st.session_state.messages.append({"role": "assistant", "content": _new_m["content"]})
    else:
        st.session_state.messages.append({"role": "assistant", "content": full_reply})
    save_current_chat()
    st.rerun()