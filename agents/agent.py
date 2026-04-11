"""
agent.py
--------
LangGraph travel planning agent — state, nodes, and graph in one file.

Flow:
  START → parse_input → clarify → (loop until all 5 fields collected)
                                → generate → END
  (if stage == "done" and user sends message → revise → END)
"""

import os
import json
import logging
from typing import TypedDict, Annotated, List, Optional
import operator

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("wander")


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
class TravelState(TypedDict):
    messages:         Annotated[List[dict], operator.add]
    destination:      Optional[str]
    origin:           Optional[str]
    num_days:         Optional[int]
    budget:           Optional[float]
    vibe:             Optional[str]
    transport:        Optional[str]   # "flying" | "driving"
    num_travelers:    Optional[int]   # total people on the trip
    budget_type:      Optional[str]   # "per_person" | "total"
    cuisine_prefs:    Optional[str]   # for food-focused destinations
    stage:            str
    itinerary:        Optional[str]
    budget_breakdown:       Optional[dict]
    budget_exceeded:        bool
    _international_attempt: Optional[bool]


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

llm_precise = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def call_llm(llm_instance, messages):
    return llm_instance.invoke(messages)


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════
def parse_input_node(state: TravelState) -> dict:
    """Extract trip details from the latest user message."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Get the latest user message specifically
    user_msgs = [m for m in messages if m["role"] == "user"]
    if not user_msgs:
        return {}
    latest_user_msg = user_msgs[-1]["content"]

    # Build recent context for other fields, but evaluate destination from latest message only
    recent = messages[-6:]
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent if m["role"] in ("user", "assistant")
    )

    prompt = """Extract trip info from this conversation.

LATEST USER MESSAGE: "{latest}"

DESTINATION RULES:
- Evaluate destination ONLY from the latest user message above
- Do NOT be influenced by international destinations mentioned earlier in the conversation
- Flag as international ONLY if the latest message contains a country/city outside the US
- US ethnic neighborhoods are valid US destinations: Jackson Heights NY, Little Tokyo LA, 
  Chinatown, French Quarter, Little Havana, Koreatown, Little India, South Beach, etc.
- Clearly international: India, Japan, Tokyo, Paris, London, Mexico, Cancun, Europe, etc.
- US cities, states, regions, neighborhoods → is_international: false

Return ONLY JSON — use null if not mentioned:
{{
  "destination": "exactly as the user said in their latest message, or null if international",
  "is_international": true if destination is outside the US, false if in the US,
  "origin": "where they travel FROM or null",
  "num_days": integer or null,
  "budget": number in USD or null,
  "vibe": "comma-separated list of vibes from: culture, food, nightlife, outdoors, road trip, beach, mix — or null",
  "transport": "flying" or "driving" or null,
  "num_travelers": integer (total people including the user) or null,
  "budget_type": "per_person" if they said 'each'/'per person', "total" if group total, or null,
  "cuisine_prefs": "specific cuisines mentioned or null"
}}

Full conversation for context (for non-destination fields only):
{transcript}""".format(latest=latest_user_msg, transcript=transcript)

    try:
        r = call_llm(llm_precise, [SystemMessage(content=prompt)])
        raw = r.content.strip().replace("```json", "").replace("```", "")
        extracted = json.loads(raw)

        updates = {}

        # Only extract destination if we don't have one yet
        if not state.get("destination"):
            if not extracted.get("is_international") and extracted.get("destination"):
                updates["destination"] = extracted["destination"].title()
            if extracted.get("is_international"):
                updates["_international_attempt"] = True
        
        # Always update these if present in latest message
        if extracted.get("origin"):
            updates["origin"] = extracted["origin"].title()
        if extracted.get("num_days"):
            updates["num_days"] = int(extracted["num_days"])
        if extracted.get("budget"):
            updates["budget"] = float(str(extracted["budget"]).replace("$", "").replace(",", ""))
        if extracted.get("vibe"):
            updates["vibe"] = extracted["vibe"].lower()
        if extracted.get("transport"):
            updates["transport"] = extracted["transport"].lower()
        if extracted.get("num_travelers"):
            updates["num_travelers"] = int(extracted["num_travelers"])
        if extracted.get("budget_type"):
            updates["budget_type"] = extracted["budget_type"]
        if extracted.get("cuisine_prefs"):
            updates["cuisine_prefs"] = extracted["cuisine_prefs"]

        logger.info("Parsed: %s", updates)
        return updates

    except Exception as e:
        logger.warning("Parse failed: %s", e)
        return {}


def clarify_node(state: TravelState) -> dict:
    """Ask for the next missing field, or advance to generating."""

    # Check if user tried an international destination
    if state.get("_international_attempt"):
        # Get what they said
        user_msgs = [m["content"] for m in state.get("messages", []) if m["role"] == "user"]
        last_msg = user_msgs[-1] if user_msgs else ""

        redirect_prompt = """The user wants to go to an international destination. TripBuddy only plans US trips.

Suggest 3-4 US neighborhoods/cities that genuinely recreate that experience. Don't limit to obvious choices like NYC and SF — include cities like Houston, Chicago, LA, Dallas, Atlanta, DC, Miami, Detroit, and others that have large authentic ethnic communities.

Format:
1. One casual sentence acknowledging where they want to go
2. "TripBuddy is US-only, but these spots will give you that same vibe:"
3. 3-4 bullet points, each with:
   - **Neighborhood, City** (bold) — lead with the specific neighborhood, not just the city
   - What to eat: 1-2 specific dishes or spots
   - What to do: 1 cultural activity, market, or festival
4. End with: "Which one sounds good, or somewhere else in mind?"

Keep it casual, punchy, college-friendly. Short sentences.

User said: {msg}""".format(msg=last_msg)

        r = call_llm(llm, [SystemMessage(content=redirect_prompt)])
        return {
            "stage": "clarifying",
            "_international_attempt": False,
            "messages": [{"role": "assistant", "content": r.content}]
        }

    # Mid-conversation: check if user is switching to a different destination
    if state.get("destination"):
        messages = state.get("messages", [])
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        asst_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
        latest = user_msgs[-1] if user_msgs else ""
        last_question = asst_msgs[-1] if asst_msgs else ""

        try:
            intent = call_llm(llm_precise, [SystemMessage(content=f"""Context:
- Current destination: {state.get('destination')}
- Last question asked: "{last_question}"
- User's reply: "{latest}"

Is the user's reply a direct answer to the question asked, or are they switching to a completely different destination?

Answer "answer" if they are responding to the question (e.g. giving a number, transport method, budget, vibe, cuisine, yes/no, or any other trip detail).
Answer "new" if they are clearly naming a different place they want to go to instead.

Reply with only "answer" or "new".""")])
            if intent.content.strip().lower().startswith("new"):
                return {
                    "stage": "clarifying",
                    "destination": None,
                    "origin": None,
                    "num_days": None,
                    "budget": None,
                    "vibe": None,
                    "transport": None,
                    "num_travelers": None,
                    "budget_type": None,
                    "cuisine_prefs": None,
                    "itinerary": None,
                    "_international_attempt": False,
                    "messages": [{"role": "user", "content": latest}]
                }
        except Exception:
            pass

    missing = []
    if not state.get("destination"):
        missing.append("destination (US city, region, or state)")
    if not state.get("origin"):
        missing.append("where they are traveling from")
    if not state.get("transport"):
        missing.append("how they are getting there (flying or driving)")
    if not state.get("num_travelers"):
        missing.append("how many people are going on this trip (so we can split costs)")
    if not state.get("num_days"):
        missing.append("number of days")
    if not state.get("budget"):
        missing.append("total budget in USD — is this per person or for the whole group?")
    if not state.get("vibe"):
        missing.append("vibe")
    if state.get("vibe") and "food" in state.get("vibe", "").lower() and not state.get("cuisine_prefs"):
        missing.append("what cuisines they want to try (e.g. Indian, Korean, Mexican, Italian)")

    if not missing:
        logger.info("All fields collected — advancing to generate")
        return {"stage": "generating"}

    known = {k: v for k, v in {
        "destination":   state.get("destination"),
        "origin":        state.get("origin"),
        "transport":     state.get("transport"),
        "travelers":     f"{state.get('num_travelers')} people" if state.get("num_travelers") else None,
        "days":          state.get("num_days"),
        "budget":        f"${state.get('budget')} {state.get('budget_type', '')}".strip() if state.get("budget") else None,
        "vibe":          state.get("vibe"),
        "cuisine":       state.get("cuisine_prefs"),
    }.items() if v}

    # Build location-aware vibe options
    destination = state.get("destination", "").lower()
    beach_cities = ["miami", "myrtle beach", "san diego", "honolulu", "virginia beach",
                    "galveston", "santa monica", "clearwater", "panama city"]
    has_beach = any(city in destination for city in beach_cities)

    if "vibe" in (missing[0] if missing else ""):
        vibe_options = "culture, food, nightlife, outdoors, road trip"
        if has_beach:
            vibe_options += ", beach"
        vibe_options += ", or a mix — pick as many as you want!"

        food_hint = ""
        if state.get("vibe") and "food" in (state.get("vibe") or ""):
            food_hint = " Since food is on the list, I'll ask about cuisines next."

        next_question = f"vibe — options: {vibe_options}.{food_hint}"
    else:
        next_question = missing[0]

    system = """You are TripBuddy, a friendly US budget travel planner for college students.
You ONLY plan trips within the United States.
Ask for ONE missing piece of info at a time. Be casual — like a well-traveled older student.

IMPORTANT: Use the conversation history to understand what each answer refers to.
If you asked "where are you traveling from?" and the user replied "Tulsa", that means Tulsa is the ORIGIN, not the destination.
Always interpret user answers in context of the last question asked.

Already know: {known}
Next to ask: {next_question}

Keep it short and conversational. Ask only for that one thing.
If asking about transport, ask: flying or driving?
If asking about vibe for a food/cultural destination, mention they can name specific cuisines.""".format(
        known=", ".join(f"{k}: {v}" for k, v in known.items()) or "nothing yet",
        next_question=next_question
    )

    msgs = [SystemMessage(content=system)]
    for m in state.get("messages", [])[-10:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))

    r = call_llm(llm, msgs)
    logger.info("Clarify asking for: %s", missing[0] if missing else "generating")

    return {
        "stage": "clarifying",
        "messages": [{"role": "assistant", "content": r.content}]
    }


def generate_node(state: TravelState) -> dict:
    """Build the full itinerary once all 5 fields are collected."""
    destination = state.get("destination", "")
    logger.info("Generating itinerary for %s", destination)

    num_travelers = state.get("num_travelers", 1)
    budget = state.get("budget", 0)
    budget_type = state.get("budget_type", "total")
    total_budget = int(budget * num_travelers if budget_type == "per_person" else budget)
    per_person_budget = int(total_budget / num_travelers) if num_travelers > 1 else total_budget

    tb = f"${total_budget}"
    pp = f"${per_person_budget}"

    system = f"""You are a US budget travel planner for college students.

DESTINATION = {destination}
This is a city/place in the UNITED STATES. Plan the trip FOR this location.
Do NOT use the cuisine preference as the destination name.
The cuisine ({state.get('cuisine_prefs', 'not specified')}) is what they want to EAT there, not where they are going.

Trip details:
- Destination city: {destination}
- Traveling from: {state.get('origin')}
- Transport: {state.get('transport', 'not specified')}
- Group size: {num_travelers} people
- Days: {state.get('num_days')}
- Total budget: {tb} total / {pp} per person
- Vibe: {state.get('vibe')}
{"- Food preferences: " + state.get('cuisine_prefs') + " (find " + state.get('cuisine_prefs') + " restaurants IN " + destination + ")" if state.get('cuisine_prefs') else ""}

FORMATTING RULES:
- Header line: write exactly "{num_travelers} people | {tb} total | {pp} per person" — no markdown math
- Activity costs: "$X/person ($Y total)" — dollar sign only before numbers
- NEVER write $total, $person, $budget — only $NUMBER like $15, $500

Generate a detailed day-by-day itinerary for {destination} staying within {tb} total.
Prioritize free and cheap activities. Flag student discounts with 🎓.
Recommend SPECIFIC real restaurants in {destination} matching cuisine preferences.

## Your {state.get('num_days')}-Day {destination} Trip 🎒
{num_travelers} people | {tb} total | {pp} per person

### Day 1 — [Theme]
**Morning:** Activity — $X/person ($Y total)
**Afternoon:** Activity — $X/person ($Y total)  
**Evening:** Dinner at [Real Restaurant in {destination}] — $X/person ($Y total)

...continue each day...

### Budget Breakdown
| Category | Per Person | Group Total |
|---|---|---|
| Transport | $X | $Y |
| Accommodation | $X | $Y |
| Food | $X | $Y |
| Activities | $X | $Y |
| **Total** | {pp} | {tb} |

💡 3 money-saving tips specific to {destination}"""

    r = call_llm(llm, [SystemMessage(content=system)])
    itinerary = r.content

    budget_exceeded = False
    try:
        import re
        totals = re.findall(r"\*\*Total\*\*.*?\$([0-9,]+)", itinerary)
        if totals:
            estimated = float(totals[-1].replace(",", ""))
            budget_exceeded = estimated > total_budget
    except Exception:
        pass

    return {
        "stage": "done",
        "itinerary": itinerary,
        "budget_exceeded": budget_exceeded,
        "messages": [{"role": "assistant", "content": itinerary + "\n\n---\n✅ Your itinerary is ready! Want me to tweak anything — swap a hotel, add an activity, or adjust the budget? Or ready to plan another trip?"}]
    }


def revise_node(state: TravelState) -> dict:
    """Handle follow-up changes after itinerary is delivered."""
    user_msgs = [m for m in state.get("messages", []) if m["role"] == "user"]
    latest = user_msgs[-1]["content"] if user_msgs else ""

    # Use LLM to detect if user wants a new trip or is modifying the current one
    try:
        intent = call_llm(llm_precise, [SystemMessage(content=f"""Does this message indicate the user wants to plan a NEW trip to a DIFFERENT destination, or are they modifying their current itinerary?

Current destination: {state.get('destination')}
User message: "{latest}"

Answer "new" if they mention a different city, state, or destination they want to go to instead.
Answer "modify" if they want to change something about the current trip — including:
- Swapping hotels, accommodations, or restaurants
- Changing activities on any day
- Adjusting budget up or down
- Adding stops or detours along the way
- Changing travel dates or number of days
- Any other tweak to the existing itinerary

Answer only "new" or "modify".""")])
        wants_new_trip = intent.content.strip().lower().startswith("new")
    except Exception:
        wants_new_trip = False

    if wants_new_trip:
        # Pass the user's message into the new state so parse+clarify handle it
        return {
            "stage": "clarifying",
            "destination": None,
            "origin": None,
            "num_days": None,
            "budget": None,
            "vibe": None,
            "transport": None,
            "num_travelers": None,
            "budget_type": None,
            "cuisine_prefs": None,
            "itinerary": None,
            "_international_attempt": False,
            # Keep the user's message so parse_input_node can extract the new destination
            "messages": [{"role": "user", "content": latest}]
        }

    prompt = """Update this student travel itinerary based on the user's request.
Make ONLY the requested change. Keep everything else the same.
This is a US-only trip — destination stays in the US.

Current itinerary:
{itinerary}

Request: {request}
Budget: ${budget}""".format(
        itinerary=state.get("itinerary", ""),
        request=latest,
        budget=state.get("budget")
    )

    r = call_llm(llm, [SystemMessage(content=prompt)])
    return {
        "stage": "done",
        "messages": [{"role": "assistant", "content": r.content}]
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════════════════════
def route_after_parse(state: TravelState) -> str:
    # Always go to clarify first if international attempt detected
    if state.get("_international_attempt"):
        return "clarify"
    if state.get("stage") == "done":
        return "revise"
    return "clarify"


def route_after_clarify(state: TravelState) -> str:
    if state.get("stage") == "generating":
        return "generate"
    return END


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("parse_input", parse_input_node)
    graph.add_node("clarify",     clarify_node)
    graph.add_node("generate",    generate_node)
    graph.add_node("revise",      revise_node)

    graph.set_entry_point("parse_input")

    graph.add_conditional_edges("parse_input", route_after_parse, {
        "clarify": "clarify",
        "revise":  "revise",
    })

    graph.add_conditional_edges("clarify", route_after_clarify, {
        "generate": "generate",
        END: END,
    })

    graph.add_edge("generate", END)
    graph.add_edge("revise",   END)

    # Use MemorySaver so state persists across nodes within a conversation
    return graph.compile(checkpointer=MemorySaver())


def get_initial_state() -> TravelState:
    return TravelState(
        messages=[],
        destination=None,
        origin=None,
        num_days=None,
        budget=None,
        vibe=None,
        transport=None,
        num_travelers=None,
        budget_type=None,
        cuisine_prefs=None,
        stage="clarifying",
        itinerary=None,
        budget_breakdown=None,
        budget_exceeded=False,
        _international_attempt=False,
    )