"""
agent.py — TripBuddy LangGraph agent + Grocery & Checklist agent
"""

import os
import re
import json
import logging
from typing import TypedDict, Annotated, List, Optional, Any
import operator

from dotenv import load_dotenv
load_dotenv()

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("tripbuddy")


# ══════════════════════════════════════════════════════════════════════════════
#  OBSERVABILITY
# ══════════════════════════════════════════════════════════════════════════════
# In this project we are implementing the Observability using LangSmith's
# tracing features. # By importing the library wrap_openai and adding the @traceable decorator for Python
# we can trace the calls to OpenAI.

from langsmith import traceable



# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
class TravelState(TypedDict):
    messages:            Annotated[List[dict], operator.add]
    destination:         Optional[str]
    origin:              Optional[str]
    num_days:            Optional[int]
    budget:              Optional[float]
    vibe:                Optional[str]
    transport:           Optional[str]
    vehicle:             Optional[str]
    num_cars:            Optional[int]
    num_travelers:       Optional[int]
    budget_type:         Optional[str]
    cuisine_prefs:       Optional[str]
    accommodation_pref:  Optional[str]
    transport_cost:      Optional[dict]
    stage:               str
    itinerary:           Optional[str]
    budget_breakdown:    Optional[dict]
    budget_exceeded:     bool
    budget_guardrail:    Optional[dict]
    _international_attempt: Optional[bool]
    _capacity_exceeded:  Optional[bool]
    _capacity_warned:    Optional[bool]   # True once we've shown the capacity warning
    _intl_destination:   Optional[str]



# ══════════════════════════════════════════════════════════════════════════════
# LOAD RAG AND DEFINE CALLER NODE
# ══════════════════════════════════════════════════════════════════════════════

try:
    from rag.service import TravelRAGService
    rag_service = TravelRAGService.from_persisted_db()
    RAG_AVAILABLE = True

    def retrieve_travel_context(user_query:str,city:str, category:str, k=5) -> str:
        query = user_query
        destination = city
        category = category

        results = rag_service.search(
            query=query,
            city =destination,
            category=category,
            k=k,
        )

        context = rag_service.format_context(results, max_chars=5000)
        return context


except Exception as e:
    RAG_AVAILABLE = False
    print("It has occurred an error reading the rag database: " , e)




# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
llm_precise = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def call_llm(llm_instance, messages):
    return llm_instance.invoke(messages)

def _clean(val) -> str | None:
    """Return None if the LLM returned the string 'null', 'none', or an empty value
    instead of proper JSON null. Prevents literal 'null' leaking into state."""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("null", "none", "n/a", ""):
        return None
    return val

def extract_text(r) -> str:
    """Safely extract string content from a LangChain response.
    Newer model versions may return r.content as a list of content blocks
    rather than a plain string — this handles both cases.
    """
    c = r.content
    if isinstance(c, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in c
        )
    return str(c or "")


# ══════════════════════════════════════════════════════════════════════════════
# DESTINATION CONTEXT  (LLM-powered, cached per session)
# ══════════════════════════════════════════════════════════════════════════════
_destination_cache: dict = {}

def get_destination_context(destination: str) -> dict:
    """
    LLM-powered destination profile. Returns:
    - has_beach: whether the destination has notable beach access
    - local_cuisine: hyper-local/unconventional foods the city is genuinely known for
    - cuisine_suggestions: broader list of cuisine types relevant to this city/region
    - extra_vibes: city-specific vibes beyond the standard set (no overlaps with standard set)
    Cached per destination so the LLM is called at most once per session.
    """
    if not destination:
        return {"has_beach": False, "local_cuisine": "", "cuisine_suggestions": "", "extra_vibes": []}

    key = destination.strip().lower()

    if key in _destination_cache:
        return _destination_cache[key]
    try:
        r = call_llm(llm_precise, [SystemMessage(content=f"""For the US travel destination "{destination}", return ONLY JSON:
{{
  "has_beach": true or false,
  "local_cuisine": "2-3 hyper-local or unconventional foods/dishes this place is genuinely famous for (e.g. Nashville → hot chicken, meat & three; New Orleans → po'boys, beignets, Creole; Memphis → dry-rub ribs; Philadelphia → cheesesteaks, water ice). Be honest and specific.",
  "cuisine_suggestions": "8-10 cuisine types that are actually well-represented in this city — mix of what the city is known for and what's commonly available there. Examples: Korean, Vietnamese, Ethiopian, Tex-Mex, soul food, seafood, BBQ, ramen, etc. Tailor to the actual dining scene of this specific place.",
  "extra_vibes": ["IMPORTANT RULES for this list: (1) The standard vibes already shown to users are: culture, food, nightlife, outdoors, road trip, beach. Do NOT include any of these or synonyms/variants of them (e.g. do NOT add foodie, live music if music is already there). (2) Only include vibes that are truly iconic for this specific destination. (3) If two vibes overlap (e.g. music + live-music), keep only the most specific one. (4) Empty array if nothing truly distinctive applies beyond the standard set. Good examples: Nashville → [music, bourbon, honky-tonk], Las Vegas → [casino, shows], New Orleans → [jazz, Mardi-Gras], Austin → [live-music, college-town, tech]."]
}}
Be specific and honest. Reflect what locals and visitors actually experience.""")])
        result = json.loads(extract_text(r).strip().replace("```json", "").replace("```", ""))
        _destination_cache[key] = result
        return result
    except Exception:
        return {"has_beach": False, "local_cuisine": "", "cuisine_suggestions": "", "extra_vibes": []}

def build_vibe_options(destination: str) -> str:
    """Build the vibe options string, deduplicating against the standard set."""
    ctx         = get_destination_context(destination)
    has_beach   = ctx.get("has_beach", False)
    extra_vibes = ctx.get("extra_vibes", [])

    standard = {"culture", "food", "nightlife", "outdoors", "road trip", "beach"}
    seen      = set(standard)
    base      = ["culture", "food", "nightlife", "outdoors", "road trip"]
    if has_beach:
        base.append("beach")

    deduped_extras = []
    for v in extra_vibes:
        normalized = v.strip().lower().replace("-", " ")

        # Skip if identical to or a substring/superset of an already-seen vibe
        if normalized not in seen and not any(normalized in s or s in normalized for s in seen):
            deduped_extras.append(v.strip())
            seen.add(normalized)

    all_vibes = base + deduped_extras
    return ", ".join(all_vibes) + " — pick as many as you want!"

def build_cuisine_question(destination: str) -> str:
    """
    Build cuisine question entirely from LLM context.
    Leads with what the city is known for, then shows broader options.
    User can always type something not in the list.
    """
    ctx               = get_destination_context(destination)
    local_cuisine     = ctx.get("local_cuisine", "").strip()
    cuisine_suggestions = ctx.get("cuisine_suggestions", "").strip()

    if local_cuisine and cuisine_suggestions:
        return (
            f"{destination} is known for {local_cuisine}! "
            f"Beyond that, you can also find {cuisine_suggestions} — "
            f"or type anything else you're craving!"
        )
    elif cuisine_suggestions:
        return (
            f"What cuisines are you into? {destination} has great options: "
            f"{cuisine_suggestions} — or type whatever you're craving!"
        )
    else:
        return (
            f"What cuisines are you into? Think about what {destination} is known for "
            f"or anything else you're craving!"
        )


# ══════════════════════════════════════════════════════════════════════════════
# NHTSA vPIC API — Free, no API key, government data
# ══════════════════════════════════════════════════════════════════════════════
import urllib.request as _urllib_req

_NHTSA_BASE  = "https://vpic.nhtsa.dot.gov/api/vehicles"
_nhtsa_make_models: dict[str, list[str]] = {}   # in-process cache: make → [models]


def _nhtsa_get(url: str, timeout: int = 4) -> dict:
    try:
        with _urllib_req.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return {}

def nhtsa_models_for_make(make: str) -> list[str]:
    """Return list of model names for a given make from NHTSA. Cached per session."""
    key = make.strip().lower()
    if key in _nhtsa_make_models:
        return _nhtsa_make_models[key]
    data = _nhtsa_get(f"{_NHTSA_BASE}/getmodelsformake/{make}?format=json")
    models = [m["Model_Name"] for m in data.get("Results", [])]
    _nhtsa_make_models[key] = models
    return models

def nhtsa_validate_vehicle(make: str, model: str) -> str | None:
    """Return the canonical model name from NHTSA if make+model is valid, else None."""
    models = nhtsa_models_for_make(make)
    if not models:
        return None
    model_lower = model.strip().lower()
    # Exact match first
    for m in models:
        if m.strip().lower() == model_lower:
            return m
    # Prefix/partial match
    for m in models:
        if m.strip().lower().startswith(model_lower) or model_lower.startswith(m.strip().lower()[:4]):
            return m
    return None

def nhtsa_suggest_models(make: str, prefix: str = "") -> list[str]:
    """Return up to 5 model suggestions for a make, optionally filtered by prefix."""
    models = nhtsa_models_for_make(make)
    if prefix:
        models = [m for m in models if m.lower().startswith(prefix.lower())]
    return models[:5]


# ══════════════════════════════════════════════════════════════════════════════
# VEHICLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Fast local check — common bare makes that definitely need a model.
# Used as a first-pass before the LLM call so vehicle is never null when driving.
_BARE_MAKES = {
    "toyota", "honda", "ford", "chevy", "chevrolet", "gmc", "dodge", "jeep",
    "nissan", "hyundai", "kia", "subaru", "mazda", "volkswagen", "vw", "bmw",
    "mercedes", "audi", "lexus", "infiniti", "acura", "volvo", "tesla",
    "buick", "cadillac", "lincoln", "chrysler", "ram", "mitsubishi", "genesis",
    "rivian", "lucid", "polestar", "jaguar", "land rover", "porsche", "ferrari",
}

def check_vehicle_input(vehicle: str) -> dict:
    """
    Determine whether the user provided both make and model.
    Fast local check first (bare make set), then LLM for anything ambiguous.
    Default on LLM failure is is_complete=False — conservative, never lets null through.
    """
    if not vehicle or not vehicle.strip():
        return {"is_complete": False, "make_only": None, "suggested_models": []}

    key = vehicle.strip().lower()

    # Fast path: known bare make with no model
    if key in _BARE_MAKES:
        return {"is_complete": False, "make_only": vehicle.strip().title(), "suggested_models": []}

    try:
        r = call_llm(llm_precise, [SystemMessage(content=f"""The user said their car is: "{vehicle}"

Identify the vehicle and infer the full make + model, even from partial input, abbreviations, or shorthand.
Use your knowledge of common cars — be generous with inference. Only mark as incomplete if it is genuinely too ambiguous.
When in doubt, pick the most popular model for that make.

Return ONLY JSON:
{{
  "is_complete": true or false,
  "corrected": "the correctly spelled make and model if identifiable, else null",
  "make_only": "the make if only a make was given and model is truly unidentifiable, else null",
  "suggested_models": ["3-4 most popular models for that make if make_only is not null, else empty array"]
}}

Examples:
- "Toyota Camry" → is_complete: true, corrected: "Toyota Camry"
- "Toyota Camr" → is_complete: true, corrected: "Toyota Camry" (obvious typo)
- "toyota cam" → is_complete: true, corrected: "Toyota Camry" (common shorthand for Camry)
- "honda crv" → is_complete: true, corrected: "Honda CR-V"
- "honda civ" → is_complete: true, corrected: "Honda Civic" (shorthand)
- "ford f150" → is_complete: true, corrected: "Ford F-150"
- "chevy mal" → is_complete: true, corrected: "Chevrolet Malibu" (shorthand)
- "Toyota" → is_complete: false, make_only: "Toyota", suggested_models: ["Camry", "RAV4", "Corolla", "Highlander"]
- "my car" → is_complete: false, make_only: null, suggested_models: []
- "Ford F" → is_complete: false, make_only: "Ford", suggested_models: ["F-150", "Explorer", "Escape", "Mustang"]
- "hond" → is_complete: false, make_only: "Honda", suggested_models: ["Civic", "Accord", "CR-V", "Pilot"]""")])
        result = json.loads(extract_text(r).strip().replace("```json", "").replace("```", ""))
        return result
    except Exception:
        # Fallback: multi-word vehicle (e.g. "Toyota Camry") → assume complete (True)
        # Single word not in BARE_MAKES → unclear, assume incomplete (False)
        _words = vehicle.strip().split()
        return {
            "is_complete": len(_words) >= 2,
            "make_only": None,
            "suggested_models": [],
        }

# Common make abbreviations → canonical make name
_MAKE_ABBREVS = {
    "toy": "Toyota", "toyo": "Toyota",
    "hon": "Honda", "hond": "Honda",
    "chev": "Chevrolet", "chvy": "Chevrolet", "chevy": "Chevrolet",
    "ford": "Ford",
    "dodge": "Dodge", "dodg": "Dodge",
    "jeep": "Jeep",
    "kia": "Kia",
    "hyun": "Hyundai", "hyund": "Hyundai",
    "nissan": "Nissan", "niss": "Nissan",
    "mazda": "Mazda",
    "subaru": "Subaru", "suba": "Subaru",
    "volvo": "Volvo",
    "bmw": "BMW",
    "audi": "Audi",
    "vw": "Volkswagen", "volk": "Volkswagen",
    "merc": "Mercedes-Benz", "benz": "Mercedes-Benz",
    "lex": "Lexus", "lexus": "Lexus",
    "acura": "Acura",
    "infin": "Infiniti",
    "buick": "Buick",
    "gmc": "GMC",
    "ram": "Ram",
    "tesla": "Tesla",
}

# Common model abbreviations per make (first few chars → full model)
_MODEL_ABBREVS: dict[str, dict[str, str]] = {
    "Toyota":      {"cam": "Camry", "rav": "RAV4", "cor": "Corolla", "hig": "Highlander",
                    "tac": "Tacoma", "tun": "Tundra", "pri": "Prius", "4ru": "4Runner"},
    "Honda":       {"civ": "Civic", "acc": "Accord", "crv": "CR-V", "pil": "Pilot",
                    "ody": "Odyssey", "rid": "Ridgeline", "fit": "Fit", "hr": "HR-V"},
    "Chevrolet":   {"sil": "Silverado", "1500": "Silverado 1500", "mal": "Malibu",
                    "equ": "Equinox", "tra": "Traverse", "sub": "Suburban",
                    "cam": "Camaro", "cor": "Corvette", "cru": "Cruze"},
    "Ford":        {"f15": "F-150", "f-1": "F-150", "150": "F-150", "f150": "F-150",
                    "250": "F-250", "350": "F-350", "mus": "Mustang", "exp": "Explorer",
                    "esc": "Escape", "edg": "Edge", "ran": "Ranger", "bro": "Bronco"},
    "Nissan":      {"alt": "Altima", "sen": "Sentra", "rog": "Rogue", "pat": "Pathfinder",
                    "max": "Maxima", "fro": "Frontier", "tit": "Titan"},
    "Hyundai":     {"son": "Sonata", "ela": "Elantra", "tuc": "Tucson", "san": "Santa Fe"},
    "Dodge":       {"cha": "Charger", "chal": "Challenger", "dur": "Durango", "ram": "RAM 1500"},
    "Jeep":        {"wra": "Wrangler", "gra": "Grand Cherokee", "che": "Cherokee", "com": "Compass"},
    "Kia":         {"sol": "Sorento", "spo": "Sportage", "opt": "Optima", "tel": "Telluride"},
    "Subaru":      {"out": "Outback", "for": "Forester", "imp": "Impreza", "cro": "Crosstrek"},
    "GMC":         {"sie": "Sierra", "yuk": "Yukon", "ter": "Terrain", "aca": "Acadia"},
    "BMW":         {"3se": "3 Series", "5se": "5 Series", "x5": "X5", "x3": "X3"},
    "Mercedes-Benz": {"c-c": "C-Class", "e-c": "E-Class", "s-c": "S-Class", "glc": "GLC"},
    "Lexus":       {"rx": "RX", "es": "ES", "is": "IS", "nx": "NX", "gx": "GX"},
    "Mazda":       {"cx5": "CX-5", "cx-": "CX-5", "3": "Mazda3", "6": "Mazda6"},
    "Tesla":       {"mod": "Model 3", "ms": "Model S", "my": "Model Y", "mx": "Model X"},
}

def infer_full_vehicle(raw: str) -> str:
    """Given partial/abbreviated vehicle input, infer the full make+model.
    Uses a local abbreviation table first, then falls back to LLM.
    Returns empty string when input is too ambiguous to identify.
    """
    if not raw or not raw.strip():
        return raw
    try:
        r = call_llm(llm_precise, [SystemMessage(content=
            f'The user typed a car abbreviation: "{raw}"\n'
            '\n'
            'Identify the most likely car make and model. Be aggressive — always pick the best guess.\n'
            '\n'
            'Make abbreviations: "toy"/"toyo" = Toyota, "hon"/"hond" = Honda, '
            '"chev"/"chvy" = Chevrolet, "ford" = Ford, "jeep" = Jeep, "kia" = Kia\n'
            '\n'
            'Examples:\n'
            '"toy cam" → Toyota Camry\n'
            '"toyota cam" → Toyota Camry\n'
            '"toyota camr" → Toyota Camry\n'
            '"hon civ" → Honda Civic\n'
            '"honda crv" → Honda CR-V\n'
            '"ford f15" → Ford F-150\n'
            '"chev sil" → Chevrolet Silverado\n'
            '"chvy mal" → Chevrolet Malibu\n'
            '"toy rav" → Toyota RAV4\n'
            '\n'
            'Rules:\n'
            '- Only return a make+model when you can CONFIDENTLY identify it\n'
            '- Abbreviations and typos are fine: "toy cam" → Toyota Camry, "hon civ" → Honda Civic\n'
            '- Do NOT guess when input is too vague to identify a specific car\n'
            '- Return ONLY "Make Model" (e.g. Toyota Camry), or "unclear" if not identifiable')])
        result = extract_text(r).strip().strip('"').strip("'")
        # Empty string when unclear — clarify_node will ask the user to specify
        return result if result and result.lower() not in ("unclear", "unknown") else ""
    except Exception:
        return ""

def is_vehicle_complete(vehicle: str) -> bool:
    """Returns False if vehicle is null, empty, or only a bare make with no model."""
    if not vehicle or not vehicle.strip():
        return False
    key = vehicle.strip().lower()
    # Fast path: single known bare make → definitely incomplete
    if key in _BARE_MAKES:
        return False
    words = key.split()
    if len(words) >= 2 and words[0] in _BARE_MAKES:
        # Multi-word starting with a known make — likely make+model.
        # Call LLM for confirmation but default True on failure (assume complete).
        return check_vehicle_input(vehicle).get("is_complete", True)
    # Single word not in BARE_MAKES, or multi-word with unknown make → default True.
    return check_vehicle_input(vehicle).get("is_complete", True)

def get_vehicle_model_prompt(vehicle: str) -> str:
    """Return a follow-up question with NHTSA-powered model suggestions."""
    _make_raw = vehicle.split()[0].title()
    # Try NHTSA first for real model names
    nhtsa_models = nhtsa_suggest_models(_make_raw)
    if nhtsa_models:
        opts = ", ".join(nhtsa_models[:4])
        return f"What's the model for your {_make_raw}? Options from NHTSA: {opts} — or type yours!"
    # Fall back to LLM suggestions
    info      = check_vehicle_input(vehicle)
    suggested = info.get("suggested_models", [])
    if suggested:
        opts = ", ".join(suggested)
        return f"What's the model for your {_make_raw}? Popular options: {opts} — or type yours!"
    return f"What's the full make and model? (e.g. {_make_raw} Camry, {_make_raw} Explorer)"

# Model keyword → typical seating capacity
_CAR_CAPACITY: dict[str, int] = {
    # Sedans
    "camry": 5, "civic": 5, "accord": 5, "corolla": 5, "altima": 5,
    "sentra": 5, "malibu": 5, "sonata": 5, "elantra": 5, "optima": 5,
    "fusion": 5, "impala": 5, "charger": 5, "challenger": 4, "mustang": 4,
    "camaro": 4, "prius": 5, "maxima": 5, "passat": 5, "jetta": 5,
    # Compact / mid-size SUVs (5 seats)
    "rav4": 5, "rav-4": 5, "cr-v": 5, "crv": 5, "equinox": 5, "rogue": 5,
    "escape": 5, "tucson": 5, "sportage": 5, "compass": 5, "cherokee": 5,
    "edge": 5, "terrain": 5, "atlas": 5, "tiguan": 5, "cx-5": 5, "cx5": 5,
    "outback": 5, "forester": 5, "crosstrek": 5, "impreza": 5, "hrv": 5,
    "hr-v": 5, "kona": 5, "seltos": 5, "bronco sport": 5, "trailblazer": 5,
    "trax": 5, "encore": 5, "envision": 5, "x3": 5, "x1": 5, "q5": 5,
    "q3": 5, "glc": 5, "nx": 5, "rx": 5, "rdx": 5,
    # Full-size SUVs (7 seats)
    "explorer": 7, "highlander": 7, "pilot": 7, "traverse": 7, "palisade": 7,
    "sorento": 7, "telluride": 7, "pathfinder": 7, "armada": 7, "4runner": 5,
    "jeep": 5, "wrangler": 4, "grand cherokee": 5, "durango": 7,
    "x5": 5, "x7": 7, "q7": 7, "gx": 7, "lx": 7, "mdx": 7,
    # Large/full-size SUVs (8 seats)
    "tahoe": 7, "suburban": 8, "yukon": 7, "expedition": 7, "sequoia": 7,
    "navigator": 7, "armada": 7, "qx80": 7, "escalade": 7,
    # Minivans (7-8 seats)
    "odyssey": 8, "sienna": 8, "pacifica": 7, "carnival": 8, "voyager": 7,
    "quest": 7, "town & country": 7, "grand caravan": 7,
    # Trucks (5 for crew cab default)
    "f-150": 5, "f150": 5, "silverado": 5, "ram 1500": 5, "ram": 5,
    "tacoma": 5, "tundra": 5, "ranger": 5, "colorado": 5, "canyon": 5,
    "frontier": 5, "ridgeline": 5, "maverick": 5, "santa cruz": 5,
}

def get_vehicle_capacity(vehicle: str) -> dict:
    """Return typical seating capacity — local lookup first, LLM fallback."""
    v = vehicle.strip().lower()
    # Check each known model keyword against the vehicle string
    for model_key, seats in _CAR_CAPACITY.items():
        if model_key in v:
            _type = (
                "minivan" if seats >= 7 and any(k in v for k in ["odyssey","sienna","pacifica","carnival","caravan"])
                else "large_suv" if seats >= 7
                else "truck" if any(k in v for k in ["f-150","f150","silverado","ram","tacoma","tundra","ranger","colorado"])
                else "sedan" if any(k in v for k in ["camry","civic","accord","corolla","altima","malibu","sonata","prius"])
                else "suv"
            )
            return {"capacity": seats, "type": _type}
    # LLM fallback for unknown vehicles
    try:
        r = call_llm(llm_precise, [SystemMessage(content=
            f'How many people typically fit in a {vehicle}? Use the common everyday version.\n'
            'Sedans/compact SUVs = 5, large SUVs = 7, minivans = 7-8, trucks = 5.\n'
            'Return ONLY JSON: {"capacity": integer, "type": "sedan|suv|large_suv|minivan|truck"}')])
        return json.loads(extract_text(r).strip().replace("```json", "").replace("```", ""))
    except Exception:
        return {"capacity": 5, "type": "suv"}


# ══════════════════════════════════════════════════════════════════════════════
# PARSE NODE
# ══════════════════════════════════════════════════════════════════════════════
@traceable(name="parse_input_node")
def parse_input_node(state: TravelState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {}
    user_msgs = [m for m in messages if m["role"] == "user"]
    if not user_msgs:
        return {}
    latest_user_msg = user_msgs[-1]["content"]
    recent = messages[-6:]
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent if m["role"] in ("user", "assistant")
    )
    # Build state summary so the LLM knows what is already collected
    _state_summary = {
        "destination":       state.get("destination"),
        "origin":            state.get("origin"),
        "transport":         state.get("transport"),
        "vehicle":           state.get("vehicle"),
        "num_travelers":     state.get("num_travelers"),
        "num_days":          state.get("num_days"),
        "budget":            state.get("budget"),
        "budget_type":       state.get("budget_type"),
        "vibe":              state.get("vibe"),
        "cuisine_prefs":     state.get("cuisine_prefs"),
        "accommodation_pref": state.get("accommodation_pref"),
        "num_cars":          state.get("num_cars"),
    }

    prompt = """You are extracting trip fields from a user message.

CURRENT STATE (null = not yet collected):
{state_json}

LATEST USER MESSAGE: "{latest}"

TASK: Extract ONLY fields that are currently null OR that the user is explicitly changing.
Do NOT re-extract fields that are already set unless the user is clearly correcting them.

CRITICAL — DESTINATION ACCURACY:
The destination must be EXACTLY what the user typed — never substitute, translate, or replace it.
If the user said "india", destination = "India". If they said "japan", destination = "Japan".
Never output a different country than what the user explicitly mentioned.

NORMALIZATION RULES (critical — apply before returning):
- budget_type: normalize to EXACTLY "per_person" or "total"
    "whole group", "together", "all of us", "total", "for everyone" → "total"
    "each", "per person", "per head", "each of us", "individually", "pp" → "per_person"
    A bare dollar amount with no qualifier → null (do not guess)
- transport: EXACTLY "flying" or "driving"
- destination: null if international (India, France, Mexico, etc.)
- is_international: true for any non-US destination
- cuisine_prefs: "all" or "everything" or "any" → use "all cuisines"
- vehicle: extract from abbreviations/shorthand (system resolves them):
    "toy cam" → extract "toy cam", "ford 150" → "ford 150", "honda crv" → "honda crv"
    Bare make only ("Toyota", "Honda") → null
    Vague ("my car", "a sedan") → null
    null if transport is not "driving"
- num_travelers: integer or null. Do NOT extract if state.capacity_warned is active
- num_cars: extract ONLY when user explicitly states car count ("2 cars", "just one car")
- budget: number in USD, null if not mentioned

Return ONLY valid JSON with the changed fields (omit fields that did not change):
{{
  "destination": "city name or null",
  "is_international": true/false,
  "origin": "departure city or null",
  "num_days": integer or null,
  "budget": number or null,
  "budget_type": "per_person" or "total" or null,
  "vibe": "comma-separated vibes or null",
  "transport": "flying" or "driving" or null,
  "vehicle": "as user typed or null",
  "num_travelers": integer or null,
  "num_cars": integer or null,
  "cuisine_prefs": "cuisine types or null",
  "accommodation_pref": "hostel/budget hotel/airbnb/camping or null"
}}

Conversation context:
{transcript}""".format(
        state_json=json.dumps(_state_summary, indent=2),
        latest=latest_user_msg,
        transcript=transcript,
    )
    try:
        r = call_llm(llm_precise, [SystemMessage(content=prompt)])
        raw = extract_text(r).strip().replace("```json", "").replace("```", "")
        extracted = json.loads(raw)
        updates = {}
        if not state.get("destination"):
            if not extracted.get("is_international") and extracted.get("destination"):
                updates["destination"] = extracted["destination"].title()

        # Capture international attempt, BUT only if we're not already mid-redirect.
        # If _intl_destination is set the user is responding to the vibe menu — don't re-trigger.
        # If stage == "done" (completed trip) always re-trigger so a new destination starts fresh.
        _already_in_intl_flow = bool(state.get("_intl_destination")) and state.get("stage") != "done"
        if extracted.get("is_international") and not _already_in_intl_flow:
            updates["_international_attempt"] = True
            if extracted.get("destination"):
                _parsed_dest = extracted["destination"].title()
                # Fix 3: verify the parsed destination matches what the user actually said.
                # If the LLM hallucinated a different country, extract directly from the message.
                _msg_words = latest_user_msg.lower().split()
                _dest_words = _parsed_dest.lower().split()
                _matches = any(w in latest_user_msg.lower() for w in _dest_words)
                if not _matches:
                    # Hallucination detected — use the last meaningful word(s) from user message
                    _candidate = " ".join(
                        w for w in _msg_words
                        if w not in ("i", "want", "to", "go", "visit", "travel", "the", "a", "an",
                                     "would", "like", "please", "we", "us", "our", "let's", "lets")
                    ).strip().title()
                    _parsed_dest = _candidate or _parsed_dest
                updates["_intl_destination"] = _parsed_dest
        if _clean(extracted.get("origin")):
            _origin_val = _clean(extracted["origin"]).title()
            # Never set origin == destination (parse error)
            if _origin_val.lower() != (updates.get("destination") or state.get("destination") or "").lower():
                updates["origin"] = _origin_val
        if extracted.get("num_days"):        updates["num_days"]        = int(extracted["num_days"])
        if extracted.get("budget"):          updates["budget"]          = float(str(extracted["budget"]).replace("$", "").replace(",", ""))
        if _clean(extracted.get("vibe")):    updates["vibe"]            = _clean(extracted["vibe"]).lower()
        if _clean(extracted.get("transport")): updates["transport"]     = _clean(extracted["transport"]).lower()
        if _clean(extracted.get("vehicle")):
            _raw_vehicle = _clean(extracted["vehicle"])
            _raw_lower   = _raw_vehicle.strip().lower()
            if len(_raw_vehicle.split()) >= 2:
                # Multi-word — infer directly from extracted text
                _inferred = infer_full_vehicle(_raw_lower)
                updates["vehicle"] = _inferred if _inferred else _raw_vehicle.title()
            elif _raw_lower in _BARE_MAKES or _raw_lower in _MAKE_ABBREVS:
                # Parse got bare make (e.g. "Toyota" from "toy cam") — also try
                # the full user message in case model abbreviation was dropped
                _inferred = infer_full_vehicle(latest_user_msg.strip().lower())
                if not _inferred:
                    _inferred = infer_full_vehicle(_raw_lower)
                updates["vehicle"] = _inferred if _inferred else _raw_vehicle.title()
            else:
                updates["vehicle"] = _raw_vehicle.title()
        # num_travelers guard:
        # 1. Never overwrite if capacity_warned is active (prevents echo loop)
        # 2. Never wipe an existing value with null (preserve what was set during capacity resolution)
        _extracted_nt = extracted.get("num_travelers")
        if _extracted_nt and not state.get("_capacity_warned"):
            updates["num_travelers"] = int(_extracted_nt)
        # If LLM returned null but state already has a value, keep the existing value (no update needed)
        # num_cars: only update when capacity is already resolved or not yet in play
        if extracted.get("num_cars"):
            if not state.get("_capacity_warned") or state.get("_capacity_exceeded") is not None:
                updates["num_cars"] = int(extracted["num_cars"])
            # Do NOT set _capacity_exceeded here — only clarify_node's explicit capacity
            # resolution should set it. A bare number like "6" can be misread as num_cars.
        # A bare dollar amount like "$700 budget" must NOT set budget_type — clarify_node will ask.
        _btype_keywords = [
            "per person", "per head", "each person", "each of us", "a person", "per individual",
            "for each", "each one", "pp",
            "total", "whole group", "all of us", "for the group", "for everyone", "combined",
        ]
        if _clean(extracted.get("budget_type")) and any(
            kw in latest_user_msg.lower() for kw in _btype_keywords
        ):
            _bt_raw = _clean(extracted["budget_type"]).lower().strip()
            # Normalize aliases → canonical values
            if _bt_raw in ("group", "whole group", "for the group", "all of us",
                           "everyone", "total", "combined", "together"):
                _bt_raw = "total"
            elif _bt_raw in ("each", "each person", "per head", "per individual",
                             "per person", "pp", "a person", "for each"):
                _bt_raw = "per_person"
            updates["budget_type"] = _bt_raw
        # cuisine_prefs: state-aware prompt normalizes "all"/"everything" → "all cuisines"
        if _clean(extracted.get("cuisine_prefs")):
            updates["cuisine_prefs"] = _clean(extracted["cuisine_prefs"])
        if _clean(extracted.get("accommodation_pref")): updates["accommodation_pref"] = _clean(extracted["accommodation_pref"])
        logger.info("Parsed: %s", updates)
        return updates
    except Exception as e:
        logger.warning("Parse failed: %s", e)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# CLARIFY NODE
# ══════════════════════════════════════════════════════════════════════════════

def _msg_text(m) -> str:
    """Safely extract string content from a message dict — handles list content blocks."""
    c = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
    if isinstance(c, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c or "")

@traceable(name="clarify_node")
def clarify_node(state: TravelState) -> dict:

    # ── CAPACITY GUARDRAIL — checked FIRST before any other logic ─────────────
    # Must run before intent detection, vibe discovery, or the clarify LLM.
    _cv_vehicle = state.get("vehicle", "")
    _cv_nt      = state.get("num_travelers")
    if (
        state.get("transport") == "driving"
        and _cv_nt
        and _cv_vehicle
        and is_vehicle_complete(_cv_vehicle)
        and state.get("_capacity_exceeded") is None
    ):
        _cap_data  = get_vehicle_capacity(_cv_vehicle)
        _max_seats = _cap_data.get("capacity", 5)
        if _cv_nt > _max_seats:
            if not state.get("_capacity_warned"):
                # First time — show the warning
                return {
                    "_capacity_warned": True,
                    "messages": [{"role": "assistant", "content":
                        f"Quick heads up — a {_cv_vehicle} seats {_max_seats} people max, "
                        f"but you have {_cv_nt} in your group! How are you planning to handle this?\n\n"
                        f"- 🚗 Take **2 or more cars**\n"
                        f"- 🚐 Rent a **minivan or larger vehicle**\n"
                        f"- 👥 Fewer people making the trip"}],
                }
            else:
                # TURN 2+: User saw the warning — parse their resolution response
                _messages   = state.get("messages", [])
                _user_msgs  = [_msg_text(m) for m in _messages if m["role"] == "user"]
                _last_reply = _user_msgs[-1] if _user_msgs else ""

                # Guard: if last user message is just the group-size number,
                # user has not answered the capacity question yet.
                # Count prior warnings to detect echo loops.
                _warning_count = sum(
                    1 for _m in _messages
                    if _m.get("role") == "assistant"
                    and f"{_cv_vehicle} seats {_max_seats}" in _msg_text(_m)
                )
                _is_group_size_echo = (
                    _last_reply.strip().lstrip("-").isdigit()
                    and int(_last_reply.strip().lstrip("-")) == _cv_nt
                )
                if _is_group_size_echo:
                    if _warning_count >= 2:
                        # Escalate: user keeps echoing — force numbered choice
                        return {
                            "_capacity_warned": True,
                            "messages": [{"role": "assistant", "content": (
                                f"Just to be clear \u2014 a {_cv_vehicle} fits {_max_seats} people max, "
                                f"but you have {_cv_nt}. Please choose one:\n\n"
                                f"1\ufe0f\u20e3 We'll take 2+ cars\n"
                                f"2\ufe0f\u20e3 We'll rent a bigger vehicle\n"
                                f"3\ufe0f\u20e3 Only {_max_seats} people are coming"
                            )}],
                        }
                    else:
                        return {
                            "_capacity_warned": True,
                            "messages": [{"role": "assistant", "content": (
                                f"Just to confirm \u2014 your {_cv_vehicle} seats {_max_seats} people max "
                                f"but you have {_cv_nt} in your group. How would you like to handle it?\n\n"
                                "- \U0001f697 Take **2 or more cars**\n"
                                "- \U0001f690 Rent a **minivan or larger vehicle**\n"
                                "- \U0001f465 Fewer people making the trip"
                            )}],
                        }
                try:
                    _ir = call_llm(llm_precise, [SystemMessage(content=f"""
A group of {_cv_nt} people are driving in a {_cv_vehicle} that seats {_max_seats}.
They were asked how they plan to handle the capacity issue.
User replied: "{_last_reply}"

Return ONLY JSON:
{{
  "decision": "two_plus_cars" | "one_car_anyway" | "fewer_people" | "larger_vehicle" | "unclear",
  "num_cars": integer or null,
  "new_traveler_count": integer or null
}}
- two_plus_cars: taking multiple cars
- one_car_anyway: insisting on 1 car
- fewer_people: reducing group size
- larger_vehicle: getting a bigger vehicle
- unclear: cannot determine""")])
                    _intent_data  = json.loads(extract_text(_ir).strip().replace("```json","").replace("```",""))
                    _decision     = _intent_data.get("decision", "unclear")
                    _det_cars     = _intent_data.get("num_cars")
                    _new_count    = _intent_data.get("new_traveler_count")
                except Exception:
                    _decision = "unclear"; _det_cars = None; _new_count = None

                # Build next question for after resolution
                _dest = state.get("destination", "your destination")
                if not state.get("num_days"):        _nxt = "How many days are you staying?"
                elif not state.get("budget"):        _nxt = "What's your total budget — per person or for the whole group?"
                elif not state.get("vibe"):          _nxt = f"What's your vibe? Options: {build_vibe_options(_dest)}"
                elif not state.get("accommodation_pref"): _nxt = "What accommodation do you prefer — hostel, budget hotel, Airbnb, or camping?"
                else:                                _nxt = "Anything else to add?"

                if _decision == "two_plus_cars":
                    _nc = _det_cars or 2
                    if _nc * _max_seats >= _cv_nt:
                        return {"num_cars": _nc, "_capacity_exceeded": False, "_capacity_warned": False,
                                "messages": [{"role": "assistant", "content":
                                    f"{_nc} cars works perfectly for {_cv_nt} people! "
                                    f"Gas costs will be split. {_nxt}"}]}
                    else:
                        _needed = -(-_cv_nt // _max_seats)
                        return {"messages": [{"role": "assistant", "content":
                            f"You'd need at least **{_needed} cars** for {_cv_nt} people. How many are you taking?"}]}
                elif _decision == "one_car_anyway":
                    return {"num_cars": 1, "_capacity_exceeded": True, "_capacity_warned": True,
                            "messages": [{"role": "assistant", "content":
                                f"Just a heads up — your {_cv_vehicle} seats {_max_seats} max. "
                                f"For safety, a bigger vehicle or second car is strongly recommended. "
                                f"Are you taking a second car, renting a larger vehicle, or going with fewer people?"}]}
                elif _decision == "fewer_people":
                    _uc = _new_count or _max_seats
                    if _uc <= _max_seats:
                        return {"num_travelers": _uc, "num_cars": 1, "_capacity_exceeded": False, "_capacity_warned": False,
                                "messages": [{"role": "assistant", "content":
                                    f"Got it — {_uc} people, one {_cv_vehicle}! {_nxt}"}]}
                    return {"messages": [{"role": "assistant", "content": "How many people are actually making the trip?"}]}
                elif _decision == "larger_vehicle":
                    return {"num_cars": 1, "_capacity_exceeded": False, "_capacity_warned": False,
                            "messages": [{"role": "assistant", "content":
                                f"A larger vehicle — smart move for {_cv_nt} people! {_nxt}"}]}
                else:
                    # unclear — re-show the full options so user can give a clear answer
                    return {"_capacity_warned": True,
                            "messages": [{"role": "assistant", "content": (
                                f"Just to confirm \u2014 your {_cv_vehicle} seats {_max_seats} people max, "
                                f"but you have {_cv_nt} in your group. How would you like to handle it?\n\n"
                                f"- \U0001f697 Take **2 or more cars**\n"
                                f"- \U0001f690 Rent a **minivan or larger vehicle**\n"
                                f"- \U0001f465 Fewer people making the trip"
                            )}]}

    # International redirect — two-step flow
    # Step 1: Ask WHAT draws them to that country (show experience/vibe options)
    # Step 2: Handled by vibe-discovery block — uses _intl_destination for context
    if state.get("_international_attempt"):
        user_msgs   = [m["content"] for m in state.get("messages", []) if m["role"] == "user"]
        last_msg    = user_msgs[-1] if user_msgs else ""
        intl_dest   = state.get("_intl_destination") or "that destination"

        # Detect whether user is asking to TRAVEL or LEARN about the place.
        # A bare country/city name always = travel. Only explicit question phrasing = learn.
        try:
            _mode_r = call_llm(llm_precise, [SystemMessage(content=
                f'User said: "{last_msg}"\n'
                'Does this contain a clear QUESTION or curiosity marker '
                '(e.g. what is, tell me about, how is, why is, explain)?\n'
                'A bare country/city name ("india", "France", "Japan") is NOT a question.\n'
                '"I want to go" is NOT a question. "I love" is NOT a question.\n'
                'Answer ONLY "learn" if there is an explicit question word. Otherwise answer "travel".')])
            _mode = extract_text(_mode_r).strip().lower()
            if _mode not in ("travel", "learn"):
                _mode = "travel"
        except Exception:
            _mode = "travel"

        if _mode == "learn":
            r = call_llm(llm, [SystemMessage(content=f"""The user is curious about {intl_dest}.
TripBuddy can share knowledge but only plans US trips.
User said: "{last_msg}"
Format:
1. 2-3 enthusiastic sentences answering their specific question about {intl_dest}
2. Bridge: "While TripBuddy only plans US trips, here are some US spots with a similar vibe:"
3. 3-4 bullets: **City/Neighborhood** — what makes it match what they asked about
4. End: "Want to plan a trip to one of these, or somewhere else in mind?"
Be specific and warm.""")])
            return {
                "stage": "clarifying",
                "_international_attempt": False,
                "messages": [{"role": "assistant", "content": extract_text(r)}],
            }

        # Travel intent — check if the user already stated a specific vibe/activity.
        # e.g. "wine tasting in Bordeaux" or "food tour in Tokyo" → vibe is clear → skip to US suggestions.
        # If vibe is ambiguous → Step 1: ask what draws them there.
        try:
            _vibe_r = call_llm(llm_precise, [SystemMessage(content=
                f'User said: "{last_msg}"\n'
                f'Destination: {intl_dest}\n'
                'Did the user already mention a SPECIFIC activity, experience, or vibe they want '
                '(e.g. wine tasting, food tour, history museums, hiking, nightlife)?\n'
                'If yes, extract it as a short phrase. If no, return "none".\n'
                'Answer ONLY the vibe phrase or "none". No other text.')])
            _stated_vibe = extract_text(_vibe_r).strip().lower()
            if _stated_vibe == "none":
                _stated_vibe = ""
        except Exception:
            _stated_vibe = ""

        if _stated_vibe:
            # Vibe already stated — skip step 1, go straight to US suggestions
            r = call_llm(llm, [SystemMessage(content=
                f'The user wants {_stated_vibe} experiences like those in {intl_dest}. TripBuddy only plans US trips.\n'
                f'Suggest 3-4 US cities/regions that offer an exceptional {_stated_vibe} experience comparable to {intl_dest}.\n'
                'Format:\n'
                f'1. One warm sentence acknowledging their love of {_stated_vibe} and {intl_dest}\n'
                f'2. "TripBuddy is US-only, but here are the best US spots for {_stated_vibe}:"\n'
                f'3. 3-4 bullets: **City/Region, State** — what makes it a great {_stated_vibe} destination\n'
                '4. End: "Which one catches your eye, or somewhere else in mind?"\n'
                f'Be specific. Compare directly to what makes {intl_dest} special for {_stated_vibe}.')])
            return {
                "stage":              "clarifying",
                "_international_attempt": False,
                "_intl_destination":  intl_dest,
                "vibe":               _stated_vibe,
                "messages":           [{"role": "assistant", "content": extract_text(r)}],
            }

        # Vibe not stated → Step 1: ask what draws them to this destination
        r = call_llm(llm, [SystemMessage(content=f"""The user said: "{last_msg}"
They want to visit {intl_dest}. TripBuddy only plans US trips.

CRITICAL: This response is EXCLUSIVELY about {intl_dest}.
Every sentence and every bullet must be about {intl_dest} — not any other country.
If you find yourself writing about Mexico when {intl_dest} is India, stop and rewrite.

Format:
1. 1-2 warm, specific sentences about what makes {intl_dest} special
   (culture, cuisine, history, landscapes, arts unique to {intl_dest})
2. Line: "What draws you most to {intl_dest}? Pick what resonates:"
3. 4-6 bullet points with emoji — real, iconic experiences that {intl_dest} is famous for.
   Each bullet must be unmistakably and exclusively about {intl_dest}.
4. Final line: "Once I know your vibe, I'll find the best US spots that recreate that experience!"

Do NOT mention any other country. Every bullet is about {intl_dest} only.""")])

        return {
            "stage":              "clarifying",
            "_international_attempt": False,
            "_intl_destination":  intl_dest,
            "messages":           [{"role": "assistant", "content": extract_text(r)}],
        }

    # Mid-conversation intent detection.
    # Handles three cases when a destination is already set:
    #   1. User is answering the current question → continue normally
    #   2. User wants a NEW US destination → reset state
    #   3. User is asking about an international place (learn/curiosity) → answer + stay US-only
    #   4. User's vibe/interest has shifted → update vibe and continue
    # Guard: only runs when the assistant has already spoken (avoids false resets on first message).
    # Intent detection only runs once destination AND transport are both known.
    # During initial Q&A (collecting origin, transport, travelers, etc.) the user
    # is always just answering questions — no need to classify intent, and
    # misclassification causes vibes/experiences to be asked out of order.
    if state.get("destination") and state.get("transport"):
        messages  = state.get("messages", [])
        user_msgs = [_msg_text(m) for m in messages if m.get("role") == "user"]
        asst_msgs = [_msg_text(m) for m in messages if m.get("role") == "assistant"]
        latest    = user_msgs[-1] if user_msgs else ""
        last_q    = asst_msgs[-1] if asst_msgs else ""
        if asst_msgs and latest:
            try:
                # ── Unified router: one LLM call replaces 4 separate classifiers ──
                _state_ctx = {
                    "destination": state.get("destination"),
                    "transport":   state.get("transport"),
                    "vibe":        state.get("vibe"),
                    "stage":       state.get("stage"),
                }
                intent_r = call_llm(llm_precise, [SystemMessage(content=f"""
You are a router for a US-only trip planner.

Current trip state: {json.dumps(_state_ctx)}
Last assistant message: "{last_q[:200]}"
User replied: "{latest}"

Classify the user's intent AND extract any new fields in one step.

Return ONLY JSON:
{{
  "intent": "answer" | "new_us" | "intl_learn" | "vibe_shift",
  "new_destination": "city name if switching US destinations, else null",
  "new_vibe": "updated vibe if shifting, else null",
  "intl_place": "international place name if curious about, else null"
}}

Intent rules:
- "answer": user is replying normally to the question asked
- "new_us": user wants to plan a DIFFERENT US city (e.g. "actually let's go to Miami")
- "intl_learn": user is asking about or curious about a non-US place
- "vibe_shift": user wants to change their travel style/vibe mid-conversation
- Default to "answer" when ambiguous
""")])
                _router = json.loads(extract_text(intent_r).strip().replace("```json","").replace("```",""))
                _intent      = _router.get("intent", "answer").strip().lower()
                _new_dest    = _router.get("new_destination")
                _new_vibe    = _router.get("new_vibe")
                _intl_place  = _router.get("intl_place")

                if _intent == "new_us":
                    return {
                        "stage": "clarifying",
                        "destination": None, "origin": None,
                        "num_days": None, "budget": None, "vibe": None,
                        "transport": None, "num_travelers": None, "budget_type": None,
                        "cuisine_prefs": None, "vehicle": None, "accommodation_pref": None,
                        "num_cars": None, "transport_cost": None, "itinerary": None,
                        "_international_attempt": False, "_capacity_exceeded": None, "_capacity_warned": False,
                        "messages": [{"role": "user", "content": latest}],
                    }

                elif _intent == "intl_learn":
                    # Use intl_place from router if available, else infer from message
                    _about = _intl_place or "that place"
                    _r = call_llm(llm, [SystemMessage(content=f"""
The user is curious about {_about} while planning a US trip to {state.get("destination")}.
TripBuddy can share knowledge but only plans US trips.
User said: "{latest}"

Format:
1. 2-3 warm sentences about {_about}
2. Bridge: "That said, TripBuddy is US-only — but {state.get("destination")} has some similar magic:"
3. 1-2 sentences connecting to their current US destination
4. Ask: "Want to keep planning your {state.get("destination")} trip, or explore a different US city?"

Be specific and warm.""")])
                    return {
                        "stage": "clarifying",
                        "messages": [{"role": "assistant", "content": extract_text(_r)}],
                    }

                elif _intent == "vibe_shift":
                    # Use new_vibe extracted by router; fall back to raw message
                    _updated_vibe = (_new_vibe or latest).lower().strip()
                    return {
                        "stage": "clarifying",
                        "vibe": _updated_vibe,
                        "messages": [{"role": "assistant", "content":
                            f"Shifting the vibe to {_updated_vibe} — love it! "
                            f"Let me adjust your {state.get('destination')} plans."}],
                    }
                # else: "answer" → fall through to normal clarify flow

            except Exception:
                pass

    destination = state.get("destination", "")

    # ── Vibe-based city discovery ─────────────────────────────────────────────
    # When destination is missing but the user said a vibe/preference word
    # (e.g. "food", "beach", "nightlife") instead of a city name, suggest
    # matching US cities rather than blindly asking "where do you want to go?"
    if not state.get("destination"):
        _messages  = state.get("messages", [])
        _user_msgs = [_msg_text(m) for m in _messages if m["role"] == "user"]
        _asst_msgs = [_msg_text(m) for m in _messages if m["role"] == "assistant"]
        _latest    = _user_msgs[-1].strip() if _user_msgs else ""
        if _asst_msgs and _latest:          # only mid-conversation, not on first message
            try:
                _intent = call_llm(llm_precise, [SystemMessage(content=
                    f'User said: "{_latest}"\n'
                    'Is this a US city/state/region name, or a travel vibe/preference '
                    '(e.g. food, beach, culture, nightlife, outdoors, history)?\n'
                    'Answer ONLY "location" or "vibe".')])
                if extract_text(_intent).strip().lower().startswith("vibe"):
                    _intl_ctx = state.get("_intl_destination")
                    if _intl_ctx:
                        # User picked a vibe after the international redirect —
                        # suggest US cities that match BOTH the country AND the vibe
                        _vibe_prompt = (
                            f'The user wanted to visit {_intl_ctx} and their vibe is "{_latest}".\n'
                            f'Suggest 3-4 specific US cities/neighborhoods that best recreate the {_latest}\n'
                            f'experience of {_intl_ctx}. Be specific.\n'
                            'Format:\n'
                            f'1. Opening line referencing {_intl_ctx} and {_latest}\n'
                            f'2. 3-4 bullets: **City, State** — what specifically makes it match {_intl_ctx} {_latest} vibe\n'
                            '3. End: "Which one would you like to explore, or somewhere else in mind?"'
                        )
                    else:
                        _vibe_prompt = (
                            f'The user wants a "{_latest}"-focused trip but hasn\'t picked a destination.\n'
                            f'Suggest 3-4 specific US cities or neighborhoods that are exceptional for "{_latest}" travel.\n'
                            'Include a mix of well-known and unexpected picks.\n'
                            'Format:\n'
                            f'1. One casual sentence: "Love that vibe!..."\n'
                            f'2. 3-4 bullets: **City, State** — what makes it the best for {_latest}\n'
                            '3. End: "Which one catches your eye, or somewhere else in mind?"'
                        )
                    _r = call_llm(llm, [SystemMessage(content=_vibe_prompt)])
                    return {
                        "stage":             "clarifying",
                        "vibe":              _latest.lower(),
                        "_intl_destination": _intl_ctx,   # carry context forward
                        "messages":          [{"role": "assistant", "content": extract_text(_r)}],
                    }
            except Exception:
                pass

    # Build missing fields list — each entry is the EXACT question to ask the user
    _dest = destination or "your destination"
    missing = []
    if not state.get("destination"):   missing.append("Which US city, state, or region are you thinking about visiting?")
    if not state.get("origin"):        missing.append("Where are you traveling from?")
    if not state.get("transport"):     missing.append(f"Are you flying or driving to {_dest}?")

    # Vehicle check — LLM determines if make+model are both present
    if state.get("transport") == "driving":
        vehicle = state.get("vehicle", "")
        if not vehicle or not is_vehicle_complete(vehicle):
            if vehicle:
                # Vehicle was given but appears incomplete or typo — ask for full make+model
                missing.append(f"vehicle_model:{vehicle}")
            else:
                missing.append("What car are you driving? I'll need both the make and model (e.g. Toyota Camry, Honda CR-V).")
    # Continue collecting remaining missing fields
    if not state.get("num_travelers"):
        # If user is flying but typed a vehicle name, gently clarify
        _latest_user = state.get("messages", [])
        _latest_user_msg = next(
            (m.get("content", "") for m in reversed(_latest_user) if m.get("role") == "user"), ""
        )
        if (
            state.get("transport") == "flying"
            and state.get("vehicle")
            and isinstance(_latest_user_msg, str)
            and any(make in _latest_user_msg.lower() for make in list(_BARE_MAKES)[:8])
        ):
            missing.append(
                f"Heads up — since you're flying, you don't need to specify a car! "
                f"How many people are going on this trip?"
            )
        else:
            missing.append("How many people are going on this trip?")
    if not state.get("num_days"):        missing.append(f"How many days are you planning to spend in {_dest}?")
    _solo = state.get("num_travelers") == 1
    if not state.get("budget"):
        missing.append(
            "What's your total budget for the trip?"
            if _solo else
            "What's your budget for the trip — and is that per person or for the whole group?"
        )
    elif not _solo and state.get("budget"):
        _bt = (state.get("budget_type") or "").lower().strip()
        if _bt not in ("per_person", "total"):
            _budget = state.get("budget", 0)
            missing.append("Quick clarification — is the $" + str(int(_budget)) + " budget per person or for the whole group of you?")
    if not state.get("vibe"):            missing.append("__VIBE__")  # resolved to options list below
    if state.get("vibe") and "food" in state.get("vibe", "").lower() and not state.get("cuisine_prefs"):
        missing.append("__CUISINE__")
    if not state.get("accommodation_pref"):
        missing.append("What accommodation do you prefer — hostel, budget hotel, Airbnb, or camping?")

    if not missing:
        logger.info("All fields collected — running early budget check before generating")

        # ── Early budget guardrail ──────────────────────────────────────────
        # Runs BEFORE itinerary generation or any transport/cost calculation.
        # Uses evaluate_budget_guardrail with no itinerary yet (empty string).
        early_guardrail = evaluate_budget_guardrail(
            budget        = state.get("budget", 0),
            num_days      = state.get("num_days", 1),
            num_travelers = state.get("num_travelers", 1),
            transport     = state.get("transport", "flying"),
            destination   = destination,
            itinerary     = "",
            budget_type   = state.get("budget_type", "total"),
        )

        verdict    = early_guardrail.get("verdict", "green")
        assessment = early_guardrail.get("assessment", "")
        per_day    = early_guardrail.get("per_person_per_day", "?")
        min_budget = early_guardrail.get("realistic_minimum", "?")

        if verdict == "red":
            warning_msg = (
                f"⚠️ **Budget heads-up before we dive in** — {assessment}\n\n"
                f"${per_day}/person/day · Realistic minimum for this trip: **${min_budget}**\n\n"
            )
            if early_guardrail.get("tips"):
                warning_msg += "**Quick tips to make it work:**\n"
                warning_msg += "\n".join(f"- {t}" for t in early_guardrail["tips"])
            warning_msg += "\n\nGenerating your itinerary now — you can always adjust the budget after!"
            return {
                "stage":           "generating",
                "budget_guardrail": early_guardrail,
                "messages":        [{"role": "assistant", "content": warning_msg}],
            }

        elif verdict == "yellow":
            warning_msg = (
                f"💛 **Budget note** — {assessment} "
                f"(${per_day}/person/day)\n\n"
                f"Generating your itinerary with money-saving options built in!"
            )
            return {
                "stage":           "generating",
                "budget_guardrail": early_guardrail,
                "messages":        [{"role": "assistant", "content": warning_msg}],
            }

        # Green — proceed silently, guardrail stored in state for later display
        return {"stage": "generating", "budget_guardrail": early_guardrail}

    known = {k: v for k, v in {
        "destination":   destination,
        "origin":        state.get("origin"),
        "transport":     state.get("transport"),
        "vehicle":       state.get("vehicle"),
        "cars":          f"{state.get('num_cars')} cars" if state.get("num_cars") else None,
        "travelers":     f"{state.get('num_travelers')} people" if state.get("num_travelers") else None,
        "days":          state.get("num_days"),
        "budget":        f"${state.get('budget')} {state.get('budget_type', '')}".strip() if state.get("budget") else None,
        "vibe":          state.get("vibe"),
        "cuisine":       state.get("cuisine_prefs"),
        "accommodation": state.get("accommodation_pref"),
    }.items() if v}

    # Resolve next_q from missing[0]
    first_missing = missing[0] if missing else ""

    if first_missing == "__VIBE__":
        _vibe_opts = build_vibe_options(destination) if destination else "culture, food, nightlife, outdoors, road trip, beach — pick as many as you want!"
        next_q = f"What's your vibe for {_dest}? Pick as many as you like: {_vibe_opts}"

    elif first_missing == "__CUISINE__":
        next_q = build_cuisine_question(destination) if destination else "What cuisines are you into? Type whatever you're craving!"

    elif first_missing.startswith("vehicle_model:"):
        bare_make = first_missing.split("vehicle_model:", 1)[1]
        next_q = get_vehicle_model_prompt(bare_make)

    else:
        next_q = first_missing  # already a ready-to-ask question

    # ── Question delivery ─────────────────────────────────────────────────────
    # For vibe and cuisine questions, use the LLM to add city-specific warmth.
    # For ALL other questions, return next_q directly — no LLM, no freestyle.
    _needs_llm = first_missing in ("__VIBE__", "__CUISINE__") or first_missing.startswith("vehicle_model:")

    if _needs_llm:
        system = """You are TripBuddy, a friendly US budget travel planner for college students.
US trips only. Be casual and warm.

CORE QUESTION TO DELIVER (preserve ALL options and choices exactly):
{next_q}

Rules:
- Include EVERY option listed — word for word
- ALWAYS end with a clear question mark — the user must know what to answer
- Ask ONLY this question, nothing else
- Do NOT summarize or conclude — your job is to ASK, not to answer
- Do NOT mention flights, accommodations, itineraries, or anything outside this question""".format(next_q=next_q)

        msgs = [SystemMessage(content=system)]
        for m in state.get("messages", [])[-6:]:
            if m["role"] == "user":        msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant": msgs.append(AIMessage(content=m["content"]))
        r = call_llm(llm, msgs)
        reply = extract_text(r)
    else:
        # Return question directly — LLM creativity causes off-script responses for simple Qs
        _warmers = ["Got it!", "Perfect!", "Awesome!", "Great!", "Nice!"]
        import random as _random
        reply = f"{_random.choice(_warmers)} {next_q}"

    return {"stage": "clarifying", "messages": [{"role": "assistant", "content": reply}]}

def sanitize_state(state: TravelState) -> dict:
    """Return safe values for all arithmetic fields — never None.
    Use at the top of generate_node, revise_node, and any node doing math on state.
    """
    return {
        "num_travelers":     state.get("num_travelers") or 1,
        "num_cars":          state.get("num_cars") or 1,
        "num_days":          state.get("num_days") or 1,
        "budget":            state.get("budget") or 0,
        "budget_type":       (state.get("budget_type") or "total").lower().strip(),
        "transport":         state.get("transport") or "driving",
        "destination":       state.get("destination") or "your destination",
        "vibe":              state.get("vibe") or "general",
        "accommodation_pref": state.get("accommodation_pref") or "budget hotel",
        "vehicle":           state.get("vehicle") or "",
        "origin":            state.get("origin") or "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# BUDGET GUARDRAIL
# ══════════════════════════════════════════════════════════════════════════════
@traceable(name="evaluate_budget_guardrail")
def evaluate_budget_guardrail(
    budget: float,
    num_days: int,
    num_travelers: int,
    transport: str,
    destination: str,
    itinerary: str,
    budget_type: str = "total",
) -> dict:
    """
    Uses LLM to evaluate whether the budget is realistic for a college student.
    Returns a dict with: status, verdict, assessment, warnings, tips,
                         per_person_per_day, realistic_minimum, total_budget, per_person
    """
    # Sanitize all inputs — callers may pass state.get(...) which can be None
    budget        = budget or 0
    num_days      = num_days or 1
    num_travelers = num_travelers or 1
    budget_type   = (budget_type or "total").lower().strip()
    total              = budget if budget_type == "total" else budget * num_travelers
    per_person         = round(total / num_travelers, 2) if num_travelers > 0 else total
    per_person_per_day = round(per_person / num_days, 2) if num_days > 0 else per_person

    prompt = f"""You are a college student travel budget advisor. Evaluate whether this trip budget is realistic.

TRIP DETAILS:
- Destination: {destination}
- Duration: {num_days} days
- Travelers: {num_travelers} people
- Transport: {transport}
- Total budget: ${total}
- Per person: ${per_person}
- Per person per day: ${per_person_per_day}

COLLEGE STUDENT BUDGET BENCHMARKS:
- Typical spring break trip (5-7 days): $800-$1,200 per person total
- Accommodation: $150-$500 per person for the trip
- Transportation: $200-$600 per person
- Food: $100-$300 per person ($15-$40/day)
- Entertainment: $50-$100 per person
- Absolute minimum viable daily budget: $50/person/day (hostel + cheap food + free activities)
- Comfortable student budget: $100-$150/person/day

The trip itinerary:
{itinerary[:1500]}

Evaluate the budget and return ONLY JSON:
{{
  "status": "healthy" | "tight" | "too_low" | "over_budget",
  "per_person_per_day": {per_person_per_day},
  "assessment": "one sentence assessment of whether this budget is realistic",
  "warnings": ["specific warning 1", "specific warning 2"],
  "tips": ["actionable tip 1", "actionable tip 2", "actionable tip 3"],
  "realistic_minimum": number (minimum realistic total budget for this trip),
  "verdict": "green" | "yellow" | "red"
}}

Status guide:
- "healthy": budget is comfortable ($100+/person/day for travel; covers transport, accommodation, food, activities)
- "tight": doable but requires careful spending ($60-$100/person/day — hostels, cheap eats, free activities)
- "too_low": budget is unrealistic (<$60/person/day, OR transport alone exceeds total budget)
- "over_budget": the itinerary as written costs more than the stated budget

Important calibration — be ACCURATE, not conservative:
- $100+/person/day for a US trip = GREEN (comfortable). Do not flag as tight.
- $125/day = GREEN. $500/person for 4 days Nashville = GREEN.
- "Tight" (yellow) only applies if $60-$100/person/day AND there's a specific cost concern (e.g. flight cost relative to budget).
- Red only if transport alone would consume >80% of budget, or <$60/person/day.
- "Realistic minimum" = cheapest possible (hostel $30-40/night, $15-20/day food, free activities). For a 4-day US domestic trip it is RARELY above $300-400 per person.
- If per_person_per_day >= 100, return verdict: "green".

Verdict guide:
- "green": budget is sufficient — proceed
- "yellow": tight but doable — flag specific concern (e.g. flight + Airbnb may be tight)
- "red": genuinely insufficient — e.g. flights alone would exceed the total budget"""

    try:
        r      = call_llm(llm_precise, [SystemMessage(content=prompt)])
        raw    = extract_text(r).strip().replace("```json", "").replace("```", "")
        result = json.loads(raw)
        result["total_budget"] = total
        result["per_person"]   = per_person
        return result
    except Exception as e:
        logger.warning("Budget guardrail failed: %s", e)
        return {
            "status":            "healthy",
            "verdict":           "green",
            "assessment":        "Budget looks reasonable.",
            "warnings":          [],
            "tips":              [],
            "per_person_per_day": per_person_per_day,
            "total_budget":      total,
            "per_person":        per_person,
            "realistic_minimum": total,
        }


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE NODE
# ══════════════════════════════════════════════════════════════════════════════
@traceable(name="generate_node")
def generate_node(state: TravelState) -> dict:
    _s            = sanitize_state(state)
    destination   = _s["destination"]
    logger.info("Generating for %s", destination)
    num_travelers = _s["num_travelers"]
    num_cars      = _s["num_cars"]
    budget        = _s["budget"]
    budget_type   = _s["budget_type"]
    num_days      = _s["num_days"]
    total_budget  = int(budget * num_travelers if budget_type == "per_person" else budget)
    per_person_budget = int(total_budget / num_travelers) if num_travelers > 1 else total_budget
    tb = f"${total_budget}"
    pp = f"${per_person_budget}"

    transport_cost    = state.get("transport_cost")
    transport_summary = ""
    if state.get("transport") == "driving":
        if transport_cost:
            fuel    = transport_cost.get("fuel", {})
            route   = transport_cost.get("route", {})
            parking = transport_cost.get("parking", {})
            tolls   = transport_cost.get("tolls", {})
            gas_total       = round(fuel.get("total", 0) * num_cars, 2)
            toll_total      = round(tolls.get("total", 0) * num_cars, 2)
            parking_total   = parking.get("estimated_total", 0)
            transport_total = round(gas_total + toll_total + parking_total, 2)
            transport_pp    = round(transport_total / num_travelers, 2)
            is_ev           = fuel.get("type") == "electric"
            fuel_line = (
                f"EV charging: ${gas_total} ({num_cars} car{'s' if num_cars > 1 else ''})"
                if is_ev else
                f"Gas: ${gas_total} ({num_cars} car{'s' if num_cars > 1 else ''} x {fuel.get('mpg', '~28')} mpg @ ${fuel.get('gas_price', 3.50)}/gal)"
            )
            transport_summary = f"""
REAL TRANSPORT COSTS (use exact numbers):
- Route: {route.get('miles', '?')} miles each way, ~{route.get('hours', '?')} hrs
- {num_cars} x {state.get('vehicle', 'car')} for {num_travelers} people
- {fuel_line} = ${gas_total} total (${round(gas_total / num_travelers, 2)}/person)
- Parking: ~${parking_total} (${round(parking_total / num_travelers, 2)}/person)
- Tolls: ~${toll_total} (${round(toll_total / num_travelers, 2)}/person)
- TOTAL TRANSPORT: ${transport_total} (${transport_pp}/person)"""
        else:
            transport_summary = (
                f"\nDriving {state.get('origin')} to {destination}, "
                f"{num_cars} car(s), {num_travelers} people — estimate realistic gas cost, NEVER say free."
            )

    rag_context = retrieve_travel_context(
        user_query="budget travel student {destination} hotels food activities {state.get('vibe', '')}",
        city=destination, category=_s["vibe"]
    )

    rag_section   = f"\n\nREAL DATA for {destination}:\n{rag_context}" if rag_context else ""
    capacity_note = "\n- Note: group may want to rent a minivan for more comfort" if state.get("_capacity_exceeded") else ""

    _traveler_word = "person" if num_travelers == 1 else "people"
    # $X and $Y are placeholders the LLM fills with real per-category amounts.
    # Only the Total row uses the actual computed values pp/tb.
    budget_table = (
        "| Category | Amount |\n"
        "|---|---|\n"
        "| Transport | $X |\n"
        "| Accommodation | $X |\n"
        "| Food | $X |\n"
        "| Activities | $X |\n"
        f"| **Total** | {pp} |"
    ) if num_travelers == 1 else (
        "| Category | Per Person | Group Total |\n"
        "|---|---|---|\n"
        "| Transport | $X | $Y |\n"
        "| Accommodation | $X | $Y |\n"
        "| Food | $X | $Y |\n"
        "| Activities | $X | $Y |\n"
        f"| **Total** | {pp} | {tb} |"
    )
    system = f"""US budget travel planner for college students.
DESTINATION = {destination} (plan trip HERE only)
Cuisine ({state.get('cuisine_prefs', '?')}) = what to EAT in {destination}, not a destination name.

Trip: {destination} | From: {state.get('origin')} | {state.get('transport')} | {num_cars} car(s) | {state.get('vehicle', '')}
{num_travelers} {_traveler_word} | {state.get('num_days')} days | {tb} total / {pp}/person
Vibe: {state.get('vibe')} | Accommodation: {state.get('accommodation_pref', '')}
{"Cuisines: " + state.get('cuisine_prefs', '') + " IN " + destination if state.get('cuisine_prefs') else ""}{transport_summary}{capacity_note}{rag_section}

STRICT BUDGET MATH — CRITICAL:
The budget table MUST satisfy: Transport + Accommodation + Food + Activities = {total_budget} exactly.
Fill in reasonable amounts for Transport, Accommodation, Food, Activities so they add up.
Example for $500 solo 4-day flying trip: Transport=$150, Accommodation=$160, Food=$120, Activities=$70 → Total=$500 ✓
Do NOT make Food > 40% of total budget. Do NOT let any single category dominate unrealistically.

RULES:
- Use EXACT transport costs — driving is NEVER free
- Dollar sign only before numbers: $15 $50 — NEVER $total $person
- Activity costs: "$X/person ($Y total)"
- Real restaurants in {destination}
- Camping far from city → nearest realistic campsite + distance
- Flag student discounts 🎓
- NEVER list flight or transport costs as day-by-day activities — transport goes ONLY in the Budget Breakdown table
- Accommodation MUST be non-zero for hostel/hotel/Airbnb (only $0 for camping)
- Day activities = sightseeing, meals, and local experiences ONLY — no travel logistics

## Your {state.get('num_days')}-Day {destination} Trip
**{num_travelers} {_traveler_word} · {tb} total · {pp}/person**

### Day 1 — [Theme]
**Morning:** Activity — $X/person ($Y total)
**Afternoon:** Activity — $X/person ($Y total)
**Evening:** Dinner at [Real Restaurant in {destination}] — $X/person ($Y total)
...continue each day...

### Budget Breakdown
{budget_table}

3 money-saving tips for {destination}"""

    r = call_llm(llm, [SystemMessage(content=system)])
    itinerary = extract_text(r)
    budget_exceeded = False
    try:
        totals = re.findall(r"\*\*Total\*\*.*?\$([0-9,]+)", itinerary)
        if totals:
            budget_exceeded = float(totals[-1].replace(",", "")) > total_budget
    except Exception:
        pass

    # Update guardrail with full itinerary context (more accurate than the early check).
    # Preserves the early guardrail if the full-context one fails.
    early_guardrail = state.get("budget_guardrail") or {}
    try:
        guardrail = evaluate_budget_guardrail(
            budget        = budget,
            num_days      = state.get("num_days", 1),
            num_travelers = num_travelers,
            transport     = state.get("transport", "flying"),
            destination   = destination,
            itinerary     = itinerary,
            budget_type   = budget_type,
        )
    except Exception:
        guardrail = early_guardrail

    return {
        "stage": "done",
        "itinerary": itinerary,
        "budget_exceeded": budget_exceeded or guardrail.get("status") in ("too_low", "over_budget"),
        "budget_guardrail": guardrail,
        "messages": [{"role": "assistant", "content": itinerary +
            "\n\n---\nItinerary ready! 🎉 Want to tweak anything — swap a hotel, change an activity, or adjust the budget?\n\n"
            "You can also use the buttons below to:\n"
            "- 🛒 **Grocery & Snacks List** — personalized to your group's dietary needs\n"
            "- ✅ **Travel Checklist** — everything to pack for this trip\n\n"
            "Or just say the word if you're ready to plan another trip!"}],
    }


# ══════════════════════════════════════════════════════════════════════════════
# REVISE NODE
# ══════════════════════════════════════════════════════════════════════════════
@traceable(name="revise_node")
def revise_node(state: TravelState) -> dict:
    user_msgs = [m for m in state.get("messages", []) if m["role"] == "user"]
    latest    = user_msgs[-1]["content"] if user_msgs else ""
    try:
        intent = call_llm(llm_precise, [SystemMessage(content=
            f'New trip to different destination or modifying current?\nCurrent: {state.get("destination")} | Message: "{latest}"\nAnswer only "new" or "modify".')])
        wants_new = extract_text(intent).strip().lower().startswith("new")
    except Exception:
        wants_new = False
    if wants_new:
        return {
            "stage": "clarifying",
            "destination": None, "origin": None, "num_days": None,
            "budget": None, "vibe": None, "transport": None, "num_travelers": None,
            "budget_type": None, "cuisine_prefs": None, "vehicle": None,
            "accommodation_pref": None, "num_cars": None, "transport_cost": None,
            "itinerary": None, "_international_attempt": False, "_capacity_exceeded": None, "_capacity_warned": False,
            "messages": [{"role": "user", "content": latest}],
        }

    # If user is switching to driving but hasn't provided a car, ask before regenerating
    _switching_to_driving = (
        "driv" in latest.lower()
        and state.get("transport") != "driving"
    )
    if _switching_to_driving and not is_vehicle_complete(state.get("vehicle") or ""):
        return {
            "transport": "driving",
            "vehicle": None,
            "messages": [{"role": "assistant", "content":
                "Switching to driving! What car are you taking? "
                "I'll need both the make and model (e.g. Toyota Camry, Honda CR-V) "
                "to calculate gas costs accurately."}],
        }

    # Detect if user is changing the budget so we can use a smarter prompt
    _s            = sanitize_state(state)
    budget        = _s["budget"]
    budget_type   = _s["budget_type"]
    num_travelers = _s["num_travelers"]
    num_days      = _s["num_days"]
    total_budget  = int(budget * num_travelers if budget_type == "per_person" else budget)
    per_person    = int(total_budget / num_travelers) if num_travelers > 1 else total_budget
    _traveler_word = "person" if num_travelers == 1 else "people"

    is_budget_change = any(
        kw in latest.lower()
        for kw in ["budget", "spend", "cost", "money", "dollar", "price", "afford", "change to", "make it"]
    )

    if is_budget_change:
        # Full budget redistribution prompt — ensures categories add up to the new total
        prompt = f"""You are a US budget travel planner for college students. The user wants to change their budget.

Current itinerary (keep the same destinations, activities, and restaurants unless the new budget forces upgrades/downgrades):
{state.get("itinerary", "")}

NEW BUDGET: ${total_budget} total / ${per_person}/{_traveler_word}
Trip: {state.get("destination")} | {num_travelers} {_traveler_word} | {num_days} days | {state.get("transport")} | {state.get("accommodation_pref", "")}
User request: {latest}

STRICT RULES:
1. ALL budget breakdown categories MUST sum exactly to ${total_budget} total
2. Dollar sign only before numbers — NEVER write $total or $person literally
3. Keep the same day-by-day structure; upgrade or downgrade individual items to fit the new budget
4. Recalculate every cost — do NOT reuse old dollar amounts if the budget changed significantly
5. Format the budget table with correct per-person and group-total columns

Rewrite the full itinerary with the updated budget. Keep the same markdown structure."""
    else:
        prompt = (
            f"Update this itinerary — ONLY change what's requested. US-only. Keep transport costs accurate.\n"
            f"ALL budget breakdown categories must still sum to the total budget.\n"
            f"Itinerary: {state.get('itinerary', '')}\n"
            f"Request: {latest} | Budget: ${total_budget} | {_traveler_word}: {num_travelers}"
        )

    r = call_llm(llm, [SystemMessage(content=prompt)])
    revised_itinerary = extract_text(r)

    # Re-run guardrail using the correct current budget (already computed above)
    try:
        revised_guardrail = evaluate_budget_guardrail(
            budget        = budget,
            num_days      = num_days,
            num_travelers = num_travelers,
            transport     = state.get("transport", "flying"),
            destination   = state.get("destination", ""),
            itinerary     = revised_itinerary,
            budget_type   = budget_type,
        )
    except Exception:
        revised_guardrail = state.get("budget_guardrail")

    return {
        "stage":            "done",
        "itinerary":        revised_itinerary,
        "budget_guardrail": revised_guardrail,
        "budget_exceeded":  revised_guardrail.get("status") in ("too_low", "over_budget") if revised_guardrail else False,
        "messages":         [{"role": "assistant", "content": revised_itinerary}],
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════════════════════
def route_after_parse(state: TravelState) -> str:
    if state.get("_international_attempt"): return "clarify"
    if state.get("stage") == "done":        return "revise"
    return "clarify"

def route_after_clarify(state: TravelState) -> str:
    return "generate" if state.get("stage") == "generating" else END

def build_graph():
    graph = StateGraph(TravelState)
    graph.add_node("parse_input", parse_input_node)
    graph.add_node("clarify",     clarify_node)
    graph.add_node("generate",    generate_node)
    graph.add_node("revise",      revise_node)
    graph.set_entry_point("parse_input")
    graph.add_conditional_edges("parse_input", route_after_parse, {"clarify": "clarify", "revise": "revise"})
    graph.add_conditional_edges("clarify", route_after_clarify, {"generate": "generate", END: END})
    graph.add_edge("generate", END)
    graph.add_edge("revise",   END)
    return graph.compile(checkpointer=MemorySaver())

def get_initial_state() -> TravelState:
    return TravelState(
        messages=[], destination=None, origin=None, num_days=None,
        budget=None, vibe=None, transport=None, vehicle=None, num_cars=None,
        num_travelers=None, budget_type=None, cuisine_prefs=None,
        accommodation_pref=None, transport_cost=None, stage="clarifying",
        itinerary=None, budget_breakdown=None, budget_exceeded=False,
        budget_guardrail=None, _international_attempt=False, _capacity_exceeded=None, _capacity_warned=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GROCERY & SNACKS AGENT
# ══════════════════════════════════════════════════════════════════════════════
def estimate_grocery_price(item: str) -> dict:
    """Estimate grocery item price via LLM. Prices are approximate."""
    try:
        r = llm.invoke([SystemMessage(content=
            f"Approximate Walmart price for '{item}'? Return ONLY a number like 3.47")])
        price = float(re.sub(r"[^\d.]", "", extract_text(r).strip()))
        return {"item": item, "price": price, "source": "estimated"}
    except Exception:
        return {"item": item, "price": 3.00, "source": "estimated"}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def generate_grocery_list(
    destination: str,
    num_days: int,
    num_travelers: int,
    transport: str,
    vibe: str,
    accommodation: str,
    budget_remaining: float,
    travelers_info: list = None,   # [{"name": "Alex", "diet": "vegetarian"}, ...]
    budget_mode: str = "group",    # "per_person" or "group"
) -> dict:
    is_road_trip = transport == "driving"
    has_kitchen  = accommodation in ("airbnb", "camping", "hostel")

    # Build traveler context string
    if travelers_info:
        traveler_lines = "\n".join(
            f"  - {t['name']}: {t['diet'] or 'no restrictions'}"
            for t in travelers_info
        )
        diet_section = f"Traveler dietary needs:\n{traveler_lines}"
    else:
        diet_section = f"{num_travelers} travelers, no dietary info provided"

    budget_note = (
        f"Budget: ~${budget_remaining / num_travelers:.0f}/person (${budget_remaining:.0f} total)"
        if budget_mode == "per_person"
        else f"Budget: ~${budget_remaining:.0f} for the whole group"
    )

    prompt = f"""Create a practical, personalized grocery and snacks list for this college student trip:
- Destination: {destination}
- Duration: {num_days} days, {num_travelers} people
- Transport: {transport} ({'road trip — include car snacks' if is_road_trip else 'flying — non-liquid snacks only'})
- Accommodation: {accommodation} ({'has kitchen — include meal prep items' if has_kitchen else 'no kitchen — packaged/ready-to-eat only'})
- Vibe: {vibe}
- {budget_note}
- {diet_section}

IMPORTANT: Respect every dietary restriction. If someone is vegan, vegetarian, gluten-free, halal, kosher, nut-free, etc., ensure ALL items are safe for them or note who each item is for.

Create 3 categories:
1. {'ROAD TRIP SNACKS' if is_road_trip else 'CARRY-ON SNACKS'}
2. {'BREAKFAST & QUICK MEALS' if has_kitchen else 'GRAB & GO BREAKFAST'}
3. DRINKS & HYDRATION

For each item: quantity for the group x {num_days} days, specific product name, and note if it's for specific travelers only.

Return ONLY JSON:
{{
  "categories": [
    {{"name": "category", "items": [{{"name": "product", "quantity": "2 packs", "for": "everyone or specific names"}}]}}
  ],
  "prep_tips": ["tip 1", "tip 2"]
}}"""

    r = llm.invoke([SystemMessage(content=prompt)])
    raw = extract_text(r).strip().replace("```json", "").replace("```", "")
    grocery_data = json.loads(raw)

    total_cost = 0.0
    for category in grocery_data.get("categories", []):
        priced_items = []
        for item in category.get("items", []):
            price_data  = estimate_grocery_price(item["name"])
            total_cost += price_data["price"]
            priced_items.append({
                "name":     item["name"],
                "quantity": item.get("quantity", "1"),
                "for":      item.get("for", "everyone"),
                "price":    price_data["price"],
                "source":   price_data["source"],
            })
        category["items"] = priced_items

    grocery_data["total_cost"]      = round(total_cost, 2)
    grocery_data["cost_per_person"] = round(total_cost / num_travelers, 2)
    grocery_data["budget_mode"]     = budget_mode
    return grocery_data


def generate_travel_checklist(
    destination: str,
    num_days: int,
    transport: str,
    vibe: str,
    accommodation: str,
    num_travelers: int,
) -> dict:
    _traveler_word = "person" if num_travelers == 1 else "people"
    prompt = f"""Create a practical travel checklist for this college student trip:
- Destination: {destination}, {num_days} days, {num_travelers} {_traveler_word}
- Transport: {transport}
- Vibe: {vibe}
- Accommodation: {accommodation}

Sections:
1. DOCUMENTS & ESSENTIALS
2. CLOTHING (based on {destination} climate + activities)
3. TECH & ENTERTAINMENT (for {transport} travel)
4. HEALTH & SAFETY
5. {'CAMPING GEAR' if accommodation == 'camping' else 'ACCOMMODATION ITEMS'}
6. MONEY-SAVING ITEMS (reusable bottle, snack containers, etc.)

CRITICAL: Every item MUST have a concrete short label in "item" (e.g. "Passport", "Phone charger", "Sunscreen").
"item" is the thing to pack. "note" is WHY to pack it. They must be different.

Return ONLY JSON:
{{
  "sections": [
    {{
      "name": "SECTION NAME",
      "items": [
        {{"item": "Passport", "priority": "essential", "note": "Required for international travel."}},
        {{"item": "Phone charger", "priority": "essential", "note": "Keep devices powered throughout the day."}}
      ]
    }}
  ],
  "trip_specific_tips": ["tip 1", "tip 2", "tip 3"]
}}

Do NOT leave "item" blank or use the note text as the item name."""

    r = llm.invoke([SystemMessage(content=prompt)])
    raw = extract_text(r).strip().replace("```json", "").replace("```", "")
    return json.loads(raw)