# TripBuddy

An AI-powered US budget travel planner for college students, built as an Agentic AI capstone project. TripBuddy takes a conversational approach to trip planning, asking the right questions and generating personalized itineraries with real cost breakdowns.

---

## Overview

TripBuddy is a multi-agent system built with LangGraph, GPT-4o-mini, and Streamlit. It guides users through trip planning via natural conversation, enforces realistic budget constraints, handles group logistics like car capacity, and generates day-by-day itineraries with grocery lists and packing checklists.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Agent framework | LangGraph |
| LLM | GPT-4o-mini (OpenAI) |
| Frontend | Streamlit |
| State persistence | LangGraph MemorySaver |
| Observability | LangSmith |
| LLM Judge | GPT-4o-mini (secondary evaluation call) |
| Vehicle data | NHTSA vPIC API (free, no key needed) |
| Gas prices | US EIA weekly average |

---

## Architecture

TripBuddy uses a four-node LangGraph pipeline. Each node reads from a shared `TravelState` TypedDict and writes back only the fields it changes. MemorySaver checkpoints the full state after every turn so the conversation can be resumed across page reloads.

```
User message
      |
      v
[input guardrails: PII masking + safety filter]
      |
      v
parse_input_node
      |
      v
clarify_node  <--(loops until all fields collected)
      |
      v
generate_node --> [LLM judge: scores + PASS/FAIL]
      |
      v
revise_node   <--(handles post-generation edits)
```

### `parse_input_node`

Runs on every user message. Its job is to extract structured trip fields from free-form text and write them into state.

- Runs PII masking and safety filtering on the raw user message before any other logic; blocks the message entirely if the safety filter fires, or redacts sensitive data and continues if PII is detected.
- Sends the latest user message plus the current state (as JSON) to the LLM so it knows what is already collected and only extracts what is missing.
- Applies normalization before storing: "whole group" and "together" become `"total"`, "per head" and "pp" become `"per_person"`, "all" for cuisine becomes `"all cuisines"`.
- Handles abbreviated vehicle input ("toy cam", "ford 150", "honda crv") by running a local lookup table first and falling back to the LLM only for unknown vehicles.
- Guards `num_travelers` from being overwritten while the capacity gate is active, preventing the group-size echo loop.
- Verifies that a parsed international destination actually appears in the user's message to catch LLM hallucinations (e.g. the LLM returning "Mexico" when the user said "India").

Fields extracted: `destination`, `origin`, `transport`, `vehicle`, `num_travelers`, `num_days`, `budget`, `budget_type`, `vibe`, `cuisine_prefs`, `accommodation_pref`, `num_cars`, `is_international`.

### `clarify_node`

Runs after every parse. Its job is to decide what to do next: ask a follow-up question, show a warning, or advance to generation.

**Capacity gate (runs first, before anything else)**

If the user is driving and the group size exceeds the vehicle's standard seating capacity, the gate fires before any other logic. On the first trigger it shows a warning with three options (take more cars, rent a larger vehicle, or reduce the group). On subsequent turns it parses the user's resolution response and either updates `num_cars`, `num_travelers`, or re-asks if the answer is unclear. The gate uses a `_capacity_warned` boolean in state rather than scanning message text, which prevents false positives from the LLM's own phrasing.

**Unified intent router**

Once basic fields (destination and transport) are known, a single LLM call classifies the user's intent and extracts any new fields in one step. The four possible intents are:
- `answer` -- user is replying to the current question; continue normal flow
- `new_us` -- user wants to plan a different US city; reset state
- `intl_learn` -- user is curious about a non-US place; answer briefly and stay on the US trip
- `vibe_shift` -- user is changing their travel style; update vibe and continue

This replaces four separate classifier calls that were chained sequentially.

**Missing-field collection**

After the capacity gate and intent router, the node builds a list of fields still needed (in a fixed order: origin, transport, vehicle, num_travelers, num_days, budget, budget_type, vibe, cuisine, accommodation). It asks for them one at a time. Standard questions are returned directly as strings with no LLM call. Vibe and cuisine questions use the LLM to generate city-specific options.

**International redirect**

When a user names a non-US destination, clarify_node runs a two-step flow: first it shows a menu of vibe options specific to that country, then (after the user picks one) it suggests US cities with a similar experience.

**Early budget check**

Once all fields are collected, the node runs `evaluate_budget_guardrail` before advancing to generation. If the budget is flagged as unrealistic, a warning appears in the chat. The itinerary still generates so the user can see the plan and decide whether to adjust.

### `generate_node`

Runs once all required fields are collected. Its job is to produce the full itinerary.

- Calls `sanitize_state` at the start to replace any None values with safe defaults (e.g. `num_travelers or 1`) before doing any arithmetic, preventing null-multiplication crashes.
- For driving trips, incorporates real transport costs: gas calculated from actual miles and the current EIA national average fuel price, plus estimated tolls and parking.
- Builds a budget breakdown table where the LLM fills in per-category amounts (transport, accommodation, food, activities) that must sum to the total budget.
- Runs `evaluate_budget_guardrail` again with the completed itinerary for a more accurate verdict than the early check.
- Returns the itinerary as a markdown string plus the budget guardrail result.
- Passes the completed itinerary to a secondary LLM judge that scores it across four dimensions (context precision, tool correction, completeness, hallucination) and returns a PASS or FAIL verdict with a score out of 10. A failing score triggers a retry up to two attempts before the result is returned.

### `revise_node`

Runs when the user sends a message after an itinerary has been generated. Its job is to decide whether the user wants a new trip or a modification to the current one.

- If the user is starting a new trip, it resets state and re-routes back through clarify.
- If the user is modifying the current trip (e.g. "make it cheaper", "swap the hotel to an Airbnb", "I want to drive instead"), it rewrites the itinerary with the requested change while preserving everything else.
- Detects budget changes and uses a targeted redistribution prompt so the new itinerary respects the updated total.
- Detects a transport switch from flying to driving and asks for the car before regenerating so gas costs can be included.


---

## RAG Module

TripBuddy's itineraries are grounded by a Retrieval-Augmented Generation layer that provides the `generate_node` with curated, destination-specific travel knowledge before the itinerary prompt is sent to the LLM.

### What it does

Without RAG, the LLM relies entirely on training data to name attractions, estimate prices, and describe neighborhoods — information that is often outdated or hallucinated. The RAG module solves this by pre-indexing content from multiple travel sources into a Chroma vector database and retrieving the most relevant passages at generation time.

### Data sources

Content is scraped and indexed from six enabled sources, organized by category:

| Source | Category | Scope |
|---|---|---|
| Wikivoyage | Activities & attractions | City |
| VisitTheUSA | Activities & attractions | City |
| Recreation.gov | Campsite prices | City |
| The Dyrt | Campground prices | State |
| Expatistan | Daily / food / lodging budgets | City |
| GasBuddy | Weekly gas prices | State |

### Pipeline

```
SOURCE_REGISTRY (rag_config.py)
        |
        v
Ingestors x6 (ingestors.py)  -- scrape + parse each source
        |
        v
TravelDocument (structures.py)  -- structured data model
        |
        v
TravelTextCleaner (indexing.py)  -- noise removal
        |
        v
SectionSplitter (splitters.py)  -- heading-based semantic sections
        |
        v
TravelSection (structures.py)
        |
        v
RecursiveCharacterTextSplitter  -- chunk_size=800, overlap=120
        |
        v
OpenAI text-embedding-3-small
        |
        v
Chroma Vector DB (data/chroma_db/)
        |
        v
TravelRAGService (service.py)  -- similarity search + context formatting
        |
        v
generate_node
```

### How retrieval works

`TravelRAGService.retrieve_context()` runs a similarity search filtered by city and state, returning the top-k most relevant passages formatted as a single context string. The `generate_node` prepends this context to the itinerary prompt so the LLM has grounded facts to draw from.

Metadata filters available at query time: `city`, `state`, `category`, `source`.

### Setup

The RAG index must be available before running TripBuddy. Choose one option:

**Option A — Use the prebuilt DB (recommended)**

Extract `chroma_db.zip` into `data/`:

```
data/chroma_db/
    chroma.sqlite3
    <uuid folders>
```

**Option B — Rebuild from scratch**

```bash
python rag/build_rag_index.py
```

Requires a valid `OPENAI_API_KEY`. Scraping all cities takes several minutes.

---

## Features

- **Conversational trip planning** -- collects all required fields through natural back-and-forth conversation
- **Vehicle inference** -- resolves abbreviations and typos to full make/model names before doing any capacity or cost calculations
- **Car capacity guardrail** -- blocks generation until the group-size-versus-vehicle-capacity conflict is explicitly resolved
- **Budget guardrail** -- runs before and after generation; $100+/person/day is green, $60-100 is yellow, below $60 or transport-exceeds-budget is red
- **Input guardrails** -- PII masking redacts emails, phone numbers, SSNs, and credit card numbers from every message before any LLM call; safety filter blocks harmful content including jailbreak attempts, prompt injection, and developer mode bypass
- **LLM Judge** -- a secondary LLM scores every generated itinerary across four dimensions (context precision, tool correction, completeness, hallucination) and returns a PASS or FAIL verdict with a score out of 10; a failing score triggers an automatic retry up to two attempts before the result is surfaced to the user
- **Road trip cost calculator** -- live gas prices via EIA, mileage from routing, split across cars and travelers
- **International redirect** -- vibe-matched US city suggestions when a non-US destination is named
- **Grocery list generator** -- personalized to group size, dietary restrictions, and trip duration
- **Travel checklist** -- packing list tailored to trip type, accommodation, and number of days
- **Chat persistence** -- completed trips saved to sidebar; in-progress trips saved as drafts

---

## Project Structure

```
.
+-- app_new.py              # Streamlit frontend and session management
+-- agents/
|   +-- agent_new_2.py      # All LangGraph nodes, state, helpers
+-- rag/
|   +-- service.py          # RAG retrieval service
|   +-- build_rag_index.py  # Builds the Chroma vector index
+-- data/
|   +-- user.json           # User profile
|   +-- chats.json          # Saved chat history
|   +-- draft.json          # In-progress draft
+-- assets/
|   +-- icon.png
+-- README.md
```

---

## Setup

**Prerequisites:** Python 3.10+, OpenAI API key, LangSmith API key

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt

export OPENAI_API_KEY=your_key_here
export LANGCHAIN_API_KEY=your_langsmith_key_here
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_PROJECT=tripbuddy

PYTHONPATH=. streamlit run app_new.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o-mini |
| `LANGCHAIN_API_KEY` | Yes | LangSmith API key for tracing |
| `LANGCHAIN_TRACING_V2` | Yes | Set to `true` to enable tracing |
| `LANGCHAIN_ENDPOINT` | Yes | Set to `https://api.smith.langchain.com` |
| `LANGCHAIN_PROJECT` | Yes | Project name shown in LangSmith dashboard |

---

## Sample Flow

```
User:  I want to go to Nashville for 4 days with a $500 budget.
Bot:   [input guardrails run]
Bot:   Where are you traveling from?
User:  Dallas, TX
Bot:   Flying or driving?
User:  Driving -- toy cam
Bot:   Got it -- Toyota Camry. How many people?
User:  6
Bot:   Quick heads up -- a Toyota Camry seats 5 max, but you have 6.
       - Take 2+ cars  |  Rent a minivan  |  Fewer people
User:  just 4 of us
Bot:   Got it -- 4 people. What is your vibe?
       [collects remaining fields]
Bot:   [generates 4-day Nashville itinerary with budget breakdown and road trip costs]
Bot:   [LLM judge scores itinerary -- PASS 9/10]
```

---

## Known Limitations

- US destinations only (international inputs redirect to similar US cities)
- Road trip costs use national average gas prices, not local prices
- Restaurant and activity names are LLM-generated and should be verified before booking
- MemorySaver state is in-memory; restarting the server clears history not saved to `data/chats.json`

---

## Built For

Spring 2026 Agentic AI course -- Oklahoma State University
