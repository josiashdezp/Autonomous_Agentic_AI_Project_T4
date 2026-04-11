from typing import Any, Dict, List
from dotenv import load_dotenv
import os
from colorama import Fore

# -------------------------------------------------------------------
# Curated source registry
# -------------------------------------------------------------------
SOURCE_REGISTRY: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "new_york_city": {
        "wikipedia": [
            {
                "title": "New York City",
                "destination": "New York City",
                "state": "NY",
                "category": "city_overview",
            },
            {
                "title": "Tourism in New York City",
                "destination": "New York City",
                "state": "NY",
                "category": "tourism",
            },
            {
                "title": "Transportation in New York City",
                "destination": "New York City",
                "state": "NY",
                "category": "transportation",
            },
        ],
        "nps": [
            {
                "url": "https://www.nps.gov/npnh/planyourvisit/index.htm",
                "title": "National Parks of New York Harbor Plan Your Visit",
                "destination": "New York City",
                "state": "NY",
                "category": "landmarks",
            }
        ],
        "visitusa": [
            {
                "url": "https://www.visittheusa.com/destination/new-york-city",
                "destination": "New York City",
                "state": "NY",
                "category": "city_guide",
                "extra_metadata": {"tags": ["urban", "landmarks", "food"]},
            }
        ],
    },
    "washington_dc": {
        "wikipedia": [
            {
                "title": "Washington, D.C.",
                "destination": "Washington, D.C.",
                "state": "DC",
                "category": "city_overview",
            },
            {
                "title": "Tourism in Washington, D.C.",
                "destination": "Washington, D.C.",
                "state": "DC",
                "category": "tourism",
            },
        ],
        "nps": [
            {
                "url": "https://www.nps.gov/nama/planyourvisit/index.htm",
                "title": "National Mall and Memorial Parks Plan Your Visit",
                "destination": "Washington, D.C.",
                "state": "DC",
                "category": "landmarks",
            }
        ],
        "visitusa": [
            {
                "url": "https://www.visittheusa.com/destination/washington-dc",
                "destination": "Washington, D.C.",
                "state": "DC",
                "category": "city_guide",
                "extra_metadata": {"tags": ["museums", "monuments", "history"]},
            }
        ],
    },
}

# -------------------------------------------------------------------
# Vector store config
# -------------------------------------------------------------------
COLLECTION_NAME = "us_travel_rag"
PERSIST_DIRECTORY = "./chroma_travel_db"
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# --- Reading the key to access OpenAI and creation of the LLM ---
try:
    load_dotenv()
except Exception:
    pass
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    # Prints only the first 8 characters for safety
    print(Fore.RED + f"DEBUG: Key found starting with: {OPENAI_API_KEY[:8]}...")
else:
    print(Fore.RED + "DEBUG: No API Key found!")