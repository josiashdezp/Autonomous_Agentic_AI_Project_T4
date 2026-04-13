from dotenv import load_dotenv
import os
from colorama import Fore

# -------------------------------------------------------------------
# Curated source registry
# -------------------------------------------------------------------
SOURCE_REGISTRY = {
    "categories": {
        "activities_attractions": {
            "description": "Sources for things to do, attractions, landmarks, and activity ideas.",
            "sources": [
                {
                    "source_name": "wikivoyage",
                    "source_type": "rag_text",
                    "enabled": True,
                    "level": "city",
                    "url":
                        {
                            "url_template": "https://en.wikivoyage.org/wiki/{city_slug}",
                            "separator": "_",
                            "casesensitive": True,
                            "scrape-level": 1
                        },
                    "data_focus": ["see", "do", "sleep", "eat"],
                    "parser": "WikivoyageIngestor",
                    "output_type": "travel_document"
                },
                {
                    "source_name": "visittheusa",
                    "source_type": "rag_text",
                    "enabled": True,
                    "level": "city",
                    "url": {
                        "url_template": "https://www.visittheusa.com/destinations/{state}/{city_slug}/",
                        "separator": "-",
                        "casesensitive": False,
                        "scrape-level": 0
                    },
                    "data_focus": ["overview", "landmarks"],
                    "parser": "VisitTheUSAIngestor",
                    "output_type": "travel_document"
                },
                {
                    "source_name": "timeout",
                    "source_type": "rag_text",
                    "enabled": False,
                    "level": "city",
                    "url":
                        {
                            "url_template": "https://www.timeout.com/{city_slug}",
                            "separator": "-",
                            "casesensitive": False,
                            "scrape-level": 1
                        },
                    "data_focus": ["free_activities"],
                    "parser": "GenericArticleIngestor",
                    "output_type": "travel_document"
                },
                {
                    "source_name": "recreation_gov",
                    "source_type": "rag_text",
                    "enabled": True,
                    "level": "city",
                    "url":
                        {
                            "url_template": "https://www.recreation.gov/search?q={city_slug}",
                            "separator": "%20",
                            "casesensitive": False,
                            "scrape-level": 0
                        },
                    "data_focus": ["campsite_prices"],
                    "parser": "GenericArticleIngestor",
                    "output_type": "travel_document"
                },
                {
                    "source_name": "alltrails",
                    "source_type": "rag_text",
                    "enabled": False,  # Is throwing errors
                    "level": "state",
                    "url":
                        {
                            "url_template": "https://www.alltrails.com/us/{state_slug}",
                            "separator": "-",
                            "casesensitive": False,
                            "scrape-level": 0
                        },
                    "data_focus": ["trail_info", "fees"],
                    "parser": "GenericArticleIngestor",
                    "output_type": "travel_document"
                },
                {
                    "source_name": "the_dyrt",
                    "source_type": "rag_text",
                    "enabled": True,
                    "level": "state",
                    "url":
                        {
                            "url_template": "https://thedyrt.com/camping/{state_slug}",
                            "separator": "-",
                            "casesensitive": False,
                            "scrape-level": 0
                        },
                    "data_focus": ["campground_prices"],
                    "parser": "GenericArticleIngestor",
                    "output_type": "travel_document"
                }]
        },
        "cost_of_living": {
            "description": "Sources for meal, housing, and transportation costs.",
            "sources": [
                {
                    "source_name": "numbeo",
                    "source_type": "structured_snapshot",
                    "enabled": False,
                    "level": "city",
                    "url":
                        {
                            "url_template": "https://www.numbeo.com/cost-of-living/in/{city_slug}",
                            "separator": "-",
                            "casesensitive": False,
                            "scrape-level": 0
                        },

                    "data_focus": ["prices", "cost_of_living_table"],
                    "parser": "NumbeoIngestor",
                    "output_type": "travel_document"
                },
                {
                    "source_name": "expatistan",
                    "source_type": "structured_snapshot",
                    "enabled": True,
                    "level": "city",
                    "url":
                        {
                            "url_template": "https://www.expatistan.com/cost-of-living/{city_slug}",
                            "separator": "-",
                            "casesensitive": False,
                            "scrape-level": 0
                        },
                    "data_focus": ["daily_budget", "food_budget", "lodging_budget"],
                    "parser": "ExpatistanIngestor",
                    "output_type": "travel_document"
                }
            ]
        },
        "transport": {
            "description": "Sources for routing, gas costs, and mobility information.",
            "sources": [
                {
                    "source_name": "gasbuddy",
                    "source_type": "structured_snapshot",
                    "enabled": True,
                    "level": "state",
                    "url":
                        {
                            "url_template": "https://www.gasbuddy.com/gasprices/{state_slug}",
                            "separator": "-",
                            "casesensitive": False,
                            "scrape-level": 0
                        },
                    "data_focus": ["weekly_gas_prices"],
                    "parser": "GasBuddyIngestor",
                    "output_type": "travel_document"
                }
            ]
        }
    }
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# -------------------------------------------------------------------
# Vector store config
# -------------------------------------------------------------------
COLLECTION_NAME = "us_travel_rag"
PERSIST_DIRECTORY = "./chroma_travel_db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# --- Function to read the key to access OpenAI and creation of the LLM ---
def get_openai_key():
    try:
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        if key:
            # Prints only the first 8 characters for safety
            print(Fore.RED + f"DEBUG: Key found starting with: {key[:8]}...")
            return key
        else:
            print(Fore.RED + "DEBUG: No API Key found!")
            return None
    except Exception as e:
        print(e)
        return None
