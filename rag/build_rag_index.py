from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from colorama import init
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.rag_config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    SOURCE_REGISTRY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PERSIST_DIRECTORY,
    get_openai_key
)

from rag.ingestors import (
    WikivoyageIngestor,
    VisitTheUSAIngestor,
    TimeOutIngestor,
    RecreationGovIngestor,
    AllTrailsIngestor,
    TheDyrtIngestor,
    NumbeoIngestor,
    ExpatistanIngestor,
    GasBuddyIngestor,
)
from rag.indexing import TravelIndexer, TravelTextCleaner, LangChainDocumentConverter
from rag.splitters import SectionSplitter
from rag.structures import TravelDocument

init(autoreset=True)

# -------------------------------------------------------------------
# Collection helpers
# -------------------------------------------------------------------
def create_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=get_openai_key(),
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
def reset_collection(vector_store: Chroma) -> None:
    """
    Best-effort collection reset so repeated runs do not duplicate content.
    """
    try:
        vector_store.delete_collection()
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        print(f"No existing collection to delete, or delete not supported: {COLLECTION_NAME}")

# -------------------------------------------------------------------
# URL helpers
# -------------------------------------------------------------------
def build_url_from_registry(source_meta: Dict[str, Any], city: Dict[str, Any], state: Dict[str, Any]) -> str:
    """
    Builds a URL from the nested source_registry format:

    "url": {
        "url_template": "...",
        "separator": "-",
        "casesensitive": False,
        "scrape-level": 0
    }
    """
    url_cfg = source_meta.get("url", {})
    template = url_cfg.get("url_template", "").strip()
    separator = url_cfg.get("separator", "-")
    case_sensitive = url_cfg.get("casesensitive", False)

    def normalize(value: str) -> str:
        text = value if case_sensitive else value.lower()
        return text.replace("-", " ").replace(" ", separator)

    city_slug = city.get("slug", "")
    state_slug = state.get("slug", "")

    # If slugs are already present in the locations file, keep them as primary.
    # Otherwise derive from names.
    if not city_slug:
        city_slug = normalize(city["name"])
    else:
        city_slug = city_slug if case_sensitive else city_slug.lower()
        if separator != "-":
            city_slug = city_slug.replace("-", separator)

    if not state_slug:
        state_slug = normalize(state["name"])
    else:
        state_slug = state_slug if case_sensitive else state_slug.lower()
        if separator != "-":
            state_slug = state_slug.replace("-", separator)

    return template.format(
        city=city["name"],
        city_slug=city_slug,
        state=state["name"] if case_sensitive else state["name"].lower(),
        state_slug=state_slug,
        state_abbr=state["abbr"] if case_sensitive else state["abbr"].lower(),
    )

# -------------------------------------------------------------------
# Document collection from SOURCE_REGISTRY
# -------------------------------------------------------------------

def collect_documents(
        source_registry: Dict[str, Any],
        locations: Dict[str, Any],
) -> List[TravelDocument]:
    ingestor_map = {
        "wikivoyage": WikivoyageIngestor(),
        "visittheusa": VisitTheUSAIngestor(),
        "timeout": TimeOutIngestor(),
        "recreation_gov": RecreationGovIngestor(),
        "alltrails": AllTrailsIngestor(),
        "the_dyrt": TheDyrtIngestor(),
        "numbeo": NumbeoIngestor(),
        "expatistan": ExpatistanIngestor(),
        "gasbuddy": GasBuddyIngestor(),
    }

    docs: List[TravelDocument] = []
    categories = source_registry.get("categories", {})

    for state in locations.get("states", []):
        for city in state.get("major_cities", []):
            for category_name, category_meta in categories.items():
                sources = category_meta.get("sources", [])

                for source_meta in sources:
                    if not source_meta.get("enabled", True):
                        continue

                    # Only sources intended for RAG documents
                    if source_meta.get("output_type") != "travel_document":
                        continue

                    source_name = source_meta.get("source_name")
                    level = source_meta.get("level", "city")

                    if source_name not in ingestor_map:
                        print(f"[SKIP] No ingestor registered for source '{source_name}'")
                        continue

                    # Respect source scope
                    if level not in {"city", "state"}:
                        print(f"[SKIP] Unsupported level '{level}' for source '{source_name}'")
                        continue

                    ingestor = ingestor_map[source_name]
                    url = build_url_from_registry(source_meta, city=city, state=state)

                    extra_metadata = {
                        "level": level,
                        "data_focus": source_meta.get("data_focus", []),
                        "source_name": source_name,
                        "scrape_level": source_meta.get("url", {}).get("scrape-level", 0),
                    }

                    try:
                        # State-level sources still produce docs associated with the current city
                        # because the current indexing flow iterates city-by-city.
                        doc = ingestor.to_document(
                            url=url,
                            destination=city["name"],
                            state=state["abbr"],
                            category=category_name,
                            extra_metadata=extra_metadata,
                        )

                        docs.append(doc)
                        print(f"[OK] {source_name}: {city['name']}, {state['abbr']}")

                    except Exception as e:
                        print(f"[ERROR] {source_name} failed for {city['name']}, {state['abbr']}: {e}")

    return docs


# -------------------------------------------------------------------
# Locations JSON loader
# -------------------------------------------------------------------
def load_locations(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "states" not in data:
        raise ValueError("Invalid locations JSON: missing 'states' key")

    return data

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("Starting offline RAG index build...")

    get_openai_key()
    vector_store = create_vector_store()
    reset_collection(vector_store)
    vector_store = create_vector_store()

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    wikivoyage = WikivoyageIngestor()
    visittheusa = VisitTheUSAIngestor()
    timeout = TimeOutIngestor()
    recreation_gov = RecreationGovIngestor()
    alltrails = AllTrailsIngestor()
    the_dyrt = TheDyrtIngestor()
    numbeo = NumbeoIngestor()
    expatistan = ExpatistanIngestor()
    gasbuddy = GasBuddyIngestor()
    cleaner = TravelTextCleaner()
    section_splitter = SectionSplitter()
    converter = LangChainDocumentConverter(splitter=chunk_splitter)

    # Adjust this path if your script lives somewhere else
    locations_path = Path("../data/usa_locations.json")
    location_data = load_locations(str(locations_path))

    indexer = TravelIndexer(
        wikivoyage=wikivoyage,
        visittheusa=visittheusa,
        timeout=timeout,
        recreation_gov=recreation_gov,
        alltrails=alltrails,
        the_dyrt=the_dyrt,
        numbeo=numbeo,
        expatistan=expatistan,
        gasbuddy=gasbuddy,
        cleaner=cleaner,
        section_splitter=section_splitter,
        converter=converter,
        vector_store=vector_store
    )

    docs = collect_documents(SOURCE_REGISTRY, location_data)
    print(f"Collected {len(docs)} raw documents.")

    if not docs:
        print("No documents were collected. Nothing to index.")
        return

    try:
        chunk_count = indexer.build_index(docs, batch_size=50)
        print(f"Indexed {chunk_count} chunks into Chroma collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Indexing failed: {e}")
        print(f"Current collection count: {vector_store._collection.count()}")
        raise


if __name__ == "__main__":
    main()