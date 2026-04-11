from __future__ import annotations

from jsonschema.benchmarks.contains import end

"""
Offline index builder for the travel RAG.

What it does:
1. Defines a curated source registry
2. Fetches content from Wikipedia, NPS, and Visit The USA
3. Converts raw pages into TravelDocument objects
4. Cleans, sections, chunks, and embeds the content
5. Stores the chunks in a Chroma collection

Run:
    python build_rag_index.py

Requirements:
- Your package layout must expose:
    rag.ingestors
    rag.indexing
    rag.splitters
- OPENAI_API_KEY must be set in your environment
"""

from colorama import Fore, init
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestors import WikipediaIngestor, NPSIngestor, VisitUSAIngestor
from rag.indexing import TravelIndexer, TravelTextCleaner, LangChainDocumentConverter
from rag.splitters import SectionSplitter
from rag.structures import TravelDocument
from config import COLLECTION_NAME, EMBEDDING_MODEL, SOURCE_REGISTRY, CHUNK_SIZE, CHUNK_OVERLAP, PERSIST_DIRECTORY, OPENAI_API_KEY

init(autoreset=True)

# -------------------------------------------------------------------
# Collection helpers
# -------------------------------------------------------------------
def create_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

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
# Document collection either from the source registry or attempting to get it for an individual location
# -------------------------------------------------------------------
def collect_documents(source_registry: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[TravelDocument]:
    wiki = WikipediaIngestor()
    nps = NPSIngestor()
    visit = VisitUSAIngestor()

    docs: List[TravelDocument] = []

    for _, sources in source_registry.items():
        # Wikipedia
        for item in sources.get("wikipedia", []):
            try:
                doc = wiki.to_document(
                    title=item["title"],
                    destination=item["destination"],
                    state=item["state"],
                    category=item["category"],
                )
                docs.append(doc)
                print(f"[OK] Wikipedia: {item['title']}")
            except Exception as e:
                print(f"[ERROR] Wikipedia failed for {item.get('title', 'unknown')}: {e}")

        # NPS
        for item in sources.get("nps", []):
            try:
                doc = nps.to_document(
                    url=item["url"],
                    title=item["title"],
                    destination=item["destination"],
                    state=item["state"],
                    category=item["category"],
                )
                docs.append(doc)
                print(f"[OK] NPS: {item['title']}")
            except Exception as e:
                print(f"[ERROR] NPS failed for {item.get('title', 'unknown')}: {e}")

        # Visit The USA
        visit_pages = sources.get("visitusa", [])
        if visit_pages:
            try:
                visit_docs = visit.ingest_many(visit_pages, skip_errors=True)
                docs.extend(visit_docs)
                print(f"[OK] Visit The USA: indexed {len(visit_docs)} pages")
            except Exception as e:
                print(f"[ERROR] Visit The USA batch failed: {e}")

    return docs
# def collect_documents(source_name:str,location_name:str,state_name:str,category_name:str) -> List[TravelDocument]:
#     wiki = WikipediaIngestor()
#     nps = NPSIngestor()
#     visit = VisitUSAIngestor()
#
#     # This is the list to return
#     docs: List[TravelDocument] = []
#
#     match source_name:
#         case "wiki":
#             try:
#                 doc = wiki.to_document(
#                     title=location_name,
#                     destination=location_name,
#                     state=state_name,
#                     category=category_name,
#                 )
#                 docs.append(doc)
#                 print(f"[OK] Wikipedia: {location_name},{state_name},{category_name}")
#             except Exception as e:
#                 print(f"[ERROR] Wikipedia failed for  {location_name},{state_name},{category_name}: {e}")
#         case "nps":
#             try:
#                 doc = nps.to_document(
#                     url=item["url"],
#                     title=item["title"],
#                     destination=item["destination"],
#                     state=item["state"],
#                     category=item["category"],
#                 )
#                 docs.append(doc)
#                 print(f"[OK] NPS: {item['title']}")
#             except Exception as e:
#                 print(f"[ERROR] NPS failed for {item.get('title', 'unknown')}: {e}")
#         case "visitusa":
#             try:
#                 visit_docs = visit.ingest_many(visit_pages, skip_errors=True)
#                 docs.extend(visit_docs)
#                 print(f"[OK] Visit The USA: indexed {len(visit_docs)} pages")
#             except Exception as e:
#                 print(f"[ERROR] Visit The USA batch failed: {e}")
#
#     return docs

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("Starting offline RAG index build...")

    vector_store = create_vector_store()
    reset_collection(vector_store)

    # Recreate store after deletion to ensure a fresh handle
    vector_store = create_vector_store()

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    cleaner = TravelTextCleaner()
    section_splitter = SectionSplitter()
    converter = LangChainDocumentConverter(splitter=chunk_splitter)

    indexer = TravelIndexer(
        wikipedia_ingestor=WikipediaIngestor(),
        nps_ingestor=NPSIngestor(),
        visitusa_ingestor=VisitUSAIngestor(),
        cleaner=cleaner,
        section_splitter=section_splitter,
        converter=converter,
        vector_store=vector_store,
    )

    docs = collect_documents(SOURCE_REGISTRY)
    print(f"Collected {len(docs)} raw documents.")

    if not docs:
        print("No documents were collected. Nothing to index.")
        return

    chunk_count = indexer.build_index(docs)
    print(f"Indexed {chunk_count} chunks into Chroma collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()