from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag.rag_config import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIRECTORY, get_openai_key


class TravelRAGService:
    """
    Retrieval layer for the travel RAG.

    Responsibilities:
    - connect to the persisted Chroma vector store
    - run similarity search
    - optionally filter by destination, state, category, or source
    - return normalized retrieval results for downstream agents
    """

    def __init__(self, vector_store: Chroma) -> None:
        self.vector_store = vector_store

    @classmethod
    def from_persisted_db(
        cls,
        persist_directory: str = PERSIST_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> "TravelRAGService":
        """
        Convenience constructor for loading the persisted Chroma DB.
        """
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=get_openai_key(),
        )

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        return cls(vector_store=vector_store)

    def _build_filter(
        self,
        destination: Optional[str] = None,
        state: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a Chroma metadata filter.
        """
        clauses: List[Dict[str, Any]] = []

        if destination:
            clauses.append({"destination": destination})
        if state:
            clauses.append({"state": state})
        if category:
            clauses.append({"category": category})
        if source:
            clauses.append({"source": source})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

    def search(
        self,
        query: str,
        destination: Optional[str] = None,
        state: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Similarity search over the vector store.

        Returns a normalized list of dicts so the result is easy to use in
        LangGraph nodes, tools, or direct debugging.
        """
        metadata_filter = self._build_filter(
            destination=destination,
            state=state,
            category=category,
            source=source,
        )

        docs = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter,
        )

        results: List[Dict[str, Any]] = []
        for rank, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}

            results.append(
                {
                    "rank": rank,
                    "content": doc.page_content,
                    "metadata": metadata,
                    "source": metadata.get("source"),
                    "destination": metadata.get("destination"),
                    "state": metadata.get("state"),
                    "category": metadata.get("category"),
                    "heading": metadata.get("heading"),
                    "section_id": metadata.get("section_id"),
                    "parent_doc_id": metadata.get("parent_doc_id"),
                }
            )

        return results

    def search_with_scores(
        self,
        query: str,
        destination: Optional[str] = None,
        state: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Similarity search that also returns distance scores.
        Lower scores are generally better in Chroma similarity_distance output.
        """
        metadata_filter = self._build_filter(
            destination=destination,
            state=state,
            category=category,
            source=source,
        )

        docs_and_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=metadata_filter,
        )

        results: List[Dict[str, Any]] = []
        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            metadata = doc.metadata or {}

            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "content": doc.page_content,
                    "metadata": metadata,
                    "source": metadata.get("source"),
                    "destination": metadata.get("destination"),
                    "state": metadata.get("state"),
                    "category": metadata.get("category"),
                    "heading": metadata.get("heading"),
                    "section_id": metadata.get("section_id"),
                    "parent_doc_id": metadata.get("parent_doc_id"),
                }
            )

        return results

    def format_context(
        self,
        results: List[Dict[str, Any]],
        max_chars: int = 6000,
    ) -> str:
        """
        Convert retrieval results into a single context string for the LLM.
        """
        if not results:
            return "No relevant travel context found."

        parts: List[str] = []
        total_chars = 0

        for item in results:
            header_bits = []

            if item.get("source"):
                header_bits.append(f"Source: {item['source']}")
            if item.get("destination"):
                header_bits.append(f"Destination: {item['destination']}")
            if item.get("state"):
                header_bits.append(f"State: {item['state']}")
            if item.get("category"):
                header_bits.append(f"Category: {item['category']}")
            if item.get("heading"):
                header_bits.append(f"Heading: {item['heading']}")

            header = " | ".join(header_bits)
            block = f"{header}\n{item['content']}".strip()

            if total_chars + len(block) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 0:
                    parts.append(block[:remaining])
                break

            parts.append(block)
            total_chars += len(block) + 2

        return "\n\n".join(parts)

    def retrieve_context(
        self,
        query: str,
        destination: Optional[str] = None,
        state: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        k: int = 5,
        max_chars: int = 6000,
    ) -> str:
        """
        Convenience method:
        search -> format results into a single LLM-ready context string.
        """
        results = self.search(
            query=query,
            destination=destination,
            state=state,
            category=category,
            source=source,
            k=k,
        )
        return self.format_context(results, max_chars=max_chars)