from __future__ import annotations
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ingestors import WikivoyageIngestor, VisitTheUSAIngestor, TimeOutIngestor, RecreationGovIngestor, \
    AllTrailsIngestor, TheDyrtIngestor, NumbeoIngestor, ExpatistanIngestor, GasBuddyIngestor
from rag.structures import TravelSection, TravelDocument
from rag.splitters import SectionSplitter
import re


class LangChainDocumentConverter:
    def __init__(self, splitter: RecursiveCharacterTextSplitter):
        self.splitter = splitter

    def from_sections(self, sections: list[TravelSection]) -> list[Document]:
        docs: list[Document] = []

        for section in sections:
            chunk_docs = self.splitter.create_documents(
                texts=[section.content],
                metadatas=[{
                    **section.metadata,
                    "section_id": section.section_id,
                    "parent_doc_id": section.parent_doc_id,
                }]
            )
            docs.extend(chunk_docs)

        return docs


# Clean the pages from contact blocks, app promos, and other non-core content around the useful visit-planning text.
class TravelTextCleaner:
    def clean(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

# This is what turns the ingestors into an actual RAG corpus.
class TravelIndexer:
    def __init__(

            # Declaration of the fields of the class.
        self,
            wikivoyage: WikivoyageIngestor,
            visittheusa: VisitTheUSAIngestor,
            timeout: TimeOutIngestor,
            recreation_gov: RecreationGovIngestor,
            alltrails: AllTrailsIngestor,
            the_dyrt: TheDyrtIngestor,
            numbeo: NumbeoIngestor,
            expatistan: ExpatistanIngestor,
            gasbuddy: GasBuddyIngestor,

            # Tools for cleaning and splitting the text.
        cleaner: TravelTextCleaner,
        section_splitter: SectionSplitter,
        converter: LangChainDocumentConverter,
        vector_store: Chroma,
    ):
        # Initialize the ingestors (fields of the class) and tools for cleaning and splitting the text.
        self.wikivoyage = wikivoyage
        self.visittheusa = visittheusa
        self.timeout = timeout
        self.recreation_gov = recreation_gov
        self.alltrails = alltrails
        self.the_dyrt = the_dyrt
        self.numbeo = numbeo
        self.expatistan = expatistan
        self.gasbuddy = gasbuddy

        self.cleaner = cleaner
        self.section_splitter = section_splitter
        self.converter = converter
        self.vector_store = vector_store

    def build_index(self, docs: list[TravelDocument], batch_size: int = 50) -> int:
        all_sections: list[TravelSection] = []

        for doc in docs:
            doc.content = self.cleaner.clean(doc.content)
            sections = self.section_splitter.split_document(doc)
            all_sections.extend(sections)

        chunked_docs = self.converter.from_sections(all_sections)

        if not chunked_docs:
            return 0

        total = len(chunked_docs)
        inserted = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = chunked_docs[start:end]

            self.vector_store.add_documents(batch)
            inserted += len(batch)

            print(f"Inserted batch {start}-{end} of {total}")
            print("Current collection count:", self.vector_store._collection.count())

        return inserted