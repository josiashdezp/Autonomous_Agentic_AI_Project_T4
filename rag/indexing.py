from rag.structures import TravelSection, TravelDocument
from rag.ingestors import WikipediaIngestor, NPSIngestor, VisitUSAIngestor
from rag.splitters import SectionSplitter
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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

# Clean the NPS pages from contact blocks, app promos, and other non-core content around the useful visit-planning text.
class TravelTextCleaner:
    def clean(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

# This is what turns the ingestors into an actual RAG corpus.
class TravelIndexer:
    def __init__(
        self,
        wikipedia_ingestor: WikipediaIngestor,
        nps_ingestor: NPSIngestor,
        visitusa_ingestor: VisitUSAIngestor,
        cleaner: TravelTextCleaner,
        section_splitter: SectionSplitter,
        converter: LangChainDocumentConverter,
        vector_store: Chroma,
    ):
        self.wikipedia_ingestor = wikipedia_ingestor
        self.nps_ingestor = nps_ingestor
        self.visitusa_ingestor = visitusa_ingestor
        self.cleaner = cleaner
        self.section_splitter = section_splitter
        self.converter = converter
        self.vector_store = vector_store

    def build_index(self, docs: list[TravelDocument]) -> int:
        all_sections: list[TravelSection] = []

        for doc in docs:
            doc.content = self.cleaner.clean(doc.content)
            sections = self.section_splitter.split_document(doc)
            all_sections.extend(sections)

        chunked_docs = self.converter.from_sections(all_sections)
        if chunked_docs:
            self.vector_store.add_documents(chunked_docs)

        return len(chunked_docs)
