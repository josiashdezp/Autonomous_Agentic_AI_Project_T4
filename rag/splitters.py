from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.structures import TravelSection, TravelDocument
import re

##---------------------------------------------------------------------
# This is a semantic splitter.
# It takes one TravelDocument and tries to break it into meaningful sections based on headings or heading-like patterns in the raw text.
##---------------------------------------------------------------------
class SectionSplitter:
    def split_document(self, doc: TravelDocument) -> list[TravelSection]:
        raw_sections = re.split(r"\n(?=[A-Z][A-Za-z0-9 ,/&()-]{2,80}\n?)", doc.content)
        sections: list[TravelSection] = []

        for i, section_text in enumerate(raw_sections, start=1):
            section_text = section_text.strip()
            if not section_text:
                continue

            lines = section_text.splitlines()
            heading = lines[0][:120] if lines else f"section_{i}"
            content = "\n".join(lines[1:]).strip() if len(lines) > 1 else section_text

            sections.append(
                TravelSection(
                    section_id=f"{doc.doc_id}::section::{i}",
                    parent_doc_id=doc.doc_id,
                    heading=heading,
                    content=content,
                    metadata={
                        **doc.metadata,
                        "source": doc.source,
                        "destination": doc.destination,
                        "state": doc.state,
                        "category": doc.category,
                        "heading": heading,
                    },
                )
            )

        return sections

##---------------------------------------------------------------------
# This is the LangChain chunker.#
# It takes each already-separated section and breaks it into embedding-sized chunks
##---------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)