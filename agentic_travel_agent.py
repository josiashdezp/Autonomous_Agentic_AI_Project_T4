#####################################################################
# CONFIGURATIONS
######################################################################

from rag.service import TravelRAGService
from rag.indexing import TravelIndexer, TravelTextCleaner, LangChainDocumentConverter
from rag.splitters import SectionSplitter
from rag.ingestors import VisitUSAIngestor, WikipediaIngestor,NPSIngestor
from rag.service import TravelRAGService
from rag.build_rag_index import create_vector_store
from structured.opentravel_service import OpenTravelDataService
from structured.opentravel_loader import OpenTravelDataLoader

from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


vector_store = create_vector_store()
rag_service = TravelRAGService(vector_store=vector_store, k=4)
service = OpenTravelDataService(airports_df=airports_df, airlines_df=airlines_df)


@tool
def search_travel_knowledge(query: str) -> str:
    """Search trusted U.S. travel knowledge for attractions, destination tips, and visitor guidance."""
    return rag_service.search(query)





@tool
def lookup_airports_by_city(city: str) -> str:
    """Find airports serving a city using local OpenTravelData."""
    results = service.find_airports_by_city(city)
    return str(results[:5])











def build_default_rag_stack() -> TravelRAGService:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma(
        collection_name="us_travel_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_travel_db"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    converter = LangChainDocumentConverter(splitter=splitter)
    cleaner = TravelTextCleaner()
    section_splitter = SectionSplitter()

    indexer = TravelIndexer(
        wikipedia_ingestor=WikipediaIngestor(),
        nps_ingestor=NPSIngestor(),
        visitusa_ingestor=VisitUSAIngestor(),
        cleaner=cleaner,
        section_splitter=section_splitter,
        converter=converter,
        vector_store=vector_store,
    )

    return TravelRAGService(vector_store=vector_store, k=4)