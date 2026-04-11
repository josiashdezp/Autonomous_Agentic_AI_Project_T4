#---------------------------------------------------------------------
# This is the runtime layer the rest of the agent should call.
#---------------------------------------------------------------------
from langchain_chroma import Chroma

class TravelRAGService:
    def __init__(self, vector_store: Chroma, k: int = 4):
        self.vector_store = vector_store
        self.k = k

    def search(self, query: str, destination: str | None = None) -> str:
        if destination:
            docs = self.vector_store.similarity_search(
                query,
                k=self.k,
                filter={"destination": destination}
            )
        else:
            docs = self.vector_store.similarity_search(query, k=self.k)

        if not docs:
            return "No relevant travel information found."

        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}] "
            f"[Destination: {d.metadata.get('destination', 'unknown')}]\n"
            f"{d.page_content}"
            for d in docs
        )