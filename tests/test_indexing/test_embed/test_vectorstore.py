from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from indexing.embed.vectorstore import create_retriever


class KeywordEmbeddings(Embeddings):
    """Small deterministic embedding model for the local vector-store test."""

    _keywords = ("quantitative", "python", "leadership")

    def _embed(self, text: str) -> list[float]:
        text = text.lower()
        return [float(keyword in text) for keyword in self._keywords]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)



def test_create_retriever_indexes_documents_and_returns_relevant_chunks():
    documents = [
        Document(
            page_content="Built quantitative forecasting models in Python.",
            metadata={"achievement_type": "technical"},
        ),
        Document(
            page_content="Led a student society and organised team events.",
            metadata={"achievement_type": "leadership"},
        ),
    ]

    retriever = create_retriever(documents, KeywordEmbeddings(), k=1)

    results = retriever.invoke("quantitative Python modelling")

    assert len(results) == 1
    assert results[0].page_content == documents[0].page_content
    assert results[0].metadata["achievement_type"] == "technical"