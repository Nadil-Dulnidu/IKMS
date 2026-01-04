from pathlib import Path
from functools import lru_cache
from typing import List

from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers import LLMImageBlobParser

from src.config import get_settings
from src.core.llm import create_chat_model

settings = get_settings()

@lru_cache(maxsize=1)
def _get_vector_store() -> PineconeVectorStore:
    """Create a PineconeVectorStore instance configured from settings."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model_name,
        api_key=settings.openai_api_key,
    )
    
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

def get_retriever(k: int | None = None):
    """Get a retriever instance from the vector store."""
    if k is None:
        k = settings.retrieval_k
    vector_store = _get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})

def retrieve(query: str, k: int | None = None) -> List[Document]:
    """Retrieve relevant documents for a given query."""
    retriever = get_retriever(k=k)
    return retriever.invoke(query)

def index_documents(file_path: Path) -> int:
    """Load a document from disk, split it into chunks, and index it into the vector DB."""
    loader = PyMuPDF4LLMLoader(
        str(file_path), 
        mode="single",
        # extract_images=True,
        # image_parser=LLMImageBlobParser(
        #         model=create_chat_model()
        #     ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    vector_store = _get_vector_store()
    vector_store.add_documents(texts)
    return len(texts)