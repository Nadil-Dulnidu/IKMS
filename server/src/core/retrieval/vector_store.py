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


def _get_vector_store(user_id: str) -> PineconeVectorStore:
    """
    Create a PineconeVectorStore instance configured from settings with user-specific namespace.

    Args:
        user_id: The user ID to use as the namespace for data isolation

    Returns:
        PineconeVectorStore instance configured for the specific user
    """
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model_name,
        api_key=settings.openai_api_key,
    )

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=user_id,  # User-specific namespace for data isolation
    )


def get_retriever(user_id: str, k: int | None = None):
    """
    Get a retriever instance from the vector store for a specific user.

    Args:
        user_id: The user ID to retrieve documents for
        k: Number of documents to retrieve (defaults to settings.retrieval_k)

    Returns:
        Retriever instance configured for the specific user
    """
    if k is None:
        k = settings.retrieval_k
    vector_store = _get_vector_store(user_id)
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, user_id: str, k: int | None = None) -> List[Document]:
    """
    Retrieve relevant documents for a given query from a specific user's namespace.

    Args:
        query: The search query
        user_id: The user ID whose documents to search
        k: Number of documents to retrieve

    Returns:
        List of relevant documents from the user's namespace
    """
    retriever = get_retriever(user_id, k=k)
    return retriever.invoke(query)


def index_documents(file_path: Path, user_id: str) -> int:
    """
    Load a document from disk, split it into chunks, and index it into the user's namespace.

    Args:
        file_path: Path to the PDF file to index
        user_id: The user ID whose namespace to index into

    Returns:
        Number of chunks indexed
    """
    loader = PyMuPDF4LLMLoader(
        str(file_path),
        mode="page",
        # extract_images=True,
        # image_parser=LLMImageBlobParser(
        #         model=create_chat_model()
        #     ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    vector_store = _get_vector_store(user_id)
    vector_store.add_documents(texts)
    return len(texts)
