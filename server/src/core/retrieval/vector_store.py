from pathlib import Path
from functools import lru_cache
from typing import List

from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import HTTPException, status

from src.config import get_settings
from src.core.llm import create_chat_model
from src.config.logging import get_logger

logger = get_logger(__name__)

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


def _check_namespace_exists(user_id: str) -> bool:
    """
    Check if a user's namespace exists and contains documents.

    Args:
        user_id: The user ID to check

    Returns:
        True if namespace exists and has documents, False otherwise
    """
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    # Get index stats for the specific namespace
    stats = index.describe_index_stats()

    # Check if namespace exists and has vectors
    if "namespaces" in stats and user_id in stats["namespaces"]:
        vector_count = stats["namespaces"][user_id].get("vector_count", 0)
        return vector_count > 0

    return False


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
    logger.info(
        "Retrieving documents for query",
        extra={
            "user": user_id,
            "action": "vector_retrieve_start",
            "k": k or settings.retrieval_k,
        },
    )

    retriever = get_retriever(user_id, k=k)
    documents = retriever.invoke(query)

    logger.info(
        f"Retrieved {len(documents)} documents",
        extra={
            "user": user_id,
            "action": "vector_retrieve_complete",
            "doc_count": len(documents),
        },
    )

    return documents


def index_documents(file_path: Path, user_id: str) -> int:
    """
    Load a document from disk, split it into chunks, and index it into the user's namespace.

    Args:
        file_path: Path to the PDF file to index
        user_id: The user ID whose namespace to index into

    Returns:
        Number of chunks indexed
    """
    logger.info(
        f"Starting document indexing: {file_path.name}",
        extra={
            "user": user_id,
            "action": "vector_index_start",
            "file_name": file_path.name,
        },
    )

    try:
        loader = PyPDFLoader(str(file_path), mode="page")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(docs)

        vector_store = _get_vector_store(user_id)
        vector_store.add_documents(texts)

        logger.info(
            f"Document indexed successfully: {file_path.name}",
            extra={
                "user": user_id,
                "action": "vector_index_complete",
                "file_name": file_path.name,
                "chunk_count": len(texts),
            },
        )

        return len(texts)
    except Exception as e:
        logger.error(
            f"Failed to index document: {str(e)}",
            extra={
                "user": user_id,
                "action": "vector_index_error",
                "file_name": file_path.name,
            },
            exc_info=True,
        )
        raise
