"""Tools available to agents in the multi-agent RAG system."""

from langchain_core.tools import tool

from ..retrieval.vector_store import retrieve
from ..retrieval.serialization import serialize_chunks_with_ids


@tool(response_format="content_and_artifact")
def retrieval_tool(query: str):
    """Search the vector database for relevant document chunks with citation support.

    This tool retrieves the top 4 most relevant chunks from the Pinecone
    vector store based on the query. The chunks are formatted with stable
    chunk IDs [C1], [C2], etc. for citation tracking.

    Args:
        query: The search query string to find relevant document chunks.

    Returns:
        Tuple of (serialized_content, artifact) where:
        - serialized_content: A formatted string containing the retrieved chunks
          with stable IDs. Format: "[C1] Chunk from page X:\n...\n\n[C2] Chunk from page Y:\n..."
        - artifact: Dict with 'documents' (List of Document objects) and
          'citations' (Dict mapping chunk IDs to metadata)
    """
    # Retrieve documents from vector store
    docs = retrieve(query, k=4)

    # Serialize chunks with stable IDs and get citation map
    context, citation_map = serialize_chunks_with_ids(docs)

    # Return tuple: (serialized content with IDs, artifact with docs and citations)
    # This follows LangChain's content_and_artifact response format
    artifact = {"documents": docs, "citations": citation_map}

    return context, artifact
