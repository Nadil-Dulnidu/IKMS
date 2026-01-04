from typing import List, Tuple, Dict

from langchain_core.documents import Document


def serialize_chunks(docs: List[Document]) -> str:
    """Serialize a list of Document objects into a formatted CONTEXT string.

    Formats chunks with indices and page numbers as specified in the PRD:
    - Chunks are numbered (Chunk 1, Chunk 2, etc.)
    - Page numbers are included in the format "page=X"
    - Produces a clean CONTEXT section for agent consumption

    Args:
        docs: List of Document objects with metadata.

    Returns:
        Formatted string with all chunks serialized.
    """
    context_parts = []

    for idx, doc in enumerate(docs, start=1):
        # Extract page number from metadata
        page_num = doc.metadata.get("page") or doc.metadata.get(
            "page_number", "unknown"
        )

        # Format chunk with index and page number
        chunk_header = f"Chunk {idx} (page={page_num}):"
        chunk_content = doc.page_content.strip()

        context_parts.append(f"{chunk_header}\n{chunk_content}")

    return "\n\n".join(context_parts)


def serialize_chunks_with_ids(docs: List[Document]) -> Tuple[str, Dict[str, Dict]]:
    """Serialize chunks with stable IDs and return citation mapping.

    This function creates:
    1. A formatted context string with stable chunk IDs [C1], [C2], etc.
    2. A citation map that links each chunk ID to its metadata

    Args:
        docs: List of Document objects with metadata.

    Returns:
        Tuple of (formatted_context, citation_map) where:
        - formatted_context: String with chunks labeled by stable IDs
        - citation_map: Dict mapping chunk IDs to metadata (page, snippet, source)

    Example:
        context, citations = serialize_chunks_with_ids(docs)
        # context = "[C1] Chunk from page 5:\nVector databases use HNSW..."
        # citations = {"C1": {"page": 5, "snippet": "Vector databases...", "source": "doc.pdf"}}
    """
    context_parts = []
    citation_map = {}

    for idx, doc in enumerate(docs, start=1):
        # Generate stable chunk ID
        chunk_id = f"C{idx}"

        # Extract metadata
        page_num = doc.metadata.get("page") or doc.metadata.get(
            "page_number", "unknown"
        )
        source = doc.metadata.get("source", "unknown")
        chunk_content = doc.page_content.strip()

        # Format chunk with stable ID
        chunk_header = f"[{chunk_id}] Chunk from page {page_num}:"
        context_parts.append(f"{chunk_header}\n{chunk_content}")

        # Build citation map entry
        citation_map[chunk_id] = {
            "page": page_num,
            "snippet": (
                chunk_content[:150] + "..."
                if len(chunk_content) > 150
                else chunk_content
            ),
            "source": source,
            "full_content": chunk_content,  # Include full content for reference
        }

    return "\n\n".join(context_parts), citation_map
