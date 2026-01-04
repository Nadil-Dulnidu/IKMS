from typing import List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


def _extract_query_from_tool_message(tool_msg: ToolMessage) -> str:
    """Extract the original query from a ToolMessage.

    The query is stored in the tool_msg.name or can be inferred from context.
    For now, we'll extract it from the artifact if available.
    """
    # Try to get from the tool call metadata if available
    if hasattr(tool_msg, "additional_kwargs") and tool_msg.additional_kwargs:
        query = tool_msg.additional_kwargs.get("query", "Unknown query")
        if query != "Unknown query":
            return query

    # Fallback: return a placeholder
    return "Query information not available"


def _extract_artifacts_from_tool_message(tool_msg: ToolMessage) -> list:
    """Extract document artifacts from a ToolMessage.

    The retrieval_tool returns (content, artifact) where artifact is a list of Documents.
    This artifact is stored in tool_msg.artifact.
    """
    if hasattr(tool_msg, "artifact") and tool_msg.artifact:
        return tool_msg.artifact
    return []


def _build_retrieval_trace(
    call_number: int, query: str, tool_msg: ToolMessage, artifacts: list
) -> str:
    """Build a human-readable trace for a single retrieval call.

    Args:
        call_number: The sequential number of this retrieval call
        query: The query string used for retrieval
        tool_msg: The ToolMessage containing the retrieval results
        artifacts: List of Document objects retrieved

    Returns:
        A formatted string describing this retrieval call
    """
    chunks_retrieved = len(artifacts)

    # Extract unique page numbers from artifacts
    page_numbers = set()
    if artifacts:
        for doc in artifacts:
            if hasattr(doc, "metadata") and "page" in doc.metadata:
                page_numbers.add(doc.metadata["page"])

    pages_str = (
        ", ".join(str(p) for p in sorted(page_numbers)) if page_numbers else "N/A"
    )

    trace = f"""Retrieval Call {call_number}:
                Query: "{query}"
                Chunks Retrieved: {chunks_retrieved}
                Sources: Pages {pages_str}"""

    return trace


def _build_structured_context(
    call_number: int, query: str, context_content: str
) -> str:
    """Build a structured context block for a single retrieval call.

    Args:
        call_number: The sequential number of this retrieval call
        query: The query string used for retrieval
        context_content: The serialized context content from the tool

    Returns:
        A formatted context block with clear separation and metadata
    """
    return f"""=== RETRIEVAL CALL {call_number} (query: "{query}") ===

{context_content}"""
