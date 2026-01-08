from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


def _extract_citations_from_tool_message(tool_msg: ToolMessage) -> Dict[str, Dict]:
    """Extract citation map from a ToolMessage.

    The retrieval_tool returns (content, artifact) where artifact is a dict with
    'documents' and 'citations'. This extracts the citations mapping.

    Returns:
        Dict mapping chunk IDs to metadata (page, snippet, source)
    """
    if hasattr(tool_msg, "artifact") and tool_msg.artifact:
        if isinstance(tool_msg.artifact, dict) and "citations" in tool_msg.artifact:
            return tool_msg.artifact["citations"]
    return {}


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


def format_final_answer_with_citations(answer: str, citations: Dict[str, Dict]) -> str:
    """
    Format the final answer with citations as production-grade markdown.

    Args:
        answer: The verified final answer text (may contain citation markers like [C1], [C2])
        citations: Dict mapping chunk IDs (e.g., "C1") to metadata (page, snippet, source)

    Returns:
        Beautifully formatted markdown with answer and references section
    """
    sections = []

    # Add the main answer section
    sections.append("## 💡 Answer")
    sections.append("")
    sections.append(answer)
    sections.append("")
    sections.append("\n---")

    # Add citations/references section if citations exist
    if citations and len(citations) > 0:
        sections.append("\n## 📚 References")
        sections.append("")

        # Sort citations by chunk ID (C1, C2, C3, etc.)
        sorted_citations = sorted(citations.items(), key=lambda x: x[0])

        for chunk_id, metadata in sorted_citations:
            # Extract metadata fields
            page = metadata.get("page", "N/A")
            snippet = metadata.get("snippet", "")
            source = metadata.get("source", "Document")

            # Format each citation
            sections.append(f"**[{chunk_id}]** - {source}, Page {page}")
            if snippet:
                # Truncate snippet if too long
                max_snippet_length = 150
                if len(snippet) > max_snippet_length:
                    snippet = snippet[:max_snippet_length] + "..."
                sections.append(f"> {snippet}")
            sections.append("")

    return "\n".join(sections).strip()
