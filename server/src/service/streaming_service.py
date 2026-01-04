"""
Pluggable adapter-based streaming service for the travel system.

This demonstrates how to use the LangGraphToVercelAdapter for clean
separation of concerns between agentic logic and streaming protocol.

Key features:
- Uses pluggable adapter instead of tightly-coupled streaming
- No graph-specific logic in streaming layer
- Easily reusable with other LangGraph graphs
- Customizable message extraction strategies
"""

from typing import AsyncGenerator
from langchain_core.messages import HumanMessage
from src.utils.langgraph_vercel_adapter import stream_langgraph_to_vercel
from src.core.agent.state import QAState
from src.core.agent.graph import get_qa_graph


async def stream_travel_system_chat(
    message: str, thread_id: str, resume: bool = False
) -> AsyncGenerator[str, None]:
    """
    Stream the travel system using the pluggable adapter.

    This uses the LangGraphToVercelAdapter which provides:
    - Clean separation between graph logic and streaming
    - Works with any LangGraph graph
    - No hardcoded field checks (requirements, itinerary, bookings)
    - Pluggable message extraction

    Args:
        message: User message or resume input
        thread_id: Thread ID for conversation continuity
        resume: Whether to resume from an interrupt

    Yields:
        SSE-formatted strings following Vercel Data Stream Protocol
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: QAState = {
        "question": message,
        "context": None,
        "draft_answer": None,
        "answer": None,
        "messages": [HumanMessage(content=message)],
    }

    # Stream using the pluggable adapter!
    # No need to specify stream_mode or graph-specific logic
    # Configure custom data fields to stream alongside messages
    async for event in stream_langgraph_to_vercel(
        graph=get_qa_graph(),
        initial_state=initial_state,
        config=config,
        custom_data_fields=[],
    ):
        yield event


async def stream_any_langgraph_graph(
    graph,  # Any compiled LangGraph graph
    message: str,
    thread_id: str,
    user_id: str,  # User ID for namespace isolation
) -> AsyncGenerator[str, None]:
    """
    Generic streaming function that works with ANY LangGraph graph.

    This demonstrates the true power of the pluggable adapter:
    - No graph-specific code
    - No hardcoded field checks
    - Just pass your graph and state

    Requirements:
    - Graph state must extend MessagesState
    - Nodes should return AIMessage objects
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Minimal initial state (works with any MessagesState-based graph)
    initial_state: QAState = {
        "user_id": user_id,  # User ID for namespace isolation
        "question": message,
        "context": None,
        "draft_answer": None,
        "answer": None,
        "messages": [HumanMessage(content=message)],
        "tool_message": None,
    }

    # Same adapter works for ANY graph!
    async for event in stream_langgraph_to_vercel(
        graph=graph,
        initial_state=initial_state,
        config=config,
    ):
        yield event
