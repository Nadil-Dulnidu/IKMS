"""LangGraph state schema for the multi-agent QA flow."""

# from typing import TypedDict
from typing import List, Dict
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from .response_modal import QueryPlan


class QAState(MessagesState):
    """State schema for the linear multi-agent QA flow.

    The state flows through multiple agents:
    1. Query Planning Agent: analyzes question and creates search strategy
       - populates `query_plan`
    2. Retrieval Agent: performs multiple strategic retrieval calls
       - populates `context` and `citations`
    3. Context Critic Agent: filters and ranks retrieved chunks
       - populates `context` (filtered), `context_rationale`
    4. Summarization Agent: generates `draft_answer` from `question` + `context`
    5. Verification Agent: produces final `answer` from `question` + `context` + `draft_answer`
    """

    user_id: str  # User ID for namespace isolation
    question: str
    query_plan: QueryPlan | None = None
    context: str = ""
    draft_answer: str = ""
    answer: str = ""

    # Context Critic fields
    context_rationale: List[str] = []

    # Citation fields
    citations: List[Dict[str, Dict]] = (
        []
    )  # Chunk ID → metadata mapping (page, snippet, source)

    # Tool extraction fields
    tool_inputs: List[Dict] = []  # Extracted tool call inputs from AI messages
    tool_outputs: List[Dict] = []
    retrieval_count: int = 0  # Extracted tool outputs from ToolMessages
