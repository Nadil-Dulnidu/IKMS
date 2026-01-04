"""LangGraph state schema for the multi-agent QA flow."""

# from typing import TypedDict
from typing import List
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from .response_modal import QueryPlan


class QAState(MessagesState):
    """State schema for the linear multi-agent QA flow.

    The state flows through multiple agents:
    1. Query Planning Agent: analyzes question and creates search strategy
       - populates `query_plan`
    2. Retrieval Agent: performs multiple strategic retrieval calls
       - populates `context`, `retrieval_traces`, and `raw_context_blocks`
    3. Context Critic Agent: filters and ranks retrieved chunks
       - populates `context` (filtered), `context_rationale`
    4. Summarization Agent: generates `draft_answer` from `question` + `context`
    5. Verification Agent: produces final `answer` from `question` + `context` + `draft_answer`
    """

    question: str
    query_plan: QueryPlan | None = None
    context: str = ""
    draft_answer: str = ""
    answer: str = ""

    # Multi-call retrieval transparency fields
    retrieval_traces: str = ""  # Human-readable log of all retrieval calls
    raw_context_blocks: List[str] = []

    # Context Critic fields
    context_rationale: List[str] = []
