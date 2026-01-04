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
    2. Summarization Agent: generates `draft_answer` from `question` + `context`
    3. Verification Agent: produces final `answer` from `question` + `context` + `draft_answer`
    """

    question: str
    query_plan: QueryPlan | None
    context: str | None
    draft_answer: str | None
    answer: str | None
