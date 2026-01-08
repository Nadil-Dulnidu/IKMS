from langchain.agents import create_agent

from src.core.llm import create_chat_model, create_reasoning_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    QUERY_PLAN_SYSTEM_PROMPT,
    CONTEXT_CRITIC_SYSTEM_PROMPT,
)
from .tools import create_retrieval_tool
from .response_modal import QueryPlan, ContextCritic


def create_query_plan_agent():
    """Create a query planning agent (user-independent)."""
    return create_agent(
        model=create_chat_model(),
        tools=[],
        system_prompt=QUERY_PLAN_SYSTEM_PROMPT,
        response_format=QueryPlan,
    )


def create_retrieval_agent(user_id: str):
    """
    Create a user-specific retrieval agent.

    Args:
        user_id: The user ID whose namespace to search

    Returns:
        Retrieval agent configured for the specific user
    """
    retrieval_tool = create_retrieval_tool(user_id)
    return create_agent(
        model=create_chat_model(),
        tools=[retrieval_tool],
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
    )


def create_context_critic_agent():
    """Create a context critic agent (user-independent)."""
    return create_agent(
        model=create_chat_model(),
        tools=[],
        system_prompt=CONTEXT_CRITIC_SYSTEM_PROMPT,
        response_format=ContextCritic,
    )


def create_summarization_agent():
    """Create a summarization agent (user-independent)."""
    return create_agent(
        model=create_chat_model(),
        tools=[],
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
    )


def create_verification_agent():
    """Create a verification agent (user-independent)."""
    return create_agent(
        model=create_reasoning_model(),
        tools=[],
        system_prompt=VERIFICATION_SYSTEM_PROMPT,
    )
