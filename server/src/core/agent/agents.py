from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    QUERY_PLAN_SYSTEM_PROMPT,
)
from .tools import retrieval_tool
from .response_modal import QueryPlan
from .agents import create_agent, create_chat_model, create_reasoning_model

query_plan_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=QUERY_PLAN_SYSTEM_PROMPT,
    response_format=QueryPlan,
)

retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_reasoning_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)
