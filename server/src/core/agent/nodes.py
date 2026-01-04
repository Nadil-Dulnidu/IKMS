from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core.llm import create_chat_model, create_reasoning_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    QUERY_PLAN_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool
from .response_modal import QueryPlan
from .utils import (
    _extract_last_ai_content,
    _extract_query_from_tool_message,
    _extract_artifacts_from_tool_message,
    _build_retrieval_trace,
    _build_structured_context,
)
from .agents import (
    query_plan_agent,
    retrieval_agent,
    summarization_agent,
    verification_agent,
)


def query_plan_node(state: QAState) -> QAState:
    """Query Planning Agent node: analyzes question and creates search strategy.

    This node:
    - Sends the user's question to the Query Planning Agent.
    - The agent analyzes the question and decomposes it into sub-questions.
    - Returns a structured QueryPlan with search strategy and sub-questions.
    - Stores the QueryPlan object in `state["query_plan"]`.
    """
    question = state["question"]

    # Invoke the query planning agent with the user's question
    result = query_plan_agent.invoke({"messages": [HumanMessage(content=question)]})

    query_plan = result.get("structured_response", None)

    return {
        "query_plan": query_plan.model_dump(),
    }


def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context from vector store with comprehensive message tracking.

    This node implements multi-call retrieval with full transparency:
    - Uses sub-questions from query_plan if available, otherwise uses the original question.
    - Sends each query to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks.
    - **Captures ALL ToolMessages** (not just the last one).
    - Extracts artifacts (document metadata) from each ToolMessage.
    - Builds human-readable retrieval traces documenting all calls.
    - Creates structured, organized context blocks for downstream agents.
    - Stores comprehensive information in state to prevent data loss.
    """
    question = state["question"]
    query_plan = state["query_plan"]

    # Determine which queries to use for retrieval
    queries = []
    if query_plan and query_plan["sub_questions"]:
        # Use decomposed sub-questions for focused retrieval
        queries = query_plan["sub_questions"]
    else:
        # Fall back to original question
        queries = [question]

    # Storage for comprehensive retrieval information
    retrieval_traces = []
    raw_context_blocks = []
    structured_context_blocks = []
    all_tool_messages = []

    # Perform retrieval for each query
    for call_number, query in enumerate(queries, start=1):
        result = retrieval_agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = result.get("messages", [])

        # Extract ALL ToolMessages from this invocation
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        all_tool_messages.extend(tool_messages)

        # Process each ToolMessage (typically one per query, but could be multiple)
        for tool_msg in tool_messages:
            # Extract the context content
            context_content = tool_msg.content
            raw_context_blocks.append(context_content)

            # Extract artifacts (document metadata)
            artifacts = _extract_artifacts_from_tool_message(tool_msg)

            # Build human-readable trace
            trace = _build_retrieval_trace(call_number, query, tool_msg, artifacts)
            retrieval_traces.append(trace)

            # Build structured context block
            structured_block = _build_structured_context(
                call_number, query, context_content
            )
            structured_context_blocks.append(structured_block)

    # Consolidate all information
    # 1. Create human-readable retrieval trace log
    retrieval_trace_log = (
        "\n\n".join(retrieval_traces) if retrieval_traces else "No retrieval calls made"
    )

    # 2. Create structured context for downstream agents
    if structured_context_blocks:
        context = "\n\n".join(structured_context_blocks)
    else:
        context = ""

    return {
        "context": context,
        "retrieval_traces": retrieval_trace_log,
        "raw_context_blocks": raw_context_blocks if raw_context_blocks else None,
    }


def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: generates draft answer from context.

    This node:
    - Sends question + context to the Summarization Agent.
    - Agent responds with a draft answer grounded only in the context.
    - Stores the draft answer in `state["draft_answer"]`.
    """
    question = state["question"]
    context = state.get("context")

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)

    return {
        "draft_answer": draft_answer,
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects the draft answer.

    This node:
    - Sends question + context + draft_answer to the Verification Agent.
    - Agent checks for hallucinations and unsupported claims.
    - Stores the final verified answer in `state["answer"]`.
    """
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""Question: {question}
    
                    Context:
                    {context}

                    Draft Answer:
                    {draft_answer}

                    Please verify and correct the draft answer, removing any unsupported claims.
                    """

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    answer = _extract_last_ai_content(messages)

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            ai_message = msg
            break

    return {
        "answer": answer,
        "messages": [ai_message],
    }
