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


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


# Define agents at module level for reuse
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
    """Retrieval Agent node: gathers context from vector store.

    This node:
    - Uses sub-questions from query_plan if available, otherwise uses the original question.
    - Sends each query to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks.
    - Extracts the tool's content (CONTEXT string) from the ToolMessage.
    - Consolidates all contexts and stores in `state["context"]`.
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

    all_contexts = []
    all_messages = []

    # Perform retrieval for each query
    for query in queries:
        result = retrieval_agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = result.get("messages", [])
        all_messages.extend(messages)

        # Extract context from ToolMessage
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                all_contexts.append(msg.content)
                break

    # Consolidate all contexts into a single string
    if all_contexts:
        context = "\n\n---\n\n".join(all_contexts)
    else:
        context = ""

    return {"context": context}


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
