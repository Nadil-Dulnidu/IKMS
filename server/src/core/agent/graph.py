"""LangGraph orchestration for the linear multi-agent QA flow."""

from functools import lru_cache
from typing import Any, Dict

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from src.core.agent.nodes import (
    retrieval_node,
    context_critic_node,
    summarization_node,
    verification_node,
    query_plan_node,
    extract_tool_inputs_node,
    extract_tool_outputs_node,
    should_stop_retrieval,
)
from src.core.agent.state import QAState
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


memory_saver = InMemorySaver()


def create_qa_graph() -> Any:
    """Create and compile the linear multi-agent QA graph.

    The graph executes in order:
    1. Query Planning Agent: analyzes question and creates search strategy
    2. Retrieval Agent: gathers context from vector store
    3. Context Critic Agent: filters and ranks retrieved chunks
    4. Summarization Agent: generates draft answer from filtered context
    5. Verification Agent: verifies and corrects the answer

    Returns:
        Compiled graph ready for execution.
    """
    builder = StateGraph(QAState)

    # Add nodes for each agent
    builder.add_node("query_plan", query_plan_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("context_critic", context_critic_node)
    builder.add_node("summarization", summarization_node)
    builder.add_node("verification", verification_node)
    builder.add_node("extract_tool_inputs", extract_tool_inputs_node)
    builder.add_node("extract_tool_outputs", extract_tool_outputs_node)

    # Define linear flow: START → query_plan → retrieval → context_critic → summarization → verification → END
    builder.add_edge(START, "query_plan")
    builder.add_edge("query_plan", "retrieval")
    builder.add_edge("retrieval", "extract_tool_inputs")
    builder.add_edge("extract_tool_inputs", "extract_tool_outputs")

    builder.add_conditional_edges(
        "extract_tool_outputs",
        should_stop_retrieval,
        {True: "context_critic", False: "extract_tool_inputs"},
    )

    builder.add_edge("context_critic", "summarization")
    builder.add_edge("summarization", "verification")
    builder.add_edge("verification", END)

    return builder.compile(checkpointer=memory_saver)


@lru_cache(maxsize=1)
def get_qa_graph() -> Any:
    """Get the compiled QA graph instance (singleton via LRU cache)."""
    return create_qa_graph()


def run_qa_flow(question: str) -> Dict[str, Any]:
    """Run the complete multi-agent QA flow for a question.

    This is the main entry point for the QA system. It:
    1. Initializes the graph state with the question
    2. Executes the linear agent flow (Retrieval -> Summarization -> Verification)
    3. Extracts and returns the final results

    Args:
        question: The user's question about the vector databases paper.

    Returns:
        Dictionary with keys:
        - `answer`: Final verified answer
        - `draft_answer`: Initial draft answer from summarization agent
        - `context`: Retrieved context from vector store
    """
    graph = get_qa_graph()

    initial_state: QAState = {
        "question": question,
        "messages": [HumanMessage(content=question)],
    }

    config = {"configurable": {"thread_id": "thread_01"}}

    final_state = graph.invoke(initial_state, config=config)

    return final_state


if __name__ == "__main__":
    result = run_qa_flow("What are the finacials of XYZ Company Limited?")
    print(result)
