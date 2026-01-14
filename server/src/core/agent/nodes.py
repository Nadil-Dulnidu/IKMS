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
from .response_modal import QueryPlan
from .utils import (
    _extract_last_ai_content,
    _extract_citations_from_tool_message,
    _build_structured_context,
    format_final_answer_with_citations,
)
from .agents import (
    create_query_plan_agent,
    create_retrieval_agent,
    create_context_critic_agent,
    create_summarization_agent,
    create_verification_agent,
)
from src.config.logging import get_logger

logger = get_logger(__name__)


def query_plan_node(state: QAState) -> QAState:
    """Query Planning Agent node: analyzes question and creates search strategy.

    This node:
    - Sends the user's question to the Query Planning Agent.
    - The agent analyzes the question and decomposes it into sub-questions.
    - Returns a structured QueryPlan with search strategy and sub-questions.
    - Generates a markdown representation for frontend display.
    - Stores the QueryPlan object in `state["query_plan"]`.
    """
    question = state["question"]
    user_id = state.get("user_id", "unknown")

    logger.info(
        "Starting query planning", extra={"user": user_id, "action": "query_plan_start"}
    )

    # Create query plan agent (user-independent)
    query_plan_agent = create_query_plan_agent()

    # Invoke the query planning agent with the user's question
    result = query_plan_agent.invoke({"messages": [HumanMessage(content=question)]})

    query_plan = result.get("structured_response", None)

    if query_plan:
        logger.info(
            f"Query plan created with {len(query_plan.sub_questions)} sub-questions",
            extra={
                "user": user_id,
                "action": "query_plan_complete",
                "sub_question_count": len(query_plan.sub_questions),
            },
        )
        # Generate markdown representation for frontend display
        markdown_content = query_plan.generate_markdown()
        query_plan.markdown = markdown_content

        # Create an AIMessage with the markdown content to stream to frontend
        ai_message = AIMessage(content=markdown_content)

        return {
            "query_plan": query_plan.model_dump(),
            "messages": [ai_message],  # Add to messages for streaming
        }

    return {
        "query_plan": None,
    }


def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context from user's namespace.

    This node implements multi-call retrieval:
    - Uses sub-questions from query_plan if available, otherwise uses the original question.
    - Sends each query to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks from user's namespace.
    - Extracts citations from each ToolMessage.
    - Creates structured, organized context blocks for downstream agents.
    - Stores context and citations in state.
    """
    question = state["question"]
    query_plan = state["query_plan"]
    user_id = state["user_id"]

    logger.info(
        "Starting retrieval", extra={"user": user_id, "action": "retrieval_start"}
    )

    # Create user-specific retrieval agent
    retrieval_agent = create_retrieval_agent(user_id)

    # Determine which queries to use for retrieval
    queries = []
    if query_plan and query_plan["sub_questions"]:
        # Use decomposed sub-questions for focused retrieval
        queries = query_plan["sub_questions"]
    else:
        # Fall back to original question
        queries = [question]

    # Storage for comprehensive retrieval information
    structured_context_blocks = []
    all_citations = {}
    ai_messages = []
    tool_messages = []

    # Perform retrieval for each query
    for call_number, query in enumerate(queries, start=1):
        result = retrieval_agent.invoke({"messages": [HumanMessage(content=query)]})
        messages = result.get("messages", [])

        # Extract last ToolMessage from this invocation
        last_tool_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, ToolMessage)), None
        )
        tool_messages.append(last_tool_message)

        # Extract first AIMessage from this invocation and add to collection
        first_ai_msg = next(
            (msg for msg in messages if isinstance(msg, AIMessage)), None
        )
        if first_ai_msg:
            ai_messages.append(first_ai_msg)

        # Process each ToolMessage (typically one per query, but could be multiple)
        for tool_msg in tool_messages:
            # Extract the context content
            context_content = tool_msg.content

            # Extract citations from this tool message
            citations = _extract_citations_from_tool_message(tool_msg)
            all_citations.update(citations)  # Merge citations from all calls

            # Build structured context block
            structured_block = _build_structured_context(
                call_number, query, context_content
            )
            structured_context_blocks.append(structured_block)

    # Create structured context for downstream agents
    if structured_context_blocks:
        context = "\n\n".join(structured_context_blocks)
    else:
        context = ""

    logger.info(
        f"Retrieval complete: {len(all_citations)} citations found",
        extra={
            "user": user_id,
            "action": "retrieval_complete",
            "citation_count": len(all_citations),
        },
    )

    return {
        "messages": [AIMessage(content="")],
        "context": context,
        "citations": all_citations,
        "tool_inputs": ai_messages,
        "tool_outputs": tool_messages,
        "retrieval_count": len(queries),
    }


def extract_tool_inputs_node(state: QAState) -> QAState:
    """Extracts the tool input message from the state.

    Args:
        state (QAState): The state containing the tool inputs.

    Returns:
        QAState: The state containing the tool input message.
    """
    tool_inputs = list(reversed(state.get("tool_inputs", [])))
    retrieval_count = state.get("retrieval_count", 0)

    tool_input_message = tool_inputs[retrieval_count - 1]

    return {
        "messages": [tool_input_message],
    }


def extract_tool_outputs_node(state: QAState) -> QAState:
    """Extracts the tool output message from the state.

    Args:
        state (QAState): The state containing the tool outputs.

    Returns:
        QAState: The state containing the tool output message.
    """
    tool_outputs = list(reversed(state.get("tool_outputs", [])))
    retrieval_count = state.get("retrieval_count", 0)

    tool_output_message = tool_outputs[retrieval_count - 1]

    return {
        "messages": [tool_output_message],
        "retrieval_count": retrieval_count - 1,
    }


def should_stop_retrieval(state: QAState) -> bool:
    """Check if the retrieval process should stop based on the number of retrievals."""
    return state.get("retrieval_count") == 0


def context_critic_node(state: QAState) -> QAState:
    """Context Critic Agent node: filters and ranks retrieved chunks.

    This node:
    - Receives the raw context from retrieval_node
    - Analyzes each chunk's relevance to the question
    - Assigns relevance scores (HIGHLY RELEVANT / MARGINAL / IRRELEVANT)
    - Filters out irrelevant chunks
    - Reorders chunks by relevance
    - Provides rationales for filtering decisions
    - Generates markdown representation for frontend display
    - Stores filtered context and rationales in state
    """
    question = state["question"]
    raw_context = state.get("context", "")

    # If no context was retrieved, skip critic analysis
    if not raw_context or raw_context.strip() == "":
        return {
            "context": "",
            "context_rationale": ["No context available for analysis"],
        }

    # Create context critic agent (user-independent)
    context_critic_agent = create_context_critic_agent()

    # Prepare the input for the context critic
    user_content = f"""Question: {question}

Retrieved Context:
{raw_context}

Please analyze each chunk's relevance to the question, filter out irrelevant content, 
and provide the filtered context with your rationales."""

    # Invoke the context critic agent
    result = context_critic_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )

    # Extract the structured response
    critic_response = result.get("structured_response", None)

    if critic_response:
        # Get filtered context and rationales from the structured response
        filtered_context = critic_response.filtered_context
        context_rationale = critic_response.context_rationale

        # Generate markdown representation for frontend display
        markdown_content = critic_response.generate_markdown()
        critic_response.markdown = markdown_content

        # Create an AIMessage with the markdown content to stream to frontend
        ai_message = AIMessage(content=markdown_content)

        return {
            "context": filtered_context,
            "context_rationale": context_rationale,
            "messages": [ai_message],  # Add to messages for streaming
        }
    else:
        # Fallback: use original context if critic fails
        filtered_context = raw_context
        context_rationale = [
            "Context critic analysis unavailable - using original context"
        ]

        return {
            "context": filtered_context,
            "context_rationale": context_rationale,
            "messages": [AIMessage(content="")],
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

    # Create summarization agent (user-independent)
    summarization_agent = create_summarization_agent()

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)

    return {
        "messages": [AIMessage(content="")],
        "draft_answer": draft_answer,
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects the draft answer.

    This node:
    - Sends question + context + draft_answer to the Verification Agent.
    - Agent checks for hallucinations and unsupported claims.
    - Generates production-grade markdown with answer and citations.
    - Stores the final verified answer in `state["answer"]`.
    """
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")
    citations = state.get("citations", {})
    user_id = state.get("user_id", "unknown")

    logger.info(
        "Starting answer verification",
        extra={"user": user_id, "action": "verification_start"},
    )

    # Create verification agent (user-independent)
    verification_agent = create_verification_agent()

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

    # Generate production-grade markdown with answer and citations
    markdown_content = format_final_answer_with_citations(answer, citations)

    # Create AIMessage with the markdown content for streaming
    final_ai_message = AIMessage(content=markdown_content)

    return {
        "answer": answer,
        "messages": [final_ai_message],
    }
