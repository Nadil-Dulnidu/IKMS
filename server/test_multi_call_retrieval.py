"""
Test script to demonstrate the Multi-Call Retrieval Tool feature.

This script shows how the enhanced retrieval_node captures all ToolMessages,
builds retrieval traces, and creates structured context blocks.
"""

from src.core.agent.graph import run_qa_flow
import json


def print_section(title: str, content: str = None):
    """Print a formatted section with title and optional content."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    if content:
        print(content)


def test_multi_call_retrieval():
    """Test the multi-call retrieval feature with a complex question."""

    # Complex question that should trigger multiple retrieval calls
    question = "What are the different indexing strategies for vector databases and how do they impact query performance?"

    print_section("TESTING MULTI-CALL RETRIEVAL FEATURE")
    print(f"\nQuestion: {question}\n")

    # Run the QA flow
    print("Running QA flow...")
    result = run_qa_flow(question)

    # Display Query Plan
    print_section("QUERY PLAN")
    if result.get("query_plan"):
        query_plan = result["query_plan"]
        print(f"\nSearch Strategy: {query_plan.get('search_strategy', 'N/A')}")
        print(f"\nSub-questions:")
        for i, sub_q in enumerate(query_plan.get("sub_questions", []), 1):
            print(f"  {i}. {sub_q}")
    else:
        print("No query plan generated (using original question)")

    # Display Retrieval Traces
    print_section("RETRIEVAL TRACES")
    if result.get("retrieval_traces"):
        print(f"\n{result['retrieval_traces']}")
    else:
        print("No retrieval traces available")

    # Display Raw Context Blocks
    print_section("RAW CONTEXT BLOCKS")
    if result.get("raw_context_blocks"):
        print(f"\nTotal context blocks: {len(result['raw_context_blocks'])}")
        for i, block in enumerate(result["raw_context_blocks"], 1):
            print(f"\n--- Block {i} ---")
            # Show first 200 characters of each block
            preview = block[:200] + "..." if len(block) > 200 else block
            print(preview)
    else:
        print("No raw context blocks available")

    # Display Structured Context (preview)
    print_section("STRUCTURED CONTEXT (Preview)")
    if result.get("context"):
        # Show first 500 characters
        preview = (
            result["context"][:500] + "..."
            if len(result["context"]) > 500
            else result["context"]
        )
        print(f"\n{preview}")
        print(f"\n\nTotal context length: {len(result['context'])} characters")
    else:
        print("No context available")

    # Display Draft Answer
    print_section("DRAFT ANSWER")
    if result.get("draft_answer"):
        print(f"\n{result['draft_answer']}")
    else:
        print("No draft answer available")

    # Display Final Answer
    print_section("FINAL VERIFIED ANSWER")
    if result.get("answer"):
        print(f"\n{result['answer']}")
    else:
        print("No final answer available")

    # Summary Statistics
    print_section("SUMMARY STATISTICS")
    stats = {
        "Query Plan Generated": bool(result.get("query_plan")),
        "Number of Sub-questions": (
            len(result.get("query_plan", {}).get("sub_questions", []))
            if result.get("query_plan")
            else 0
        ),
        "Number of Retrieval Calls": len(result.get("raw_context_blocks", [])),
        "Total Context Length": len(result.get("context", "")),
        "Draft Answer Length": len(result.get("draft_answer", "")),
        "Final Answer Length": len(result.get("answer", "")),
    }

    print()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print_section("TEST COMPLETE")

    return result


def test_simple_question():
    """Test with a simple question that may not trigger query decomposition."""

    question = "What is a vector database?"

    print_section("TESTING WITH SIMPLE QUESTION")
    print(f"\nQuestion: {question}\n")

    print("Running QA flow...")
    result = run_qa_flow(question)

    # Display Retrieval Traces
    print_section("RETRIEVAL TRACES")
    if result.get("retrieval_traces"):
        print(f"\n{result['retrieval_traces']}")

    # Display Final Answer
    print_section("FINAL ANSWER")
    if result.get("answer"):
        print(f"\n{result['answer']}")

    print_section("TEST COMPLETE")

    return result


if __name__ == "__main__":
    print(
        """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║        MULTI-CALL RETRIEVAL TOOL WITH MESSAGE ORGANIZATION TEST          ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    )

    # Test 1: Complex question with multiple retrieval calls
    print("\n\n>>> TEST 1: Complex Question (Multiple Retrieval Calls)")
    result1 = test_multi_call_retrieval()

    # Test 2: Simple question
    print("\n\n>>> TEST 2: Simple Question (Single Retrieval Call)")
    result2 = test_simple_question()

    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
