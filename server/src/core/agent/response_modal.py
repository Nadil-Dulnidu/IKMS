from pydantic import BaseModel, Field
from typing import List, Optional


class QueryPlan(BaseModel):
    plan: Optional[str] = Field(description="Natural language search strategy")
    sub_questions: Optional[List[str]] = Field(
        description="Decomposed questions for focused retrieval"
    )


class ContextCritic(BaseModel):
    filtered_context: str = Field(description="Context after filtering")
    context_rationale: List[str] = Field(description="Reasoning for context selection")
