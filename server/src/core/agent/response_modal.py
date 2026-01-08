from pydantic import BaseModel, Field
from typing import List, Optional


class QueryPlan(BaseModel):
    plan: Optional[str] = Field(description="Natural language search strategy")
    sub_questions: Optional[List[str]] = Field(
        description="Decomposed questions for focused retrieval"
    )
    markdown: Optional[str] = Field(
        default=None,
        description="Formatted markdown representation of the query plan for frontend display",
    )

    def generate_markdown(self) -> str:
        """
        Generate a production-grade markdown representation of the query plan.

        Returns:
            Formatted markdown string suitable for chat app display
        """
        sections = []

        # Add search strategy section
        if self.plan:
            sections.append(f"## 🔍 Search Strategy\n\n{self.plan}")
            sections.append("\n---")

        # Add sub-questions section
        if self.sub_questions and len(self.sub_questions) > 0:
            sections.append("\n## 📋 Search Queries")
            sections.append("")  # Empty line for spacing
            for i, question in enumerate(self.sub_questions, start=1):
                sections.append(f"{i}. {question}")

        return "\n".join(sections) if sections else "Query plan generated successfully."


class ContextCritic(BaseModel):
    filtered_context: str = Field(description="Context after filtering")
    context_rationale: List[str] = Field(description="Reasoning for context selection")
    markdown: Optional[str] = Field(
        default=None,
        description="Formatted markdown representation of the critic's analysis for frontend display",
    )

    def generate_markdown(self) -> str:
        """
        Generate a production-grade markdown representation of the context critic's analysis.

        Returns:
            Formatted markdown string suitable for chat app display
        """
        if not self.context_rationale or len(self.context_rationale) == 0:
            return "## ✅ Context Analysis\n\nAll retrieved context is relevant."

        sections = []

        # Add header
        sections.append("## 🔎 Context Quality Analysis")
        sections.append("")  # Empty line for spacing

        # Add rationale items
        for i, rationale in enumerate(self.context_rationale, 1):
            # Parse the rationale to extract relevance indicator if present
            if "HIGHLY RELEVANT" in rationale.upper():
                emoji = "✅"
            elif "MARGINAL" in rationale.upper():
                emoji = "⚠️"
            elif "IRRELEVANT" in rationale.upper():
                emoji = "❌"
            else:
                emoji = "📌"
            sections.append(f"{emoji} **Analysis {i}**")
            sections.append(f"{rationale}")
            sections.append("")

        sections.append("\n\n---")  # Empty line for spacing

        return "\n".join(sections).strip()
