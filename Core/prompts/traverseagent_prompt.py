from pydantic import BaseModel, Field


# Define a Pydantic schema for the LLM's navigation decision
class NavigatorDecision(BaseModel):
    choice: int = Field(description="The number of the chosen sub-section. 0 to stop.")
    reason: str = Field(description="A brief explanation for the choice.")


# This prompt instructs the LLM to act as a navigator AND to output a structured JSON object.
NAVIGATOR_PROMPT_TEMPLATE = """
You are a research assistant navigating a document outline to answer a user's question.
Your task is to analyze the user's query, your current location summary, and a JSON array detailing the available nodes to explore next. Then, you must decide which node is the most relevant one to explore.

**User's Question**: "{query}"

**Current Section Summary**:
---
{current_summary}
---

**Available Nodes to Explore Next (in JSON format)**:
---
{options_str}
---

**Your Decision**:
Based on the query and the JSON data for each node, make your choice. Your response MUST be a valid JSON object that conforms to the following schema:
{{
  "choice": "<integer: The number of the most relevant node from the JSON array above.>",
  "reason": "<string: A brief, one-sentence explanation for your choice.>"
}}
"""

# If the context does not contain enough information to answer the question, output "Final Answer: Not answerable".

# This is the base instruction for the final answer generation.
# The agent will programmatically add the multimodal context to this.

ANSWER_GENERATOR_INSTRUCTION_TEMPLATE = """
User Question: {query}

You are a helpful AI assistant. Please provide a comprehensive answer to the user's question based on the following context, which may include text, tables, and images.
First, provide a step-by-step explanation of your reasoning.
Then, provide a clear, final answer prefixed with "Final Answer:".

--- START OF CONTEXT ---
{context_str}
--- END OF CONTEXT ---

Explanation:
"""
