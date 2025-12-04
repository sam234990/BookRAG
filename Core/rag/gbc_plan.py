from pydantic import BaseModel, model_validator, field_validator
from typing import List, Optional, Literal
from Core.provider.llm import LLM


# Step 1 Output Model
class QueryTypeResult(BaseModel):
    query_type: Literal["simple", "complex", "global"]


# Step 2 'complex' Output Model
class SubQuestion(BaseModel):
    question: str
    type: Literal["retrieval", "synthesis"]


class ComplexResult(BaseModel):
    sub_questions: List[SubQuestion]


# Step 2 'global' Output Models (统一为Filter列表)
class Filter(BaseModel):
    filter_type: Literal["section", "image", "table", "page"]
    filter_value: Optional[str] = None

    @field_validator("filter_value")
    def check_filter_value(cls, v, info):
        filter_type = info.data.get("filter_type")
        if v is not None and filter_type not in ["section", "page"]:
            raise ValueError(
                f"filter_value can only be set for 'section' or 'page', not for '{filter_type}'."
            )
        return v


class GlobalResult(BaseModel):
    filters: List[Filter]
    operation: Literal["COUNT", "LIST", "SUMMARIZE", "ANALYZE"]


# Final combined result model (for returning to the user)
class PlanResult(BaseModel):
    query_type: Literal["simple", "complex", "global"]
    original_query: str
    sub_questions: Optional[List[SubQuestion]] = None
    filters: Optional[List[Filter]] = None  # 原scope字段已改为filters
    operation: Optional[Literal["COUNT", "LIST", "SUMMARIZE", "ANALYZE"]] = None

    @model_validator(mode="after")
    def check_fields_for_type(self):
        # ... (校验逻辑更新以包含 operation)
        if self.query_type == "complex" and not self.sub_questions:
            raise ValueError(
                "For 'complex' query_type, 'sub_questions' must be provided."
            )
        if self.query_type == "global" and (not self.filters or not self.operation):
            raise ValueError(
                "For 'global' query_type, 'filters' and 'operation' must be provided."
            )
        if self.query_type == "simple" and (
            self.sub_questions or self.filters or self.operation
        ):
            raise ValueError(
                "For 'simple' query_type, all other specific fields must be null."
            )
        return self


# --- The Refactored Planner Class ---
class TaskPlanner:
    """
    A two-step Planner for user queries.
    Step 1: Classify the query type.
    Step 2: Process the query based on its type.
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def _get_classification_prompt(self, query: str) -> str:
        # Prompt for Step 1: Classification Only
        # 强调了 "simple" 和 "global" 的区别
        return f"""
You are an expert query analyzer. Your *only* task is to classify the user's question into one of three categories: "simple", "complex", or "global". Respond only with the specified JSON object.

Category Definitions:
1.  **simple**: The question can be fully answered by retrieving information from a **SINGLE, contiguous location** in the document (e.g., one specific paragraph, one complete table, or one figure).
    - This includes questions that require reasoning or comparison, **as long as all the necessary data is present within that single retrieved location.**
    - Example: "What is the title of Figure 2?"
    - Example: "How do 5% of the Latinos see economic upward mobility for their children?" -> This is SIMPLE because the answer can be found by looking at a **single chart or paragraph**. The system just needs to find that one location, find the '5%' data point, and extract the corresponding label.

2.  **complex**: The question requires decomposition into multiple simple sub-questions, where **each sub-question must be answered by a separate retrieval action**.
    - It often contains a **nested or indirect constraint** that requires a preliminary step to resolve before the main question can be answered.
    - The key is that **no single location contains all the necessary information** to answer the original query in one go.
    - Example: "What is the color of the personality vector in the soft-labled personality embedding matrix that with the highest Receptiviti score for User A2GBIFL43U1LKJ?" -> This is COMPLEX because it requires two **separate retrieval actions**: 1) A retrieval to find the vector with the 'highest score', and 2) a separate retrieval to find the 'color' associated with that vector.

3.  **global**: The question requires an **aggregation operation** (e.g., counting, listing, summarizing) over a set of items that are identified by a **clear structural filter**. The available filters are by layout type (such as `table`, `image`, `section`) or by a document range (such as `page` or `section`). This operation must be performed across all items that match the specified filter(s).
    - Example: "How many tables are in the document?" -> This is GLOBAL because the process is to **filter** for all items of type 'table' across the document, and then **aggregate** by counting them.
    - Example: "Summarize the discussion about 'data augmentation' in the 'Methodology' section." -> This is GLOBAL because the process is to first apply a **structural filter** (retrieve all content from the 'Methodology' section), and then perform an **aggregation** (summarize that content) with a focus on 'data augmentation'.

User Query: "{query}"
"""

    def _get_complex_prompt(self, query: str) -> str:
        # Prompt for Step 2 - 增加了对输出字段的明确说明
        return f"""
You are a query decomposition expert. You have been given a "complex" question. Your task is to break it down into a series of simple, atomic sub-questions and classify each one by type.

**Crucial Instructions:**
1.  Each `retrieval` sub-question MUST be a direct information retrieval task that can be answered independently by looking up a specific fact, number, or value in the document.
2.  **`retrieval` sub-questions MUST NOT depend on the answer of another sub-question.** They should be parallelizable. All logic for combining their results must be placed in a final `synthesis` question.
3.  A `synthesis` question requires comparing, calculating, or combining the answers of the previous `retrieval` questions. It does **NOT** require a new lookup in the document.

You MUST provide your response in a JSON object with a single key 'sub_questions', which contains a list of objects. Each object must have a 'question' (string) and a 'type' (string: "retrieval" or "synthesis").

--- EXAMPLE 1 (Correct Decomposition with Independent Lookups) ---
Complex Query: "What is the color of the personality vector in the soft-labled personality embedding matrix that with the highest Receptiviti score for User A2GBIFL43U1LKJ?"

Expected JSON Output:
{{
  "sub_questions": [
    {{
      "question": "What are all the Receptiviti scores for each personality vector for User A2GBIFL43U1LKJ?",
      "type": "retrieval"
    }},
    {{
      "question": "What is the mapping of personality vectors to their colors in the soft-labled personality embedding matrix?",
      "type": "retrieval"
    }},
    {{
      "question": "From the gathered scores, identify the personality vector with the highest score, and then find its corresponding color from the vector-to-color mapping.",
      "type": "synthesis"
    }}
  ]
}}
--- END EXAMPLE 1 ---

--- EXAMPLE 2 (Decomposition with retrieval and synthesis steps) ---
Complex Query: "According to the report, which one is greater in population in the survey? Foreign born Latinos, or the Latinos interviewed by cellphone?"

Expected JSON Output:
{{
  "sub_questions": [
    {{
      "question": "According to the report, what is the population of foreign born Latinos in the survey?",
      "type": "retrieval"
    }},
    {{
      "question": "According to the report, what is the population of Latinos interviewed by cellphone in the survey?",
      "type": "retrieval"
    }},
    {{
      "question": "Which of the two population counts is greater?",
      "type": "synthesis"
    }}
  ]
}}
--- END EXAMPLE 2 ---

Now, perform the decomposition for the following query.

Complex Query: "{query}"
"""

    def _get_global_prompt(self, query: str) -> str:
        return f"""
You are a highly specialized AI assistant. Your ONLY function is to analyze a "Global Query" and return a single, valid JSON object that specifies **both** the filtering steps and the final aggregation operation. You MUST NOT output any other text or explanation.

### INSTRUCTIONS & DEFINITIONS ###

1.  **Filters**: You MUST determine the list of `filters` to apply. Even if the filter is for the whole document (e.g., all tables), the `filters` list must be present.
    - `filter_type`: One of ["section", "image", "table", "page"].
        - `section`: Use for structural parts like chapters, sections, appendices, or references.
        - `image`: Use for visual elements like figures, images, pictures, or plots.
        - `table`: Use for tabular data.
        - `page`: Use for specific page numbers or ranges.
    - `filter_value`: (Optional) Can be provided for "section" (e.g., a section title) or "page" (e.g., '3-10' or '5'). **For "image" or "table", this value MUST be null.**

2.  **Operation**: Determine the final aggregation operation.
    - `operation`: One of ["COUNT", "LIST", "SUMMARIZE", "ANALYZE"].

### EXAMPLES OF YOUR TASK ###

User: "How many figures are in this paper from Page 3 to Page 10?"
Assistant: {{"filters": [{{"filter_type": "page", "filter_value": "3-10"}}, {{"filter_type": "image"}}], "operation": "COUNT"}}

User: "Summarize the discussion about 'data augmentation' in the 'Methodology' section."
Assistant: {{"filters": [{{"filter_type": "section", "filter_value": "Methodology"}}], "operation": "SUMMARIZE"}}

User: "How many chapters are in this report?"
Assistant: {{"filters": [{{"filter_type": "section"}}], "operation": "COUNT"}}

### YOUR CURRENT TASK ###

User: "{query}"
Assistant:
"""

    def _classify_query_type(
        self, query: str
    ) -> Literal["simple", "complex", "global"]:
        prompt = self._get_classification_prompt(query)
        result = self.llm.get_json_completion(prompt, schema=QueryTypeResult)
        return result.query_type

    def _process_complex_query(self, query: str) -> List[str]:
        prompt = self._get_complex_prompt(query)
        result = self.llm.get_json_completion(prompt, schema=ComplexResult)
        return result.sub_questions

    def _process_global_query(self, query: str) -> GlobalResult:
        prompt = self._get_global_prompt(query)
        result = self.llm.get_json_completion(prompt, schema=GlobalResult)
        return result

    def analyze(self, query: str) -> PlanResult:
        query_type = self._classify_query_type(query)

        if query_type == "simple":
            return PlanResult(query_type=query_type, original_query=query)

        elif query_type == "complex":
            sub_questions = self._process_complex_query(query)
            return PlanResult(
                query_type=query_type, original_query=query, sub_questions=sub_questions
            )

        elif query_type == "global":
            global_result = self._process_global_query(query)
            return PlanResult(
                query_type=query_type,
                original_query=query,
                filters=global_result.filters,
                operation=global_result.operation,
            )
