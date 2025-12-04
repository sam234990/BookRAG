from typing import List
from pydantic import BaseModel, Field


class QuestionEntity(BaseModel):
    """
    Represents an entity extracted from a question.
    """

    entity_name: str = Field(default="")
    entity_type: str = Field(default="")


class QuestionEntityExtraction(BaseModel):
    """
    Represents the result of entity extraction from a question.
    """

    entities: List[QuestionEntity]


QUESTION_ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "TIME",
    "MONEY",
    "PERCENTAGE",
    "PRODUCT",
    "EVENT",
    "LANGUAGE",
    "NATIONALITY",
    "RELIGION",
    "TITLE",
    "PROFESSION",
    "ANIMAL",
    "PLANT",
    "DISEASE",
    "MEDICATION",
    "CHEMICAL",
    "MATERIAL",
    "COLOR",
    "SHAPE",
    "MEASUREMENT",
    "WEATHER",
    "NATURAL_DISASTER",
    "AWARD",
    "LAW",
    "CRIME",
    "TECHNOLOGY",
    "SOFTWARE",
    "HARDWARE",
    "VEHICLE",
    "FOOD",
    "DRINK",
    "SPORT",
    "MUSIC_GENRE",
    "INSTRUMENT",
    "ARTWORK",
    "BOOK",
    "MOVIE",
    "TV_SHOW",
    "ACADEMIC_SUBJECT",
    "SCIENTIFIC_THEORY",
    "POLITICAL_PARTY",
    "CURRENCY",
    "STOCK_SYMBOL",
    "FILE_TYPE",
    "PROGRAMMING_LANGUAGE",
    "MEDICAL_PROCEDURE",
    "CELESTIAL_BODY",
    # Academic & Technical Extensions
    "TASK_OR_PROBLEM",
    "MODEL_OR_ARCHITECTURE",
    "METHOD_OR_TECHNIQUE",
    "DATASET_OR_CORPUS",
    "EVALUATION_METRIC",
    "PARAMETER_OR_VARIABLE",
    "BENCHMARK",
    "RESEARCH_FIELD",
    "PUBLICATION_VENUE",
    # Document Structure Entities
    "SECTION_TITLE",
    "EQUATION_OR_FORMULA",
    "TABLE",
    "IMAGE",
]

# Question Entity Extraction Prompt
QUESTION_EE_PROMPT = """
-Goal-
Given a question and a list of predefined entity types, identify all potential entities within that question and describe them based on the context of the question.

-Output Format-
The output must be a single, valid JSON object that adheres to the structure and field descriptions below.

### JSON Structure
```json
{{
  "entities": [
    {{
      "entity_name": "<String>",
      "entity_type": "<String>"
    }}
  ]
}}
```

### Field Descriptions
* **`entity_name`** (String): The name of the entity, extracted verbatim from the question.
* **`entity_type`** (String): The category of the entity. It MUST be one of the following types: {entity_types}

-Instructions-
1.  **Analyze the Question**: Carefully read the input question to understand its main subject and the context it provides.
2.  **Comprehensive Entity Identification**: Scrutinize the question to identify **all** potential entities. Your extraction should be exhaustive. It is crucial to understand that an entity can be a real-world concept (e.g., a person, product, or location), but it can **also refer to structural or contextual elements** within a document, image, or user interface. You must identify both types. For example, extract real-world entities like `'Google'` or `'the Eiffel Tower'`, as well as contextual references like `'Figure 3'` or `'Table A'`.
3.  **Extract and Describe**: For each entity identified, create a JSON object with the following key-value pairs:
    * `entity_name`: The exact name of the entity from the question.
    * `entity_type`: The appropriate category from the provided list.
4.  **Assemble the Final JSON**: Combine all the entity objects into a list and place it within the final JSON structure under the `"entities"` key.

-Rules-
* The entities must be based **strictly** on the text of the question provided. Do not infer information or use any external knowledge.
* The value for `entity_type` must exactly match one of the items from the `entity_types` list.
* Fallback Rule: If, after a thorough analysis, no specific, smaller entities can be identified within the question, you must treat the entire question as a single entity. In this case, set the `entity_name` to the full text of the question and assign it the `entity_type` of "TASK_OR_PROBLEM". Do not return an empty list.

######################
-Real Data-
######################
Entity_types: {entity_types}
Question: {input_text}
######################
Output:
"""

QUESTION_ENT_PROMPT = """
-Goal-
Given a question, a list of predefined entity types, and a list of potentially relevant retrieved entities, identify all entities within the question and describe them.


-Output Format-
The output must be a single, valid JSON object that adheres to the structure and field descriptions below.

### JSON Structure
```json
{{
  "entities": [
    {{
      "entity_name": "<String>",
      "entity_type": "<String>"
    }}
  ]
}}
```

### Field Descriptions
* **`entity_name`** (String): The name of the entity, extracted verbatim from the question.
* **`entity_type`** (String): The category of the entity. It MUST be one of the following types: {entity_types}

-Retrieved Information-
You will be provided with a list of `retrieved_entities` containing `name` and `type`. These are pre-identified entities that might be present in the question. Your first step should be to check this list for relevant entities before searching for new ones.


-Instructions-
1.  **Analyze and Incorporate Retrieved Entities**: Carefully read the input question and review the `retrieved_entities` list. If any of the retrieved entities are present and contextually accurate in the question, you should include them in your final output.
2.  **Complete the Extraction**: After processing the retrieved list, perform a comprehensive scan of the question to find any **additional** entities that were not included in the `retrieved_entities`. It is crucial to understand that an entity can be a real-world concept (e.g., a person, product, or location), but it can **also refer to structural or contextual elements** within a document, image, or user interface. You must identify both types. For example, extract real-world entities like `'Google'` or `'the Eiffel Tower'`, as well as contextual references like `'Figure 3'` or `'Table A'`.
3.  **Extract and Describe**: For each identified entity (both from the retrieved list and those you found), create a JSON object with the following key-value pairs:
      * `entity_name`: The exact name of the entity from the question.
      * `entity_type`: The appropriate category from the provided list.
4.  **Assemble the Final JSON**: Combine all the entity objects into a list and place it within the final JSON structure under the `"entities"` key.

-Rules-
  * The final list of entities must be based **strictly** on the text of the question provided. Do not infer information or use any external knowledge.
  * The value for `entity_type` must exactly match one of the items from the `entity_types` list.
  * **Fallback Rule**: If the `retrieved_entities` list is empty AND no specific, smaller entities can be identified within the question after a thorough analysis, you must treat the entire question as a single entity. In this case, set the `entity_name` to the full text of the question and assign it the `entity_type` of "TASK_OR_PROBLEM". Do not return an empty list.

######################
-Real Data-
######################
Entity_types: {entity_types}
Retrieved_entities: {retrieved_entities}
Question: {input_text}
######################
Output:
"""


class SectionSelection(BaseModel):
    """
    Represents the selection of a section based on user intent and entity analysis.
    """

    select_id: int
    explanation: str


LLM_SELECT_PROMPT = """
-ROLE-
You are an expert AI assistant for document analysis.

-TASK-
Your task is to analyze the provided inputs and select the single most relevant chapter that answers the user's question. Follow the subsequent instructions precisely to make your selection.


-INSTRUCTIONS-
1.  **Analyze Intent:** Analyze the user's intent from **Input 1**.
2.  **Prioritize Structure:** A chapter's `title` and `path` (from **Input 3**) are the primary selection criteria. A good structural fit is more important than entity matches. (e.g., for an error query, a "Troubleshooting" chapter is best).
3.  **Use Entities as Support:** Use the entity mapping in **Input 2** to evaluate the relevance of a chapter's `contained_entities` (from **Input 3**). Use this as a supporting signal to confirm a structurally relevant choice.
4.  **Select and Justify:** Based on your analysis, choose the single best chapter `id`.
  * **If a suitable chapter is found:** Place its `id` in the `select_id` field and provide a justification in the `explanation` field, prioritizing structure over entities.
  * **If no chapter is suitable:** You **MUST** follow this fallback rule: set `select_id` to `-1` and use the `explanation` to clearly state why no chapter was relevant.

-Output Schema & Format-
Your response MUST be a single, valid JSON object that adheres to the following schema. Do not include any other text, explanation, or markdown formatting like ```json.

### JSON Structure
```json
{{
  "select_id": "<The integer ID of the chosen chapter. MUST be -1 if no chapter is relevant.>",
  "explanation": "<A concise justification for your selection. If id is -1, explain why no chapter was suitable.>"
}}
```
---

-CONTEXT-

Here is the information you must use to make your decision. The numbering corresponds to the inputs referenced in the INSTRUCTIONS section.

**1. User Question (Input 1):**

```
{user_question}
```

**2. Entity Analysis Data (Input 2):**
A single JSON object will be provided here, containing both the original entities from the user's question and their mapping to the document's internal Knowledge Graph (KG).

```json
{entity_analysis_json}
```

**3. Candidate Chapters Data (Input 3):**
A JSON array of candidate chapters will be provided here. Each object represents a chapter and contains the key fields for your analysis:

  * `id`: The unique identifier for the chapter.
  * `title` and `path`: Your **primary criteria** for evaluating the chapter's structural relevance.
  * `contained_entities`: A **supporting field** used to confirm relevance.
  
```json
{candidate_chapters_json}
```
---
Output:
"""


class SecEXPSelection(BaseModel):
    supplementary_ids: List[int]
    explanation: str


LLM_EXPANSION_SELECT_PROMPT = """
-ROLE-
You are an expert AI assistant for document analysis, specializing in expanding search context to ensure high recall.

-TASK-
Your task is to analyze a user's question and a list of remaining document sections. Based on semantic relevance, you will select any additional sections that could be important for answering the question, even if they don't contain direct keyword matches.

-INSTRUCTIONS-
1.  **Analyze User Intent:** Understand the core topic and intent of the user's question provided in **Input 1**.
2.  **Review Primary Selections:** Examine the sections already selected as primary candidates (from **Input 2**). 
    - **Pay attention to the `contained_entities` field.** This tells you which specific topics have already been covered, helping you to select supplementary sections that offer **new, complementary information** rather than redundant details.
    - This is for context only. **Do not re-select any of these primary sections.**
3.  **Evaluate Remaining Sections:** Your main task is to analyze the list of **remaining sections** provided in **Input 3**. For each remaining section, evaluate if its `title` and `path` are semantically relevant to the user's question. This is a step to find potentially relevant context that was missed by keyword/entity matching.
4.  **Select and Justify:**
    * **If you find one or more relevant sections:** Place their integer `id`s in the `supplementary_ids` list. In the `explanation` field, briefly justify why each selected section is relevant based on its title.
    * **If no remaining sections are relevant:** You **MUST** return an empty list for `supplementary_ids` and state in the `explanation` that no further relevant sections were found.

-Output Schema & Format-
Your response MUST be a single, valid JSON object. Do not include any other text or markdown.

### JSON Structure
```json
{{
  "supplementary_ids": "<A list of integer IDs for the chosen supplementary sections. Can be empty.>",
  "explanation": "<A concise justification for your selection(s), or a statement that none were relevant.>"
}}
````

-----

\-CONTEXT-

Here is the information you must use to make your decision:

**1. User Question (Input 1):**

```
{user_question}
```

**2. Primary Candidate Sections (Already Selected, for Context Only):**

```json
{primary_candidates_json}
```

**3. Remaining Sections (Your Selection Pool):**
*Note: This list does not contain entity information. Your selection must be based on the `title` and `path`.*

```json
{remaining_sections_json}
```

-----

Output:
"""


TEXT_RERANKER_PROMPT = """You are a precise document relevance analyzer. Your task is to score a document chunk based on its relevance to a user's query.

The document chunk may be a standard text paragraph, a mathematical formula, a description of a table, or an image caption.

Use a scale from 0.0 (irrelevant) to 1.0 (highly relevant). A high score means the chunk directly answers or provides critical evidence for the query. A low score means it is off-topic.
"""

MM_RERANKER_INSTRUCTION = "Retrieve the most relevant document for the given query."


# Iterative Generation System Prompt and User Prompt
# 269 tokens
ITER_GENERATION_SYS_PROMPT_ORI = """### **Instructions & Output Format**

You are an expert AI assistant specializing in information synthesis. Your primary task is to analyze the provided context (which may include text, data, and images) to give a precise, factual answer to the user's question.

**Your response must adhere to the following rules:**

1.  **Strictly Grounded in Context:** This is the most important rule. You **MUST** base your entire answer *exclusively* on the provided context. Do not use any external knowledge. If the information is not present, you cannot answer.

2.  **Mandatory Two-Part Structure:** Your response must be organized into two distinct sections, in this exact order:
  * **`Analysis:`**
  Start with this literal heading. Briefly explain how the provided context relates to the user's question and critically assess if the information is sufficient. For example, you must point out discrepancies, such as the user asking for "50-shot" data when the context only provides "100-shot" data.

  * **`Final Answer:`**
  Start with this literal heading.
    * If the context is sufficient, provide a direct and concise answer.
    * If the context is insufficient, you **MUST** write "**Not Answerable.**" followed by a brief explanation of why.
"""

ITER_GENERATION_SYS_PROMPT_1 = """### **Instructions & Output Format**

You are an expert AI assistant specializing in information synthesis. Your primary task is to analyze the provided context (which may include text, data, and images) to give a precise, factual answer to the user's question.

**Your response must adhere to the following rules:**

1.  **Strictly Grounded in Context:** This is the most important rule. You **MUST** base your entire answer *exclusively* on the provided context. Do not use any external knowledge. If the information is not present, you cannot answer.

2.  **Mandatory Two-Part Structure:** Your response must be organized into two distinct sections, in this exact order:
  * **`Analysis:`**
  Start with this literal heading. In this section, you must first critically assess the provided context against the user's question and state its relevance. Your assessment **must** fall into one of three categories: **Fully Sufficient**, **Partially Sufficient**, or **Not Relevant**.
    Then, briefly justify your assessment.
    - If **Partially Sufficient**, you must specify which part of the question the context can answer and which part it cannot.
    - If **Not Relevant**, briefly explain why.


  * **`Final Answer:`**
  Start with this literal heading. Your response here depends directly on your conclusion in the `Analysis` section:
    - **If `Analysis` is 'Fully Sufficient':** Provide a direct and concise answer to the entire question.
    - **If `Analysis` is 'Partially Sufficient':** Provide a direct answer for *only the part of the question that can be answered* based on the context. You must also state which information is missing.
    - **If `Analysis` is 'Not Relevant':** You **MUST** write "**Not Answerable.**" and briefly state that the context is not relevant to the question.

"""


# 125 tokens
ITER_GENERATION_USER_PROMPT_1 = """### Inputs for Analysis
You will be provided with the following information to construct your answer:

**1. User's Original Question:**
The question you must answer.

{user_question}

**2. Context for Analysis:**
This is the **only** information you are allowed to use to formulate your answer.

* **Source Text (from Document):**
The primary text excerpts containing the potential answer.

```
{retrieved_content}
```

* **Knowledge Graph Data:**
Key entities and their relationships identified in the source text to aid understanding.

```
{knowledge_graph_subgraph}
```
    
---
Output:
"""

# 99 tokens
VLM_GENERATION_USER_PROMPT_1 = """### Inputs for Analysis

**1. User's Original Question:**
{question}

**2. Context for Analysis:**
Your primary task is to answer the question by analyzing the provided image. The text metadata below is supplementary information to assist your understanding, but your analysis **must prioritize the visual content of the image**.

* **Image Metadata (Supplementary Context):**
{content}

---
Output:
"""

# Synthesis System Prompt
SYNTHESIS_SYS_PROMPT_ORI = """You are an expert AI assistant specializing in synthesizing information.
Your task is to take a user's original question and several pieces of analysis generated by other AI models (based on different document chunks, images, and tables) and combine them into a single, final, comprehensive, and coherent answer.

**Your response must adhere to the following rules:**

1.  **Synthesize, Don't List:** Do not mention "chunks," "parts," or "analysis pieces" in your output. Your goal is to seamlessly integrate the information as if it came from a single, authoritative source.
2.  **Comprehensive Answer:** Your answer should be as complete as possible based on ALL the provided analysis.
3.  **Follow Original Format:** The final synthesized answer should still follow the two-part structure (`Analysis:` and `Final Answer:`).
4.  **Handle Insufficient Information:** After reviewing ALL the provided analyses, if you determine that there is still not enough information to definitively answer the user's question, the `Final Answer:` section **MUST** contain only the words "**Not Answerable.**" followed by a brief explanation of why the collective information was insufficient.
"""

SYNTHESIS_SYS_PROMPT_1 = """
You are an expert AI assistant specializing in synthesizing information.
Your task is to act as a final arbiter, taking a user's original question and several pieces of analysis from other AI models and combining them into a single, final, comprehensive, and coherent answer.

**Your response must adhere to the following rules:**

1. **Filter Non-Informative Inputs:** Before synthesizing, you **MUST** first filter the provided analyses. Completely **ignore** any analysis piece that resulted in "Not Answerable" or indicates that no relevant information was found. Base your synthesis ONLY on the analyses that provided positive findings.

2. **Handle Contradictions vs. Complementary Facts:**
  * **True Contradictions:** If analyses provide directly conflicting facts about the *exact same subject* (e.g., one source states a price is $10, another states it's $12), you **MUST** present both conflicting points in the final answer.
  * **Complementary Facts:** If analyses provide information that seems different but originates from different contexts (e.g., an image shows a *specific* car is red, while a text document lists *available* colors as blue and green), treat this as complementary, not conflicting. Synthesize them to provide a richer, more complete picture. For example: "An image shows a specific instance of the product in red, while the general specifications state that the available colors for order are blue and green."

3. **Aggregate Partial Information:** Carefully evaluate if the combination of all *informative* partial answers is now sufficient to form a complete answer. Your `Analysis` section should reflect this aggregation process.

4. **Strictly Adhere to Final Output Format:** The final synthesized response must follow the two-part structure (`Analysis:` and `Final Answer:`). The content of these sections is dictated by the rules below.

5. **Refined Structure for Final Output:**
  * **`Analysis:` Section:**
    * If the aggregated information is sufficient to answer, briefly explain that the pieces of information combine to form a complete answer.
    * If the information is insufficient, explain *why* it's insufficient and clearly state what information is still missing.
  * **`Final Answer:` Section:**
    * If the information is sufficient, provide the single, synthesized, and comprehensive answer.
    * If the information is still insufficient, the section **MUST** be structured *exactly* as follows, containing only the summary of what *was* found:
      **Not Answerable.**
      * **Summary of Findings:** Briefly summarize all the partial, confirmed information that was successfully gathered.
"""

# Synthesis User Prompt
SYNTHESIS_USER_PROMPT_1 = """### Task: Synthesize a Final Answer

**1. Original User Question:**
{user_question}

**2. Provided Analyses from Different Sources:**
Here are the results from analyzing various pieces of context. You must synthesize these into a single answer.

---
{partial_answers_str}
---

Output:
"""


# GBC Prompts -- Version 2
ITER_GENERATION_SYS_PROMPT = """
Please refer to the following background information to answer the question."""

ITER_GENERATION_USER_PROMPT = """
--- User Question ---
{user_question}

--- Retrieved Documents ---
## Document Data
{retrieved_content}

"""

ITER_GENERATION_GRAPH = """
## Knowledge Graph Data
{knowledge_graph_subgraph}

"""

VLM_GENERATION_USER_PROMPT = """
--- User Question ---
{question}

**Image Metadata (Supplementary Context):**
{content}
"""

SYNTHESIS_SYS_PROMPT = """
You are an expert AI assistant specializing in synthesizing information.
Given the user question, please merge all analyses into a final answer.
"""

SYNTHESIS_USER_PROMPT = """
--- User Question ---
{user_question}

--- Provided Analyses from Different Sources ---
{partial_answers_str}
"""
