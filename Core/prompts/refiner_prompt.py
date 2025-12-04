from pydantic import BaseModel
from typing import List, Optional


class MergeJudgment(BaseModel):
    # The `pdf_id_1` from the input if the tables should be merged, otherwise -1
    merged_id: int
    explanation: str


class MergeJudgmentsResponse(BaseModel):
    """The final response object containing a list of all judgments."""

    judgments: List[MergeJudgment]


TABLE_MERGE_PROMPT = """
You are an expert AI assistant specializing in document structure analysis. Your task is to determine if pairs of tables, extracted from a PDF and provided in HTML format, should be merged. A common scenario is a single logical table being split across two separate pages during PDF processing.

You will receive a list of table pairs to analyze. For each pair, you must decide if the second table is a direct continuation of the first.

**Input:**
You will be provided with a JSON list of objects. Each object represents a pair of tables to be compared and contains the following keys:
- `pdf_id_1`: An integer identifier for the first table.
- `table_1_html`: An HTML string for the first table. This table might have a `<caption>` and footnotes.
- `table_2_html`: An HTML string for the second table. This table is potentially a continuation and usually lacks a `<caption>`.
- `caption`: An optional string containing the caption associated with `table_1_html`. This may be an empty string if no caption exists.

**Decision Criteria:**
You can assume both tables provided have an identical column count. Base your decision on the following criteria:

1.  Your decision must depend on logical continuity. Merge the tables only if the data rows in `table_2_html` are a direct and seamless continuation of the data rows in `table_1_html`. Analyze corresponding columns for consistent data formats, sequential patterns (like dates or item numbers), and overall topical cohesion.

2. Evaluate the first row of each table carefully. If the first row of `table_1_html` acts as a header (describing the columns) and the first row of `table_2_html` is an identical repetition of it, this strongly supports a merge. However, if both tables appear to contain only data from the first row onwards, your decision should rely solely on the continuous flow of that data.

3. Do NOT merge if there is a clear break in the topic or data logic between the tables. A sudden change in the type of content or subject matter between the last row of `table_1_html` and the first data row of `table_2_html` is the strongest reason to keep them separate.

4. Use the provided `caption` as high-level context. The content within both `table_1_html` and `table_2_html` must be consistent with the topic described in the caption. 

**Output Format:**

Your response MUST be a single, valid JSON object. This object must contain a single key, "judgments", whose value is a JSON array (a list).


You must provide a judgment for every single pair in the input. The "judgments" list must never be empty if the input list contains table pairs.

Each judgment object within the list must conform to the following structure:


```json
{{
    "judgments": [
        {{
            "merged_id": "<Integer>",
            "explanation": "<String>"
        }}
    ]
}}
```

Field Descriptions:
- `merged_id`: The `pdf_id_1` from the input if the tables should be merged, otherwise `-1`.
- `explanation`: A brief, one-sentence rationale for your merge or no-merge decision.

Process all the pairs provided in the input and return a single JSON object with the list of results.

---
**Example of Usage**

**LLM Input**

Here is a sample input containing three pairs of tables for evaluation, each representing a different scenario.

```JSON
[
    {{
        "pdf_id_1": 101,
        "table_1_html": "<table><tbody><tr><td>Region</td><td>Q1 Sales</td><td>Q2 Sales</td></tr><tr><td>East</td><td>1.2M</td><td>1.4M</td></tr><tr><td>North</td><td>0.9M</td><td>0.8M</td></tr></tbody></table>",
        "table_2_html": "<table><tbody><tr><td>West</td><td>2.1M</td><td>2.3M</td></tr><tr><td>South</td><td>1.5M</td><td>1.7M</td></tr></tbody></table>",
        "caption": "Table 1. Quarterly Regional Performance"
    }},
    {{
        "pdf_id_1": 205,
        "table_1_html": "<table><tbody><tr><td>Part #100-A</td><td>Ready</td></tr><tr><td>Part #100-B</td><td>Ready</td></tr></tbody></table>",
        "table_2_html": "<table><tbody><tr><td>Part #100-C</td><td>In Progress</td></tr><tr><td>Part #100-D</td><td>Ready</td></tr></tbody></table>",
        "caption": "Appendix A: Component List (continued)"
    }},
    {{
        "pdf_id_1": 310,
        "table_1_html": "<table><tbody><tr><td>Employee ID</td><td>Name</td><td>Role</td></tr><tr><td>E-045</td><td>Alice</td><td>Sr. Developer</td></tr><tr><td>E-046</td><td>Bob</td><td>Jr. Developer</td></tr></tbody></table>",
        "table_2_html": "<table><tbody><tr><td>A-001</td><td>Laptop</td><td>Dell XPS 15</td></tr><tr><td>A-002</td><td>Monitor</td><td>Dell 27 Inch</td></tr></tbody></table>",
        "caption": "Table 5. Engineering Department Roster"
    }}
]
```

**Expected JSON Output**

Based on the input, the LLM should generate the following JSON list.

```JSON
{{
    "judgments": [
        {{
            "merged_id": 101,
            "explanation": "Merged as Table 2's data logically follows the schema set by Table 1's first row."
        }},
        {{
            "merged_id": 205,
            "explanation": "Merged due to a clear sequential pattern in the content, indicating continuity without any headers."
        }},
        {{
            "merged_id": -1,
            "explanation": "Not merged as Table 2's content (assets) is inconsistent with the topic of Table 1 (employees)."
        }}
    ]
}}
```

**Explanation of the Output**

1. First Case (Merge with Header): The first pair is merged because the first row of `table_1_html` acts as a header, and the data in `table_2_html` perfectly continues the data pattern and topic established in `table_1_html`.
2. Second Case (Merge without Header): The second pair is merged because the content itself shows a clear and logical sequence (part numbers `...-B` followed by `...-C`), which is the strongest indicator of a table split across pages, even without any headers.
3. Third Case (Do Not Merge): The third pair is not merged because the content of `table_2_html` (listing assets like "Laptop") is completely unrelated to the topic of `table_1_html` (listing employees), despite having the same number of columns.

---

Now, based on the provided input table pairs, generate a single JSON object that contains the list of your judgments.

{json_pairs}

Output:"""


class StitchingJudgment(BaseModel):
    stitched_pdf_ids: List[int]
    explanation: str


class StitchingJudgmentsResponse(BaseModel):
    judgments: List[StitchingJudgment]


TEXT_MERGE_PROMPT = """
You are an expert AI assistant specializing in document analysis and natural language continuity. Your primary task is to analyze an incomplete text fragment and determine which, if any, candidate fragments from a list should be stitched onto it to form a coherent whole.

**Your core task is to find the single best continuation. In almost all cases, you will select exactly ONE candidate or NONE. Multiple selections are ONLY permitted for tasks involving mathematical formulas.**

**Input**

You will receive a JSON list of stitching tasks to evaluate. Each object in the list represents a single task to be judged and contains two keys: `incomplete_text` and `candidate_list`.

* **`incomplete_text`**: A string containing the starting piece of text that needs a continuation.
* **`candidate_list`**: A list of potential fragments to append. Each candidate in this list is an object with two keys:
    * `pdf_id`: An integer identifier for the candidate fragment.
    * `text`: The content of the candidate fragment.

---

**Decision Criteria**

Follow this step-by-step process to make your decision for each task:

1.  **Step 1: Check for Completeness.** First, determine if the `incomplete_text` is already a complete sentence or thought. If it is, and no candidate provides a direct, seamless continuation (e.g., they all start new paragraphs), your decision is to select none. In this case, the `stitched_pdf_ids` field **must be `[-1]`**.

2.  **Step 2: Identify ALL Plausible Continuations.** If the text is incomplete, review every candidate in the `candidate_list`. Identify all candidates that provide a grammatically correct and logically coherent continuation. Pay special attention to candidates that complete broken words or mathematical formulas (`...y = \\alpha x +` or `... \\sum_{{i=1}}`).

3.  **Step 3: Apply the Final Selection Rule.** Now, based on the plausible candidates you identified in Step 2, make your final selection:
    * **For ALL non-formula tasks:** You **MUST** select **ONLY ONE** candidate. Even if multiple candidates seem grammatically correct, choose the single one that creates the most direct and natural continuation.
    * **For formula tasks ONLY:** This is the **exclusive** exception where you may select multiple candidates. Select the candidate that completes the formula **AND** any candidate that immediately explains it. The formula part's `pdf_id` **MUST** be listed first.

---

**Output Format**

Your response MUST be a single, valid JSON array (a list), where the order of objects perfectly matches the input task order. Each object in the list must have the following structure:

```json
{{
    "judgments": [
        {{
            "stitched_pdf_ids": ["<List_of_Integers>"],
            "explanation": "<String>"
        }}
    ]
}}
```

**Field Descriptions:**

* **`stitched_pdf_ids`**: A list of the selected candidate `pdf_ids`. This list should contain a single ID, multiple IDs (for formula tasks only), or `[-1]` if no selection is made.

* **`explanation`**: A brief, one-sentence rationale explaining your selection.

---

**Example of Usage**

**LLM Input**
Here is a sample input containing three different stitching tasks for evaluation

```json
[
    {{
        "incomplete_text": "The experiment concluded successfully.",
        "candidate_list": [
            {{ "pdf_id": 11, "text": "In the next chapter, we will discuss the implications." }}
        ]
    }},
    {{
        "incomplete_text": "The new framework supports inter-",
        "candidate_list": [
            {{ "pdf_id": 21, "text": "national collaboration between research teams." }},
            {{ "pdf_id": 22, "text": "This approach has several advantages over previous methods." }}
        ]
    }},
    {{
        "incomplete_text": "We use a two-layer MLP head that inputs contextual text and image and outputs the binary aligned/unaligned labels with a binary cross-entropy loss:",
        "candidate_list": [
            {{ 
                "pdf_id": 31, 
                "text": "L_{{WPA}}(\\theta) = - \\frac{{1}}{{|L-L'|}} \\sum_{{\\ell \\in L-L'}} [z_{{\\ell}} \\log p_{{\\theta}}(z_{{\\ell}} | X^{{M'}}, Y^{{L'}}) + (1-z_{{\\ell}}) \\log(1 - p_{{\\theta}}(z_{{\\ell}} | X^{{M'}}, Y^{{L'}}))]" 
            }},
            {{ 
                "pdf_id": 32, 
                "text": "where |L-L'| is the number of unmasked text tokens, z_{{\\ell}} is the binary label of language token in the {{\\ell}} position." 
            }},
            {{ 
                "pdf_id": 33, 
                "text": "The network architecture of LayoutLMv3 follows that of LayoutLM [54] and LayoutLMv2 [56] for a fair comparison."
            }}
        ]
  }}
]
```

**Expected JSON Output**

Based on the input, the LLM should generate the following JSON list.

```json
{{
    "judgments": [
        {{
            "stitched_pdf_ids": [-1],
            "explanation": "The initial fragment is a complete sentence and no candidate is a direct continuation."
        }},
        {{
            "stitched_pdf_ids": [21],
            "explanation": "Stitched to complete the split word 'inter-national' and finish the sentence."
        }},
        {{
            "stitched_pdf_ids": [31, 32],
            "explanation": "Selected the formula (31) and its direct explanation (32) as they form a single logical unit following the introductory text."
        }}
    ]
}}
```

**Explanation of the Output**
1. First Case: No stitching occurs because the initial fragment is already a complete sentence and the candidate starts a new, separate thought.

2. Second Case: Only candidate 21 is selected because it correctly completes the split word "international"; as this is a non-formula case, only one candidate is permitted.

3. Third Case: Candidate 31 is selected because it provides the mathematical formula directly following the introductory text, and candidate 32 is selected as it provides the immediate explanation of the terms used in that formula, forming a complete logical unit.

---
Now, based on the provided stitching tasks, generate a single JSON object that contains the list of your judgments

{json_text}

Output:"""
