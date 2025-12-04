from pydantic import BaseModel
from typing import List


class OutlineExtractionOutput(BaseModel):
    pdf_id: int
    level: int
    parent_id: int


class OutlineExtraction(BaseModel):
    outline: List[OutlineExtractionOutput]


# 2219 tokens
OUTLINE_EXTRACTION_PROMPT = """
You are an expert in document structure analysis. Your task is to generate a structured outline based on a given list of text segments.

Each input segment is provided with a `pdf_id`, `text` (the title content), its page number `page_idx`, and an estimated font size proxy `height`. Your task is to analyze the text content and **infer a consistent, logical hierarchy**.

Please follow these guidelines:

---

### Input Format

Each entry in the input list will have the following structure:

```json
{{
    "pdf_id": 1,
    "text": "Title Text",
    "page_idx": 0,
    "height": 35.5
}}
```

  - `pdf_id`: A unique identifier for the text segment.
  - `text`: The title content.
  - `page_idx`: The 0-indexed page number where the text appears.
  - `height`: A numeric value representing the effective font size. **This is your primary signal for determining the hierarchy.**

---
### Guidelines:

1. **Hierarchical Structure**: Construct a clear, hierarchical outline that reflects the document's logical organization. Use heading levels (`level: 0`, `1`, `2`, etc.) to represent section depth:
    * **Font Size (`height`)**: The `height` field is your primary indicator. Larger `height` values correspond to higher-level headings (e.g., `level: 0` > `level: 1` > `level: 2`).
    * ** Numbering Patterns**: Look for explicit numbering like "1. Introduction", "2.1 Methodology", etc. This is a very reliable signal for hierarchy.
    * ** Page Order**: A section's parent **must** appear on the same or an earlier page (`page_idx`). 
    * **Text Content**: Use semantic meaning as a final check (e.g., "Introduction", "Conclusion", "Abstract" are almost always `level: 1`).
    
2. **Document Title Rule**: You **MUST** identify the single, primary title of the entire document. This title **MUST** be assigned `level: 0`. 
    * It is almost always on the first page (`page_idx: 0`).
    * It typically has the **largest `height`** value in the entire document.
    * There **MUST** be exactly one entry with `level: 0`. All top-level sections (`level: 1`) must be children of this node.

3.  **Parent-Child Relationships**: You **MUST** correctly assign the `parent_id` for every entry:
    * `level: 0` (Root Node): `parent_id` **MUST** be `0`.
    * `level: 1` (Top-Level Sections): `parent_id` **MUST** be the `pdf_id` of the `level: 0` node.
    * `level >= 2` (Sub-Sections): `parent_id` **MUST** be the `pdf_id` of the nearest preceding entry with a `level` that is exactly one less than its own.
    * `level: -1` (Invalid Entries): `parent_id` **MUST** be `-1`.
    * **Critical Constraint**: With the sole exception of the root node, no other valid entry (where `level > 0`) can have a `parent_id` of `0`.
    
4. ** Invalid Entries**: Mark segments as invalid (`"level": -1`, `"parent_id": -1`) if they are not structural section titles. These entries are often caused by layout analysis errors. Common examples include headers, footers, page numbers, standalone phrases, list items, captions for figures and tables, or text extracted from within images. e.g., "Podcast transcripts" in an Academic paper, which is likely a list item under a "Datasets" section

5. **Output Length Must Match Input Length**: You **MUST** generate one output object for **every single** input object. The `outline` list in your JSON output must have the same number of elements as the input list. **Do not skip or omit any entries.**

---

### Output Format (JSON):

You should return only the outline in JSON format, structured as follows:

```json
{{
    "outline": [
        {{"pdf_id": 1, "level": 0, "parent_id": 0}},
        {{"pdf_id": 2, "level": 1, "parent_id": 1}},
        {{"pdf_id": 3, "level": 2, "parent_id": 2}},
        {{"pdf_id": 5, "level": -1, "parent_id": -1}},
    ]
}}
```

---
### Example

### Example Input:

```json
[
    {{"pdf_id": 1, "text": "Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!", "page_idx": 0, "height": 16.5}},
    {{"pdf_id": 4, "text": "Abstract", "page_idx": 0, "height": 13.0}},
    {{"pdf_id": 6, "text": "1 Introduction", "page_idx": 0, "height": 12.0}},
    {{"pdf_id": 13, "text": "2 Related Work", "page_idx": 1, "height": 13.0}},
    {{"pdf_id": 14, "text": "2.1 LLMs for Information Extraction", "page_idx": 1, "height": 13.0}},
    {{"pdf_id": 16, "text": "2.2 Few-shot IE with ICL", "page_idx": 1, "height": 13.0}},
    {{"pdf_id": 20, "text": "3 Large LMs v.s. Small LMs", "page_idx": 1, "height": 14.0}},
    {{"pdf_id": 56, "text": "4 LLMs are Good Few-shot Reranker", "page_idx": 5, "height": 13.0}},
    {{"pdf_id": 100, "text": "Limitations", "page_idx": 9, "height": 13.0}},
    {{"pdf_id": 104, "text": "Acknowlegement", "page_idx": 9, "height": 13.0}},
    {{"pdf_id": 171, "text": "A Datasets", "page_idx": 13, "height": 13.0}},
    {{"pdf_id": 172, "text": "A.1 Full Datasets", "page_idx": 13, "height": 12.0}},
    {{"pdf_id": 176, "text": "Algorithm 1 Greedy Sampling", "page_idx": 13, "height": 13.0}},
    {{"pdf_id": 207, "text": "E Auxiliary Experiments", "page_idx": 16, "height": 14.0}},
    {{"pdf_id": 214, "text": "Instruction0: [empty]", "page_idx": 16, "height": 12.0}},
    {{"pdf_id": 255, "text": "Correct Answer: (c)", "page_idx": 19, "height": 13.0}},
    {{"pdf_id": 277, "text": "G Details on Adaptive Filter-then-rerank Paradigm", "page_idx": 20, "height": 14.0}}
]
```

### Example Output:

```json
{{
    "outline": [
        {{"pdf_id": 1, "level": 0, "parent_id": 0}},
        {{"pdf_id": 4, "level": 1, "parent_id": 1}},
        {{"pdf_id": 6, "level": 1, "parent_id": 1}},
        {{"pdf_id": 13, "level": 1, "parent_id": 1}},
        {{"pdf_id": 14, "level": 2, "parent_id": 13}},
        {{"pdf_id": 16, "level": 2, "parent_id": 13}},
        {{"pdf_id": 20, "level": 1, "parent_id": 1}},
        {{"pdf_id": 56, "level": 1, "parent_id": 1}},
        {{"pdf_id": 100, "level": 1, "parent_id": 1}},
        {{"pdf_id": 104, "level": 1, "parent_id": 1}},
        {{"pdf_id": 171, "level": 1, "parent_id": 1}},
        {{"pdf_id": 172, "level": 2, "parent_id": 171}},
        {{"pdf_id": 176, "level": -1, "parent_id": -1}},
        {{"pdf_id": 207, "level": 1, "parent_id": 1}},
        {{"pdf_id": 214, "level": -1, "parent_id": -1}},
        {{"pdf_id": 255, "level": -1, "parent_id": -1}},
        {{"pdf_id": 277, "level": 1, "parent_id": 1}}
    ]
}}
```

### Explanation:

  - `pdf_id: 1` is `level: 0` because it is on page 0 and has the largest `height` (16.5). All other valid sections are its descendants.
  - Sections with major numbering ("1", "2", "A", etc.) or standard names ("Abstract", "Limitations") are identified as `level: 1`.
  - Sections with sub-numbering ("2.1", "A.1", etc.) become `level: 2` children of their corresponding `level: 1` parent.
  - `pdf_id: 176`, `214`, and `255` are marked `level: -1` because their text content ("Algorithm 1...", "Instruction0...", "Correct Answer...") indicates they are captions or content elements, not structural section titles.

---
Now, based on the following input text segments, generate the corresponding hierarchical outline, ensuring you process every single entry from the input.

{json_title}

Output:
"""
