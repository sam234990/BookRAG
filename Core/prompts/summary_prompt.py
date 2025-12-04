ORI_NODE_SUMMARY_PROMPT = """You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

Partial Document Text: {node_text}

Directly return the description, do not include any other text.
"""

NODE_SUMMARY_PROMPT = """Condense the content below into a self-contained, informative summary. The summary must capture the key points and be easily understandable on its own.

The content may be text, an image, or a table. If a title is provided, use it to understand the core subject.

---
**Content to Summarize:**
{node_text}
---

Directly output the summary. Your response must begin with the core conclusion, strictly avoiding any introductory phrases that describe the input, such as "The input is a table that shows...".
"""


ORI_SEC_SUMMARY_PROMPT = """Your are an expert in generating descriptions for a document.
You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

Document Structure: {structure}

Directly return the description, do not include any other text.
"""

SEC_SUMMARY_PROMPT = """
You are an expert AI assistant specializing in analyzing technical documents. Your task is to generate a concise, one-sentence summary that captures the core function of a specific document section.

You will be provided with the section's text and a summary of its contents (which includes its subsections and key text). Your generated description must be a synthesis of this information, making it easy to distinguish this section from others.

**Section Text:**
{section_text}

**Summary of Section's Contents:**
{content_summary}

**Your Task:**
Based on the information above, generate a single, descriptive sentence.
- Focus on the section's primary purpose (e.g., "details a process," "introduces a methodology," "presents the results," "compares architectures").
- Be specific. For example, instead of "This section is about graphs," a better summary is "This section details the methodology for constructing a knowledge graph from unstructured text."

**Output:**
Directly return the one-sentence description. Do not include any other text, labels, or preamble.
"""
