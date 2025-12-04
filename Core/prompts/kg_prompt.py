# -*- coding: utf-8 -*-
from typing import List
from pydantic import BaseModel, Field


class ExtractEntity(BaseModel):
    entity_name: str  # Primary key for entity
    entity_type: str = Field(default="")  # Entity type
    description: str = Field(default="")  # The description of this entity


class ExtractRelationship(BaseModel):
    src_entity_name: str  # Name of the entity on the left side of the edge
    tgt_entity_name: str  # Name of the entity on the right side of the edge
    weight: float = Field(
        default=1.0
    )  # Weight of the edge, used in GraphRAG and LightRAG
    description: str = Field(
        default=""
    )  # Description of the edge, used in GraphRAG and LightRAG


class ExtractionResult(BaseModel):
    entities: List[ExtractEntity]
    relationships: List[ExtractRelationship]


class FormulaEntity(BaseModel):
    entity_name: str  # Primary key for entity
    description: str = Field(default="")  # The description of this entity


class FormulaExtractionResult(BaseModel):
    entities: List[FormulaEntity]


class EntityExtractionResult(BaseModel):
    entities: List[ExtractEntity]


# This file is part of the Knowledge Graph Prompting project.
# Our KG construction is build on the JayLZhou/GraphRAG projects.
# Please refer to the following references:
# 1. https://github.com/JayLZhou/GraphRAG
# 2. https://github.com/gusye1234/nano-graphrag
# 3. https://github.com/HKUDS/LightRAG


DEFAULT_ENTITY_TYPES = [
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

# DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

# 1968 tokens
ENTITY_EXTRACTION = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: A brief, one-sentence description summarizing the entity's role or attributes based *only* on the provided text. If the text offers limited information, keep the description simple (e.g., "A person mentioned in the text."). Do not add any external knowledge.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2.  From the entities identified in Step 1, identify all pairs of (`source_entity`, `target_entity`) that are *clearly and directly related* to each other based **only** on the provided text.
* **Crucial Rule**: The `source_entity` and `target_entity` for any relationship **must exactly match** an `entity_name` from the list you generated in Step 1. Do not create new entities or list relationships for entities not found in Step 1.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: A brief explanation for the relationship, using evidence found directly in the text.
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [PERSON, TECHNOLOGY, MISSION, ORGANIZATION, LOCATION]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"LOCATION"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"MISSION"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"ORGANIZATION"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [PERSON, ROLE, TECHNOLOGY, ORGANIZATION, EVENT, LOCATION, CONCEPT]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"PERSON"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"PERSON"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"CONCEPT"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"CONCEPT"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"EVENT"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"EVENT"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
Example 3:

Entity_types: [PERSON, ORGANIZATION]
Text:
Yubo Ma $^{{1}}$ , Yixin Cao $^{{2}}$ , YongChing Hong $^{{1}}$ , Aixin Sun $^{{1}}$
#############
Output:
("entity"{tuple_delimiter}"Yubo Ma"{tuple_delimiter}"PERSON"{tuple_delimiter}"Yubo Ma is a person listed as an author of the document."){record_delimiter}
("entity"{tuple_delimiter}"Yixin Cao"{tuple_delimiter}"PERSON"{tuple_delimiter}"Yixin Cao is a person listed as an author of the document."){record_delimiter}
("entity"{tuple_delimiter}"YongChing Hong"{tuple_delimiter}"PERSON"{tuple_delimiter}"YongChing Hong is a person listed as an author of the document."){record_delimiter}
("entity"{tuple_delimiter}"Aixin Sun"{tuple_delimiter}"PERSON"{tuple_delimiter}"Aixin Sun is a person listed as an author of the document."){record_delimiter}
("relationship"{tuple_delimiter}"Yubo Ma"{tuple_delimiter}"Yixin Cao"{tuple_delimiter}"Listed as co-authors on the same document."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Yubo Ma"{tuple_delimiter}"Aixin Sun"{tuple_delimiter}"Listed as co-authors on the same document."{tuple_delimiter}8){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


ENTITY_CONTINUE_EXTRACTION = """MANY entities were missed in the last extraction.  Add them below using the same format:"""


ENTITY_IF_LOOP_EXTRACTION = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added."""


# The following PROMPTs are designed for the specifial layout in document

# 1646 tokens
TABLE_ENTITY_EXTRACTION = """
-Goal-
Act as a sophisticated information extraction system. Your task is to analyze a JSON object containing an HTML table and a textual description. You must identify all relevant entities, formatting the output as a single, valid JSON object that adheres to the provided Pydantic schema.

---
-Steps-

1.  **Analyze the Input:** Carefully examine the entire input JSON, which includes an `html_table` and a `description`. The `description` provides crucial context for interpreting the table's data.

2.  **Identify Entities:**
    * **Crucial First Step: The Table Itself**: You **MUST** identify the table as the very first entity. Its `entity_type` must be `TABLE` and it **must** be the first object in the final `entities` list. **This is a non-negotiable rule.** Use its title or caption as its `entity_name` and create a description summarizing its content and purpose.
    * **For Column Headers:** You **MUST** extract each column header (e.g., 'Roberta', 'T5') as a distinct entity. Use the table's overall context to determine its `entity_type`.
    * **For Row Headers:** You **MUST** extract each row header (e.g., 'FewNERD (NER)') as a distinct entity. Use the table's overall context to determine its `entity_type`.
    * **From the Description:** Extract any additional relevant entities mentioned in the `description` text.
    * **Create Entity Objects:** For each unique entity found, create a JSON object following the `Entity` structure. The `description` for each entity should be a summary synthesized **strictly from its role and associated data within the provided `html_table` and `description` text only. Do not use any external knowledge.**

3.  **Construct the Final Output:** Combine all identified entities and relationships into the final JSON object, ensuring it strictly follows the specified output format.

---
-Input JSON Format-
{{
  "html_table": "An HTML table as a string.",
  "description": "A textual description providing context for the table."
}}

---
-Output Format-

### 1. General Instruction
Your response MUST be a single, valid JSON object with a single root key "entities".

### 2. JSON Structure
```json
{{
  "entities": [
    {{
      "entity_name": "<String>",
      "entity_type": "<String>",
      "description": "<String>"
    }}
  ]
}}

### 3. Field Descriptions

#### For an `Entity` object:
* **`entity_name`** (String): The primary name of the identified entity, capitalized.
* **`entity_type`** (String): The category of the entity. It MUST be one of the following types: {entity_types}
* **`description`** (String): A brief, comprehensive description of the entity's attributes and role in the text.

---
-Example-

-Input-
```json
{{
  "html_table": "<table><tr><td>Dataset (Task)</td><td>Roberta</td><td>T5</td><td>LLaMA</td><td>CODEX</td></tr><tr><td>FewNERD (NER)</td><td>2.8</td><td>39.4</td><td>1135.4</td><td>179.4</td></tr><tr><td>TACREV (RE)</td><td>1.4</td><td>45.6</td><td>1144.9</td><td>151.6</td></tr><tr><td>ACE05 (ED)</td><td>6.6</td><td>62.5</td><td>733.4</td><td>171.7</td></tr></table>",
  "description": "Table 1: The inference seconds over 500 sentences (run on single V100 GPU). Here LLaMA is extremely slow since we set batch size as 1 due to memory limit."
}}
```

-Output-
```json
{{
  "entities": [
    {{
      "entity_name": "Table 1",
      "entity_type": "TABLE",
      "description": "A table comparing the inference speeds (in seconds) of four models (Roberta, T5, LLaMA, CODEX) across three datasets, run on a single V100 GPU."
    }},
    {{
      "entity_name": "Roberta",
      "entity_type": "MODEL_OR_ARCHITECTURE",
      "description": "A model whose inference time was measured on the FewNERD (NER), TACREV (RE), and ACE05 (ED) datasets, yielding results of 2.8, 1.4, and 6.6 seconds respectively."
    }},
    {{
      "entity_name": "T5",
      "entity_type": "MODEL_OR_ARCHITECTURE",
      "description": "A model whose inference time was measured on the FewNERD (NER), TACREV (RE), and ACE05 (ED) datasets, yielding results of 39.4, 45.6, and 62.5 seconds respectively."
    }},
    {{
      "entity_name": "LLaMA",
      "entity_type": "MODEL_OR_ARCHITECTURE",
      "description": "A model whose inference time was measured on three datasets. It was noted as being 'extremely slow' because its test was run with a batch size of 1 due to a memory limit."
    }},
    {{
      "entity_name": "CODEX",
      "entity_type": "MODEL_OR_ARCHITECTURE",
      "description": "A model whose inference time was measured on the FewNERD (NER), TACREV (RE), and ACE05 (ED) datasets, yielding results of 179.4, 151.6, and 171.7 seconds respectively."
    }},
    {{
      "entity_name": "FewNERD (NER)",
      "entity_type": "DATASET_OR_CORPUS",
      "description": "A dataset and task used as a benchmark to measure the inference speeds (in seconds) of the Roberta, T5, LLaMA, and CODEX models."
    }},
    {{
      "entity_name": "TACREV (RE)",
      "entity_type": "DATASET_OR_CORPUS",
      "description": "A dataset and task used as a benchmark to measure the inference speeds (in seconds) of four different models."
    }},
    {{
      "entity_name": "ACE05 (ED)",
      "entity_type": "DATASET_OR_CORPUS",
      "description": "A dataset and task used as a benchmark to measure the inference speeds (in seconds) of four different models."
    }},
    {{
      "entity_name": "V100 GPU",
      "entity_type": "HARDWARE",
      "description": "The specific hardware ('single V100 GPU') on which the inference tests for all models were conducted, as stated in the description."
    }},
    {{
      "entity_name": "Inference Seconds",
      "entity_type": "EVALUATION_METRIC",
      "description": "The unit of measurement for the experiment, representing the time taken to process 500 sentences."
    }},
    {{
      "entity_name": "Batch Size",
      "entity_type": "PARAMETER_OR_VARIABLE",
      "description": "An experimental parameter that was set to 1 for the LLaMA model specifically, due to a memory limit."
    }}
  ]
}}
```

---
-Task Execution-

Analyze the following input data using the allowed entity types provided. Your response must be the complete JSON object and nothing else.

- Allowed Entity Types -
{entity_types}

- Input to Process -
{input_json}

- JSON Output -
"""

# 1089 tokens
TABLE_DESCRIPTION_EXTRACTION = """
-Goal-
Act as a precise Information Extraction system. Your task is to analyze a table's description and its column headers to extract all relevant entities.

---
-Instructions-
1.  **Analyze Both Inputs:** You will receive a JSON object containing two keys:
    * `"description"`: A text providing the overall context of the table.
    * `"column_headers"`: A list of strings representing the headers of the table columns.
    You must carefully analyze **both** sources of information.
    
2.  **Identify the Primary Table Entity:** Your most important first step is to create an entity for the table itself.
  * Its `entity_type` **MUST** be `TABLE`.
  * It **MUST** be the very first object in the final `entities` list.
  * Use the title from the `description` (e.g., "Table 1") as its `entity_name`.

3.  **Synthesize Entities from Headers:** This is a mandatory step. You must process every non-empty string in the `column_headers` list. Your task is to **synthesize**, not just split, the hierarchical parts into a single, natural entity name. For example, from `"ACE (ED) > 10-shot"`, extract a single entity named `"10-shot ACE (ED)"`.

4.  **Extract Remaining Entities from Description:** After handling the table and column headers, scan the `description` text again to find any additional relevant entities that have not yet been extracted.

5.  **Format the Final Output:** Your response **must be a single, valid JSON object** with a single root key named `entities`.

---
-Output JSON Schema-
```json
{{
  "entities": [
    {{
      "entity_name": "<String>",
      "entity_type": "<String>",
      "description": "<String>"
    }}
  ]
}}

### Field Descriptions
#### For an `Entity` object:
* **`entity_name`** (String): The primary name of the identified entity, capitalized.
* **`entity_type`** (String): The category of the entity. It MUST be one of the following types: {entity_types}
* **`description`** (String): A brief, comprehensive description of the entity's attributes and role in the text.

-----

- Example -

**Input:**
```json
{{
  "description": "Table 1: The inference seconds over 500 sentences (run on single V100 GPU). Here LLaMA is extremely slow since we set batch size as 1 due to memory limit.",
  "column_headers": [
    "Dataset (Task)",
    "Roberta",
    "T5",
    "LLaMA",
    "CODEX"
  ]
}}
```

**Output:**

```json
{{
  "entities": [
    {{
      "entity_name": "Table 1",
      "entity_type": "TABLE",
      "description": "A table comparing the inference speeds (in seconds) of four models (Roberta, T5, LLaMA, CODEX) across several datasets, run on a single V100 GPU."
    }},
    {{
      "entity_name": "Dataset (Task)",
      "entity_type": "PARAMETER_OR_VARIABLE",
      "description": "The primary categorical column of the table, listing the benchmarks or datasets on which models were tested."
    }},
    {{
      "entity_name": "Roberta",
      "entity_type": "MODEL_OR_ARCHITECTURE",
      "description": "A model whose inference speed is evaluated in the table, identified as a column header."
    }},
    {{
      "entity_name": "T5",
      "entity_type": "MODEL_OR_ARCHITECTURE",
      "description": "A model whose inference speed is evaluated in the table, identified as a column header."
    }},
    {{
      "entity_name": "LLaMA",
      "entity_type": "MODEL_OR_ARCHITECTURE",
      "description": "A model whose inference speed is evaluated. The description notes it was 'extremely slow' due to a batch size of 1."
    }},
    {{
      "entity_name": "CODEX",
      "entity_type": "MODEL_OR_ARCHITECTURE",
      "description": "A model whose inference speed is evaluated in the table, identified as a column header."
    }},
    {{
      "entity_name": "Inference Seconds",
      "entity_type": "EVALUATION_METRIC",
      "description": "The unit of measurement for the experiment, representing the time taken to process 500 sentences, as stated in the description."
    }},
    {{
      "entity_name": "V100 GPU",
      "entity_type": "HARDWARE",
      "description": "The specific hardware ('single V100 GPU') on which the inference tests were conducted."
    }},
    {{
      "entity_name": "Batch Size",
      "entity_type": "PARAMETER_OR_VARIABLE",
      "description": "An experimental parameter mentioned in the description, which was set to 1 for the LLaMA model due to a memory limit."
    }}
  ]
}}
```

-----
-Task Execution-

Analyze the following input data using the allowed entity types provided. Your response must be the complete JSON object and nothing else.

- Allowed Entity Types -
{entity_types}

- Input Text -
{input_json}

- JSON Output -
"""


#  949 tokens
TABLE_BODY_EXTRACTION = """
-Goal-
Act as a precise, row-by-row information extraction engine. Your task is to analyze a batch of **semi-structured row data strings**. You must **correlate** each row's data with the provided **`column_headers`**, using the main **`description`** for overall context, to extract all relevant entities found strictly within the row data.

---
-Instructions-

1.  **Use Context, Don't Extract From It:** First, carefully study the `description` and `column_headers`. Use this information **strictly as context** to understand the row data's meaning and to assign correct entity types. **Do not** extract entities that appear only in the `description` or `column_headers`.

2.  **Correlate and Extract from Each Row:** Iterate through each `row_string` in the `rows_batch` list. For each string:
    * It represents a single table row, with cells likely separated by a delimiter (e.g., '|').
    * You must mentally map the data cells in the string to their corresponding header in the `column_headers` list based on their order.
    * Extract entities from the row's **categorical data** (e.g., names of models, methods, or groups like 'CODEX' or 'SLM + LLM'). You generally **should NOT** extract entities from purely **numerical data** (e.g., '53.8(0.5)').
    * **Minimum Extraction Mandate:** Each row string describes a primary subject. You **must** aim to extract **at least one main entity** from each row, representing the subject of that row (e.g., the specific model or method being evaluated).

3.  **Consolidate and Format:** Collect all unique entities found across all rows into a single list. Your final response **must be a single, valid JSON object** with a single root key named `entities`.

4.  **Strict Extraction Boundaries:** This is the final and most important rule.
    * All extracted entities **MUST** originate from the data within the `rows_batch` strings.
    * Therefore, you **MUST NOT** extract entities that appear *only* in the `description` or `column_headers`. These are for context only.
    * You also **MUST NOT** create an entity for the table itself (e.g., 'Table 1').

---
-Output JSON Schema-
```json
{{
  "entities": [
    {{
      "entity_name": "<String>",
      "entity_type": "<String>",
      "description": "<String>"
    }}
  ]
}}
```

### Field Descriptions
#### For an `Entity` object:
* **`entity_name`** (String): The primary name of the identified entity, capitalized.
* **`entity_type`** (String): The category of the entity. It MUST be one of the following types: {entity_types}
* **`description`** (String): A brief, comprehensive description of the entity's attributes and role in the text.
-----

- Example -

**Input:**

```json
{{
  "description": "Table 1: The inference seconds over 500 sentences (run on single V100 GPU).",
  "column_headers": [
    "Dataset (Task)",
    "Roberta",
    "T5",
    "LLaMA",
    "CODEX"
  ]
  "rows_batch": [
    "FewNERD (NER)|2.8|39.4|1135.4|179.4",
    "TACREV (RE)|1.4|45.6|1144.9|151.6",
    "ACE05 (ED)|6.6|62.5|733.4|171.7"
  ]
}}
```

**Output:**

```json
{{
  "entities": [
    {{
      "entity_name": "FewNERD (NER)",
      "entity_type": "DATASET_OR_CORPUS",
      "description": "A Named Entity Recognition dataset used as a benchmark to measure model inference speeds."
    }},
    {{
      "entity_name": "TACREV (RE)",
      "entity_type": "DATASET_OR_CORPUS",
      "description": "A Relation Extraction dataset used as a benchmark to measure model inference speeds."
    }},
    {{
      "entity_name": "ACE05 (ED)",
      "entity_type": "DATASET_OR_CORPUS",
      "description": "An Entity Disambiguation dataset used as a benchmark to measure model inference speeds."
    }}
  ]
}}
```
-----
-Task Execution-

Analyze the following input data using the allowed entity types provided. Your response must be the complete JSON object and nothing else.

- Allowed Entity Types -
{entity_types}

- Input to Process -
{input_json}

- JSON Output -
"""


# 1197 tokens
IMAGE_ENTITY_EXTRACTION = """
-Goal-
Act as an expert AI system for visual and semantic analysis. Your primary task is to **comprehensively** analyze a given image and an accompanying textual description. You must identify **all possible** relevant entities, formatting the output as a single, valid JSON object that adheres to the provided Pydantic schema.

---
-Steps-

1.  **Analyze Full Context:** Carefully examine both the provided image and the textual `description`. The `description` often provides crucial context, names, or other details that are not visually obvious.

2.  **Identify Entities:**
    * **Crucial First Step: The Image Itself**. You **MUST** identify the entire image as the very first entity. Its `entity_type` **must** be `IMAGE` and it **must** be the first object in the final `entities` list. **This is a non-negotiable rule.** If the description provides a title or figure number (e.g., 'Figure 2'), use that as the `entity_name`; otherwise, use the first few words of the description as the entity name. If the description is empty, use 'The Image' as the entity name.
    * **From Visual Objects:** Identify distinct physical objects, people, animals, and general locations shown in the image.
    * **From Text Within the Image (High Priority):** Pay attention to any text inside the image, such as labels, titles, annotations, or data points in diagrams and flowcharts. **This text is a critical source of entities.** You **MUST** treat every distinct label, title, or significant term as a candidate for an entity.
    * **From the Description:** Extract any additional relevant entities mentioned in the text that might not be visible or clearly identifiable in the image.
    * **Principle of Comprehensiveness:** When in doubt, it is better to extract a potential entity than to omit it. **Be thorough and aim for maximum detail.**
    * **Create Entity Objects:** For each unique entity found, create a JSON object following the `ExtractEntity` structure.

3.  **Construct the Final Output:** Combine all identified entities into a single JSON object with the root keys "entities", ensuring it strictly follows the specified output format.

---
-Output Format-

### 1. General Instruction
Your response MUST be a single, valid JSON object with the root key "entities".

### 2. JSON Structure
```json
{{
  "entities": [
    {{
      "entity_name": "<String>",
      "entity_type": "<String>",
      "description": "<String>"
    }}
  ]
}}
```

### 3. Field Descriptions
* **`entity_name`** (String): The name of the entity.
* **`entity_type`** (String): The category of the entity. It MUST be one of the following types: {entity_types}
* **`description`** (String): A brief, comprehensive summary of the entity's role and attributes.

---
-Example-

### Conceptual Input:
* **Image:** A simple diagram showing three boxes. "User PC" has an arrow pointing to "Web Server", which has an arrow pointing to "Database".
* **Description:** "Figure 1: A diagram of a basic three-tier web application architecture."

### Correct Output:
```json
{{
  "entities": [
    {{
      "entity_name": "Figure 1",
      "entity_type": "IMAGE",
      "description": "A diagram illustrating a basic three-tier web application architecture, showing the data flow between components."
    }},
    {{
      "entity_name": "User PC",
      "entity_type": "SYSTEM_COMPONENT",
      "description": "The client-tier component in the architecture diagram, representing the end-user's machine."
    }},
    {{
      "entity_name": "Web Server",
      "entity_type": "SYSTEM_COMPONENT",
      "description": "The application-tier component that processes requests from the User PC."
    }},
    {{
      "entity_name": "Database",
      "entity_type": "SYSTEM_COMPONENT",
      "description": "The data-tier component that stores and retrieves data for the Web Server."
    }}
  ]
}}
```

---
-Task-

Now, analyze the provided image and its description. Generate the JSON object according to all the instructions above.

- Allowed Entity Types -
{entity_types}

- Image description -
{image_description}

-JSON Output-
"""

# 530 tokens
EQUATION_ENTITY_EXTRACTION = """-Goal-
You are a specialized data extraction component. Your task is to analyze a given LaTeX mathematical equation and format specific pieces of information into a JSON object.

-Context-
You will be given a single mathematical equation in LaTeX format. You must adhere strictly to the output instructions.

---
-Instructions & Output Schema-

### 1. General Instruction
Your response MUST be a single, valid JSON object with a single root key named "entities". Do not include any extra text, explanations, or markdown formatting.

### 2. JSON Structure & Field Descriptions

The JSON object must conform to the following structure and rules:

```json
{{
  "entities": [
    {{
      "entity_name": "<String>",
      "description": "<String>"
    }}
  ]
}}
```

Inside the `entities` object:
* **`entity_name`** (String): You must create a unique and human-readable name for the equation.
    * If the equation has a tag number like `\\tag{{1}}`, the name MUST be exactly "Formula (1)".
    * If the equation does not have a tag number, create a name by using the first 15 characters of the LaTeX code, like "Formula [first 15 chars...]".
* **`description`** (String): You must provide a brief, one-sentence summary of the equation's role, followed by its complete and unmodified LaTeX code. The summary should be based on the equation's structure (e.g., "An equation defining the variable P..."). The final string must follow the format: `"<brief summary> LaTeX: <full_latex_code>"`.

---
-Example-

### Input:
```latex
\\mathcal{{P}}_{{\\mathcal{{E}},I,f}}(D,s) = [I;f(\\mathcal{{E}}(D,s));f(s)] \\tag{{1}}
```

### Correct Output:
```json
{{
  "entities": [
    {{
      "entity_name": "Formula (1)",
      "description": "An equation defining the function P in terms of several input variables. LaTeX: \\mathcal{{P}}_{{\\mathcal{{E}},I,f}}(D,s) = [I;f(\\mathcal{{E}}(D,s));f(s)] \\tag{{1}}"
    }}
  ]
}}
```

---
-Task-

Now, analyze the input equation provided at the beginning of this prompt and generate the JSON object.

### Input LaTeX Equation
```latex
{formula_latex_code}
```

### JSON Output

"""


SECTION_ENTITY_EXTRACTION = """
-Goal-
You are a highly specialized knowledge extraction engine for structured documents. Your task is to analyze a single section title, its hierarchical context, and its surrounding sections to extract a comprehensive set of entities and their relationships.

-Context-
You will receive a primary JSON input containing:
1.  `context`: An object detailing the title's position in the document (its full path, previous title, and next title).
2.  `title_to_process`: The specific title string you need to analyze.

You MUST use this full context to inform your extraction, especially for generating rich descriptions.

-Instructions & Extraction Logic-

1.  **Create the Main Section Entity**:
    * First, create one primary entity for the `title_to_process` itself.
    * Its `entity_type` **MUST** be `SECTION_TITLE`.
    * Its `description` **MUST** be a rich summary that explains the section's role, using the provided `context` (e.g., "As a subsection of 'Methodology', this section details...").

2.  **Extract Sub-Entities (If Applicable)**:
    * Next, analyze the text of the `title_to_process` to determine if it contains multiple, distinct conceptual entities.
    * **If** the title clearly lists separate concepts (e.g., from "Task, Dataset and Evaluation"), you **MUST** extract each one as a sub-entity.
    * **If** the title represents a single, indivisible concept (e.g., "Introduction"), then **no sub-entities should be extracted**, and you can skip to Step 4.
    * For any extracted sub-entity, you must assign the most accurate `entity_type` from the allowed list:
    {entity_types}

3.  **Create Relationships (If Sub-Entities Were Extracted)**:
    * **If, and only if,** you extracted one or more sub-entities in Step 2, you **MUST** create relationships to link them to the main section entity created in Step 1.
    * For each sub-entity, create one relationship where the sub-entity is the source (`src_entity_name`) and the main section entity is the target (`tgt_entity_name`).
    * The relationship description should clarify that the sub-entity is a topic of the section.

4.  **Final Output**:
    * Combine the main section entity, all sub-entities, and all relationships into a single JSON object adhering to the output format below.

---
-Output Format-

### 1. General Instruction
Your response MUST be a single, valid JSON object with the root keys "entities" and "relationships".

### 2. JSON Structure
```json
{{
  "entities": [
    {{
      "entity_name": "<String>",
      "entity_type": "<String>",
      "description": "<String>"
    }}
  ],
  "relationships": [
    {{
      "src_entity_name": "<String>",
      "tgt_entity_name": "<String>",
      "weight": "<Float>",
      "description": "<String>"
    }}
  ]
}}
```

### 3. Field Descriptions
* **`entity_name`** (String): The name of the entity.
* **`entity_type`** (String): The category of the entity. It MUST be one of the following types: {entity_types}
* **`description`** (String): A brief, comprehensive summary of the entity's role and attributes.
* **`src_entity_name`**, **`tgt_entity_name`** (String): Must match an `entity_name` from the "entities" list.
* **`weight`** (Float): A score from 1.0 to 10.0 indicating confidence.
* **`description`** (String): A detailed explanation of the relationship.

---
-Example-

### Allowed Entity Types (for this example):

["SECTION_TITLE", "TASK_OR_PROBLEM", "DATASET_OR_CORPUS", "EVALUATION_METRIC", "METHODOLOGY"]

### Input to Process (for this example):
```json
{{
  "context": {{
    "previous_section_title": null,
    "current_section_path": [
      {{
        "depth": 0,
        "title": "Large Language Model Is Not a Good Few-shot Information Extractor..."
      }},
      {{
        "depth": 1,
        "title": "3. Methodology"
      }},
      {{
        "depth": 2,
        "title": "3.1 Task, Dataset and Evaluation"
      }}
    ],
    "next_section_title": "3.2 Small Language Models"
  }},
  "title_to_process": "3.1 Task, Dataset and Evaluation"
}}
```

### Correct JSON Output (for this example):
```
{{
  "entities": [
    {{
      "entity_name": "3.1 Task, Dataset and Evaluation",
      "entity_type": "SECTION_TITLE",
      "description": "As a subsection of 'Methodology', this section details the experimental framework, including the tasks, datasets, and evaluation methods used in the study."
    }},
    {{
      "entity_name": "Task",
      "entity_type": "TASK_OR_PROBLEM",
      "description": "Refers to the specific challenge being addressed, which is detailed within section 3.1."
    }},
    {{
      "entity_name": "Dataset",
      "entity_type": "DATASET_OR_CORPUS",
      "description": "Refers to the data corpora used for experiments, as specified in section 3.1."
    }},
    {{
      "entity_name": "Evaluation",
      "entity_type": "EVALUATION_METRIC",
      "description": "Refers to the metrics for performance assessment, which are defined in section 3.1."
    }}
  ],
  "relationships": [
    {{
      "src_entity_name": "Task",
      "tgt_entity_name": "3.1 Task, Dataset and Evaluation",
      "weight": 10.0,
      "description": "The concept of 'Task' is a primary topic of section 3.1."
    }},
    {{
      "src_entity_name": "Dataset",
      "tgt_entity_name": "3.1 Task, Dataset and Evaluation",
      "weight": 10.0,
      "description": "The concept of 'Dataset' is a primary topic of section 3.1."
    }},
    {{
      "src_entity_name": "Evaluation",
      "tgt_entity_name": "3.1 Task, Dataset and Evaluation",
      "weight": 10.0,
      "description": "The concept of 'Evaluation' is a primary topic of section 3.1."
    }}
  ]
}}
```

---
-Task Execution-

- Allowed Entity Types -
{entity_types}

- Input to Process -

```json
{input_json}
```
-JSON Output-
"""

# The following PROMPTs are designed for KG refiner

# 957 tokens
SUMMARIZE_ENTITY_DESCRIPTIONS_OLD = """
-Goal-
You are an expert Knowledge Graph Consolidator. Your task is to analyze two entity objects, which are believed to represent the same real-world concept. You must intelligently merge their information into a single, comprehensive, and canonical entity. The final output must be a single, valid JSON object that adheres to the provided schema.

---
-Context-
You will be given two entity objects, `entity_1` and `entity_2`, in a JSON format. They may have identical or slightly different names, types, and descriptions. Your goal is to resolve these differences and produce the best possible merged version.

---
-Input Data Format-
```
{{
  "entity_1": {{
    "entity_name": "<String>",
    "entity_type": "<String>",
    "description": "<String>"
  }},
  "entity_2": {{
    "entity_name": "<String>",
    "entity_type": "<String>",
    "description": "<String>"
  }}
}}
```

---
-Instructions & Steps-

1.  **Determine the Canonical `entity_name`**:
    * First, check if one name is an abbreviation or acronym of the other (e.g., "LLM" and "Large Language Model"). If so, create a new, combined name that includes both, such as `Large Language Model (LLM)`.
    * Otherwise, if the names are simply different versions (e.g., "Apple" and "Apple Inc."), choose the more formal or complete one.
    * If the names are identical, simply use that name.

2.  **Determine the Canonical `entity_type`**:
    * Analyze `entity_1.entity_type` and `entity_2.entity_type`.
    * Select the most specific and accurate type that is available in the `-Allowed Entity Types-` list provided below. For example, `MODEL_OR_ARCHITECTURE` is more specific than `PRODUCT`.
    * Your final choice **MUST** be one of the types from the allowed list:
    {entity_types}

3.  **Synthesize the Comprehensive `description`**:
    * Read and understand both `entity_1.description` and `entity_2.description`.
    * Combine all unique, non-contradictory facts from both descriptions into a single, coherent paragraph.
    * If you find contradictions, use your reasoning to resolve them or state the most likely fact.
    * The final description should be written in the third person and be as informative as possible.

4.  **Construct the Final JSON Output**:
    * Assemble the chosen `entity_name`, `entity_type`, and the synthesized `description` into a single JSON object that matches the required output format.

---
-Output Format & Schema-

### 1. General Instruction
Your response MUST be a single, valid JSON object, representing the merged entity.

### 2. JSON Structure (conforming to `ExtractEntity`)
```
{{
  "entity_name": "<String>",
  "entity_type": "<String>",
  "description": "<String>"
}}
```

### 3. Field Descriptions
* **`entity_name`** (String): The final, canonical name of the merged entity.
* **`entity_type`** (String): The final, most accurate category for the entity. It MUST be one of the types from the allowed list below.
* **`description`** (String): The final, comprehensive summary combining information from both source entities.

---
-Example-

### Allowed Entity Types (for this example):
["ORGANIZATION", "PERSON", "PRODUCT", "TECHNOLOGY"]

### Input Entities to Merge (for this example):

```json
{{
  "entity_1": {{
    "entity_name": "OpenAI",
    "entity_type": "COMPANY",
    "description": "OpenAI is an AI research lab."
  }},
  "entity_2": {{
    "entity_name": "OpenAI Inc.",
    "entity_type": "ORGANIZATION",
    "description": "An artificial intelligence company that developed the GPT series of models."
  }}
}}
```

### Correct JSON Output (for this example):
```
{{
  "entity_name": "OpenAI Inc.",
  "entity_type": "ORGANIZATION",
  "description": "OpenAI Inc. is an artificial intelligence (AI) research lab and company that developed the GPT series of models."
}}
```

---
-Task Execution-

Now, perform the merge operation based on the following data.

- Allowed Entity Types -
{entity_types}

- Input Entities to Merge -
```json
{input_json}
```

-JSON Output-
"""


class MergedEntitySchema(BaseModel):
    entity_name: str  # Primary key for entity
    entity_type: str = Field(default="")  # Entity type


SUMMARIZE_ENTITY = """
-Goal-
You are an expert Knowledge Graph Consolidator. Given two entity objects believed to represent the same concept, your task is to determine the canonical `entity_name` and the most appropriate `entity_type`.

---
-Context-
You will analyze two entities, `entity_1` and `entity_2`. Use all their information—especially their `description`s—as context to make the best decision for the final name and type.

---
-Instructions & Steps-

1.  **Determine the Canonical `entity_name`**:
    * If one name is an abbreviation of the other (e.g., "LLM" vs "Large Language Model"), create a combined name like `Large Language Model (LLM)`.
    * If the names are different versions (e.g., "Apple" vs "Apple Inc."), choose the more formal or complete one.
    * If identical, use that name.

2.  **Determine the Canonical `entity_type`**:
    * Analyze both `entity_type` fields and the context from the descriptions.
    * Select the most specific and accurate type from the `-Allowed Entity Types-` list below.
    * Your final choice **MUST** be one of the types from the allowed list:
    {entity_types}

3.  **Construct the Final JSON Output**:
    * Assemble the chosen `entity_name` and `entity_type` into a single JSON object as specified in the output format.

---
-Output Format & Schema-
Your response MUST be a single, valid JSON object containing only the following two keys.

### 2. JSON Structure
```json
{{
  "entity_name": "<String>",
  "entity_type": "<String>",
}}
```

---
-Example-

### Allowed Entity Types (for this example):
["ORGANIZATION", "PERSON", "PRODUCT", "TECHNOLOGY"]

### Input Entities to Merge (for this example):

```json
{{
  "entity_1": {{
    "entity_name": "OpenAI",
    "entity_type": "COMPANY",
    "description": "OpenAI is an AI research lab."
  }},
  "entity_2": {{
    "entity_name": "OpenAI Inc.",
    "entity_type": "ORGANIZATION",
    "description": "An artificial intelligence company that developed the GPT series of models."
  }}
}}
```

### Correct JSON Output (for this example):
```
{{
  "entity_name": "OpenAI Inc.",
  "entity_type": "ORGANIZATION",
}}
```

---
-Task Execution-

Now, perform the merge operation based on the following data.

- Allowed Entity Types -
{entity_types}

- Input Entities to Merge -
```json
{input_json}
```

-JSON Output-
"""


class ERExtractSel(BaseModel):
    select_id: int
    explanation: str = Field(default="")  # Entity type


DESCRIPTION_SYNTHESIS = """
**Task**: You will be given a JSON object containing an entity. The `description` field is fragmented with `<SEP>` separators. Your goal is to synthesize these fragments into a single, clean paragraph and output it as a valid Python string.

**Instructions**:

1.  **Synthesize**: Combine all unique and non-contradictory facts from the description fragments into one coherent paragraph.
2.  **Contextualize**: The new paragraph must be written in the third person and include the entity's name to ensure the description is self-contained and clearly understandable.
3.  **Output**: Provide the final, synthesized paragraph directly as your response.

**Crucial Rule**: Your entire response must ONLY be the final description text. Do not include any extra text, labels, or formatting.

-----

**Input Entity**:

```json
{input_json}
```

**Output**:
"""


ENTITY_RESOLUATION_PROMPT = """
-Goal-
You are an expert Entity Resolution Adjudicator. Your task is to determine if a "New Entity" refers to the exact same real-world concept as one of the "Candidate Entities" provided from a knowledge graph. Your output must be a JSON object containing the ID of the matching candidate (or -1) and a brief explanation for your decision.


---
-Context-
You will be given one "New Entity" recently extracted from a text. You will also be given a list of "Candidate Entities" that are semantically similar, retrieved from an existing knowledge base. Each candidate has a unique `id` for you to reference.

---
-Core Task & Rules-

1.  **Analyze the "New Entity"**: Carefully read its name, type, and description to understand what it is.

2.  **Field-by-Field Adjudication**: To determine a match, you must evaluate each field with a specific focus:
    * **`entity_name` (High Importance):** The names must be extremely similar, a direct abbreviation (e.g., "LLM" vs. "Large Language Model"), or a well-known alias. **If the names represent distinct, parallel concepts (like "Event Detection" and "Named Entity Recognition"), they are NOT a match, even if their descriptions are very similar.**
    * **`entity_type` (Medium Importance):** The types do not need to be identical, but they must be closely related and compatible (e.g., `COMPANY` and `ORGANIZATION` could describe the same entity).
    * **`description` (Contextual Importance):** The descriptions may differ as they are often extracted from different parts of a document. Your task is to look past surface-level text similarity and determine if they fundamentally describe the **same underlying object or concept**.

3.  **Be Strict and Conservative**: Your standard for a match must be very high. An incorrect merge can corrupt the knowledge graph. A missed merge is less harmful.
    * Surface-level similarities are not enough. The underlying concepts must be identical.
    * For example, "Apple" (the fruit) and "Apple Inc." (the company) are NOT a match.
    * **When in doubt, you MUST output -1.**
    * **Assume No Match by Default**: In a large knowledge graph, most new entities are genuinely new. You should start with the assumption that the "New Entity" is unique. You must find **strong, convincing evidence** across all fields, especially the `entity_name`, to overturn this assumption and declare a match.

4.  **Format the Output**: **You must provide your answer in a valid JSON format. The JSON object should contain two keys:**
    * `select_id`: An integer. The `id` of the candidate you've determined to be an exact match. If no exact match is found, this value MUST be `-1`.
    * `explanation`: A brief, one-sentence string explaining your reasoning. For a match, explain why they are the same entity. For no match, explain the key difference.
    
---
-Input Data Format-
The input will be a JSON object containing the `new_entity` and a `candidate_entities` list.

```json
{{
  "new_entity": {{
    "entity_name": "<String>",
    "entity_type": "<String>",
    "description": "<String>"
  }},
  "candidate_entities": [
    {{
      "id": 0,
      "entity_name": "<String>",
      "entity_type": "<String>",
      "description": "<String>"
    }},
    {{
      "id": 1,
      "entity_name": "<String>",
      "entity_type": "<String>",
      "description": "<String>"
    }},
    // ... more candidates
  ]
}}
```

---
-Output Schema & Format-
Your response MUST be a single, valid JSON object that adheres to the following schema. Do not include any other text, explanation, or markdown formatting like ```json.

```json
{{
  "select_id": "integer",
  "explanation": "string"
}}
```

---
-Example-

### Example 1: Match Found

**Input Data (for this example):**

```json
{{
  "new_entity": {{
    "entity_name": "GPT-4",
    "entity_type": "MODEL_OR_ARCHITECTURE",
    "description": "A powerful large language model developed by OpenAI, known for its advanced reasoning capabilities."
  }},
  "candidate_entities": [
    {{
      "id": 0,
      "entity_name": "Generative Pre-trained Transformer 4",
      "entity_type": "MODEL_OR_ARCHITECTURE",
      "description": "GPT-4 is a multimodal model from OpenAI, the fourth in its series of foundational models."
    }},
    {{
      "id": 1,
      "entity_name": "OpenAI",
      "entity_type": "ORGANIZATION",
      "description": "The research lab and company that created the GPT series."
    }},
    {{
      "id": 2,
      "entity_name": "LLM",
      "entity_type": "TECHNOLOGY",
      "description": "A Large Language Model (LLM) is a type of AI model trained on vast amounts of text data."
    }}
  ]
}}
```

**Correct Output (for this example):**

```
json
{{
  "select_id": 0,
  "explanation": "The new entity 'GPT-4' and the candidate 'Generative Pre-trained Transformer 4' are different names for the exact same OpenAI language model."
}}
```

### Example 2: No Match Found

**Input Data (for this example):**

```
{{
  "new_entity": {{
    "entity_name": "Microsoft Word",
    "entity_type": "SOFTWARE",
    "description": "A word processor developed by Microsoft, used for creating and editing text documents."
  }},
  "candidate_entities": [
    {{
      "id": 0,
      "entity_name": "Microsoft Corporation",
      "entity_type": "ORGANIZATION",
      "description": "An American multinational technology corporation which produces computer software, consumer electronics, personal computers, and related services."
    }},
    {{
      "id": 1,
      "entity_name": "Microsoft Excel",
      "entity_type": "SOFTWARE",
      "description": "A spreadsheet developed by Microsoft for Windows, macOS, Android and iOS. It features calculation, graphing tools, pivot tables, and a macro programming language."
    }},
    {{
      "id": 2,
      "entity_name": "Windows 11",
      "entity_type": "SOFTWARE",
      "description": "A major release of the Windows NT operating system developed by Microsoft."
    }}
  ]
}}
```

**Correct Output (for this example):**

```json
{{
  "select_id": -1,
  "explanation": "While related, 'Microsoft Word' is a specific software product, which is distinct from the parent company 'Microsoft Corporation' or other software like 'Excel' or 'Windows 11'."
}}
```

----
-Task Execution-

Now, perform the selection task based on the following data. Remember to output only a single integer.

- Input Data -

{input_json}

-Output ID-
"""


ER_RERANK_INSTRUCTION = (
    "Your primary goal is to determine if a 'Candidate Entity' and a 'Query Entity' "
    "refer to the exact same real-world entity. This is a task of semantic identity verification, "
    "not just textual similarity.\n\n"
    "Even though their descriptions may not be identical (as they are extracted from "
    "various sources), you must analyze the information presented in them. Your decision "
    "should be based on whether this information, when combined, points to a single, "
    "unique real-world entity. Be aware that in some cases, it is expected that "
    "NONE of the provided candidates will be a correct match.\n\n"
    "A high score (e.g., >0.9) means you are highly confident they are the same entity, despite "
    "variations in their descriptions. A low score means they are fundamentally different entities, "
    "even if their names or descriptions share some keywords."
)
