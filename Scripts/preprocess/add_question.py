from Core.configs.system_config import load_system_config
import pandas as pd
import os
import logging
import sys

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.provider.vlm import VLM
from typing import List, Dict, Any, Optional
import json
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

section = NodeType("title")
print(section)

log = logging.getLogger(__name__)

logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


class QAPair(BaseModel):
    question: str = Field(
        description="The generated question about the document content."
    )
    answer: int = Field(description="The numerical answer to the question.")


# 定义了VLM最终需要返回的完整结构，即一个问答对列表
class VLMAnalysisResponse(BaseModel):
    qa_pairs: List[QAPair] = Field(
        description="A list of question and answer pairs generated from the document analysis."
    )


def build_vlm_prompt(page_numbers: List[int]) -> str:
    page_str = ", ".join(map(str, sorted(page_numbers)))

    prompt_template = f"""
You are an expert academic assistant specializing in multi-page document analysis.
Your task is to analyze a batch of images from a document and generate a list of diverse question-answer pairs about its contents.

CONTEXT:
You have been provided with images from the following pages of a document: {page_str}.
IMPORTANT: When referring to page numbers for questions (e.g., "from page X to page Y"), you MUST use these explicitly provided page numbers, not any page numbers visible in the image footers or headers. This ensures consistency.
Assume these are the ONLY pages that contain relevant content like figures or tables for the entire document.

INSTRUCTIONS:
1.  Carefully examine all provided images as a whole.
2.  Generate a variety of questions based on the instructions below. The answer to each question MUST be a single numerical value.
3.  IMPORTANT: Only generate question-answer pairs where the answer is greater than zero. Do not include questions if the count is 0.
4.  Generate questions about BOTH "figures" and "tables".
5.  Your generated questions should cover the following types:
    a.  **Related Count**: Ask for the count of figures/tables related to a specific topic or category.
    b.  **Sub-range Count**: Ask for the count of figures/tables within a logical sub-range of the provided pages (e.g., "from page X to page Y").
    c.  **Section-based Count**: Identify section titles (e.g., "Introduction", "Methodology", "Results") and ask for the count of figures/tables within a specific section visible in the images.
6.  Your final output MUST be a single, valid JSON object that adheres to the required schema, containing a list of these question-answer pairs.

EXAMPLE JSON OUTPUT:
{{
  "qa_pairs": [
    {{
      "question": "Among figure 1-4, how many figures show more than one breccia gash?",
      "answer": 2,
    }},
    {{
      "question": "How many tables are on pages 3 to 5?",
      "answer": 2,
    }},
    {{
      "question": "How many figures are in the 'Results' section?",
      "answer": 4,
    }}
  ]
}}

Provide only the JSON object as your response.
"""
    return prompt_template


def generate_qa_sets_from_images(
    processed_image_paths: List[str],
    page_numbers: List[int],
    vlm: VLM,
) -> Optional[List[Dict[str, Any]]]:
    """
    Generates a set of question-answer pairs from a pre-filtered list of images.

    Args:
        processed_image_paths (List[str]): The list of image file paths to be analyzed.
        page_numbers (List[int]): The corresponding page numbers for the images.
        vlm (VLM): An instance of the VLM client.

    Returns:
        A list of dictionaries, where each dictionary is a QA pair, or None on failure.
    """
    if not processed_image_paths or not page_numbers:
        print("Error: Image paths and page numbers cannot be empty.")
        return None

    prompt = build_vlm_prompt(page_numbers)

    try:
        vlm_response: VLMAnalysisResponse = vlm.generate_json(
            images=processed_image_paths,
            prompt_or_memory=prompt,
            schema=VLMAnalysisResponse,
        )
        return [qa.model_dump_json() for qa in vlm_response.qa_pairs]

    except Exception as e:
        print(f"An error occurred during VLM processing: {e}")
        return None


def late_check(qa_pair):
    question_str = qa_pair["question"].lower()
    answer = qa_pair["answer"]
    if not isinstance(answer, int):
        try:
            answer = int(answer)
        except:
            return False

    figure_flag = False
    if (
        "figure" in question_str
        or "figures" in question_str
        or "image" in question_str
        or "images" in question_str
    ):
        figure_flag = True

    return figure_flag


def process_document_group(
    group_key: tuple, index_path: str, vlm: Any
) -> List[Dict[str, Any]]:
    """
    Processes a single document group to extract nodes, generate QA pairs, and validate them.
    This function is designed to be executed in a separate thread.
    """
    doc_uuid, doc_path = group_key
    log.info(f"Starting processing for document: {doc_uuid}")

    try:
        tree_path = os.path.join(index_path, doc_uuid, "tree.pkl")
        if not os.path.exists(tree_path):
            log.warning(
                f"Tree file not found for doc {doc_uuid} at {tree_path}, skipping."
            )
            return []

        tree_index = DocumentTree.load_from_file(tree_path)

        image_nodes = tree_index.get_filtered_nodes(NodeType.IMAGE)
        table_nodes = tree_index.get_filtered_nodes(NodeType.TABLE)
        log.info(
            f"Doc {doc_uuid} has {len(image_nodes)} image nodes and {len(table_nodes)} table nodes."
        )

        image_pages = {node.meta_info.page_idx for node in image_nodes}
        table_pages = {node.meta_info.page_idx for node in table_nodes}
        all_pages = sorted(list(image_pages.union(table_pages)))

        if not all_pages:
            log.info(
                f"No image or table pages found for document {doc_uuid}, skipping."
            )
            return []

        # Limit the number of pages to process
        MAX_PAGES = 10
        pages_to_process = all_pages[:MAX_PAGES]

        trans_images_dir = os.path.join(index_path, doc_uuid, "vlm", "pages_images")
        filter_page_idx = []
        filter_image_paths = []
        for page_idx in pages_to_process:
            image_path = os.path.join(trans_images_dir, f"{page_idx}.png")
            if os.path.exists(image_path):
                filter_page_idx.append(page_idx + 1)  # Adjust to 1-based index for VLM
                filter_image_paths.append(image_path)
            else:
                log.warning(f"Image file not found: {image_path}")

        if not filter_image_paths:
            log.info(
                f"No valid image paths found for {doc_uuid} after filtering, skipping."
            )
            return []

        log.info(f"Doc {doc_uuid}: Generating QA pairs using pages: {filter_page_idx}")
        generated_qa = generate_qa_sets_from_images(
            processed_image_paths=filter_image_paths,
            page_numbers=filter_page_idx,
            vlm=vlm,
        )

        processed_pairs = []
        if generated_qa:
            log.info(f"Generated {len(generated_qa)} QA pairs for document {doc_uuid}.")
            for qa_pair in generated_qa:
                try:
                    qa_pair = json.loads(qa_pair)
                    figure_flag = late_check(qa_pair)
                    qa_pair["doc_uuid"] = doc_uuid
                    qa_pair["doc_path"] = doc_path
                    qa_pair["figure_flag"] = figure_flag
                    processed_pairs.append(qa_pair)
                except Exception as e:
                    log.error(
                        f"Error during late check for doc {doc_uuid}: {e}",
                        exc_info=True,
                    )
        return processed_pairs

    except Exception as e:
        log.error(
            f"An unexpected error occurred while processing doc {doc_uuid}: {e}",
            exc_info=True,
        )
        return []


if __name__ == "__main__":
    cfg = load_system_config("gbc.yaml")
    data_path = "m3docrag.json"
    index_path = "m3docrag/index"

    data_df = pd.read_json(data_path)
    print(data_df.shape)
    api_key = "your_api_key_here"  # your_api_key_here should be replaced with your actual API key for the VLM service
    api_base = "your_api_base_here"  # your_api_base_here should be replaced with the actual base URL for the VLM API you are using

    cfg_vlm = cfg.model_copy()
    cfg_vlm.vlm.api_base = api_base
    cfg_vlm.vlm.api_key = api_key
    cfg_vlm.vlm.model_name = "gemini-3-pro"  # You can change this to the specific model you want to use, e.g., "gemini-3-pro", "gemini-2.5-pro", etc.
    vlm = VLM(cfg_vlm.vlm)

    group_cols = ["doc_uuid", "doc_path"]
    document_groups = data_df.groupby(group_cols)
    log.info(f"Total documents to process: {len(document_groups)}")

    res_paris = []
    # Adjust max_workers based on your API rate limits and system capabilities
    MAX_WORKERS = 4

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all document processing tasks to the thread pool
        future_to_group = {
            executor.submit(
                process_document_group, group_key, index_path, vlm
            ): group_key
            for group_key, _ in document_groups
        }

        # Process results as they are completed
        for future in as_completed(future_to_group):
            group_key = future_to_group[future]
            try:
                result_list = future.result()
                if result_list:
                    res_paris.extend(result_list)
            except Exception as e:
                log.error(
                    f"Task for doc {group_key[2]} generated an exception: {e}",
                    exc_info=True,
                )

    log.info(
        f"Finished processing all documents. Total QA pairs collected: {len(res_paris)}"
    )

    print(f"Total QA pairs generated: {len(res_paris)}")
    generate_df = pd.DataFrame(res_paris)

    output_path = "test_qa_pairs.json"
    generate_df.to_json(output_path, orient="records", indent=4, force_ascii=False)
    print(f"Saved generated QA pairs to {output_path}")
