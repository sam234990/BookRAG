from typing import Optional, List, Dict

from Core.provider.llm import LLM
from Core.prompts.refiner_prompt import (
    TABLE_MERGE_PROMPT,
    MergeJudgmentsResponse,
    TEXT_MERGE_PROMPT,
    StitchingJudgmentsResponse,
)
from Core.utils.utils import num_tokens, get_json_content, enumerate_pdf_list
import json
import re
import logging
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def is_likely_incomplete_paragraph(text: str) -> bool:
    """
    Determine if an English paragraph is likely incomplete (truncated due to page/column breaks).

    :param text: input text to check
    :return: bool, True if the paragraph is likely incomplete, False otherwise
    e.g. "He said, "This method is the best." -> False (complete)
    e.g. "The quick brown fox jumps over the lazy dog and" -> True (incomplete)
    """
    if not text:
        return False  # 空文本不是我们关心的“不完整段落”

    text = text.strip()

    # Rule 1: Filter out very short strings. They are likely standalone titles/captions, not paragraphs to be merged.
    if len(text.split()) < 5 or len(text) < 25:
        return False

    # --- From here, we look for clear signals of INCOMPLETENESS ---

    # Rule 2: Ending with a hyphen is a very strong signal of incompleteness (a word was split).
    if text.endswith("-"):
        return True

    # Handles cases like "said he," or "he said."
    cleaned_text = re.sub(r"['\"]+$", "", text)

    # Rule 3: Ending with a comma, colon, or semicolon is also a strong signal.
    if cleaned_text.endswith((",", ":", ";")):
        return True

    # Rule 4: Not ending with a standard terminal punctuation mark. This is the most common case.
    if not re.search(r"[.!?]$", cleaned_text):
        return True

    # Rule 5: Ending with a common connector word (even if mistakenly followed by a period).
    # e.g., "The quick brown fox jumps over the lazy dog and."
    incomplete_endings = {
        "and",
        "or",
        "but",
        "because",
        "although",
        "however",
        "if",
        "while",
        "when",
        "to",
        "for",
        "in",
        "of",
        "with",
        "on",
        "as",
        "at",
        "by",
        "from",
        "such",
        "the",
        "a",
        "an",
    }
    last_word_match = re.findall(r"\b\w+\b", cleaned_text)
    if last_word_match and last_word_match[-1].lower() in incomplete_endings:
        return True

    # If no "incomplete" signals were triggered, we assume it's complete.
    return False


def is_first_word_acronym(text: str) -> bool:
    """
    Check if the first word of the text is an acronym (all uppercase letters).
    :param text: The input text to check
    :return: True if the first word is an acronym, False otherwise
    e.g. "LLM is a powerful tool." -> True
    e.g. "This is a test." -> False
    """
    if not text:
        return False
    parts = text.split()
    if not parts:
        return False
    first_word = parts[0]
    # Check if the first word is all uppercase and longer than 1 character
    # This is a simple heuristic for acronyms
    return first_word.isupper() and len(first_word) > 1


def search_continuation_candidates(
    cur_content: dict,
    pdf_list: list[dict],
    cur_idx: int,
    end_idx: int,
    max_page_gap: int = 2,
) -> list[dict]:
    """
    Hint for searching continuation candidates in a PDF list.
    :param cur_content: current incomplete paragraph content
    :param pdf_list: all PDF paragraphs
    :param cur_idx: current paragraph index
    :param end_idx: end index for searching (inclusive)
    :param max_page_gap: maximum number of pages to skip backward
    :return: a group of candidate paragraphs (for splicing)
    """
    candidates = []
    cur_page_idx = cur_content["page_idx"]

    # For Rule #6: counts the number of "normal" capitalized paragraphs encountered.
    normal_uppercase_count = 0

    i = cur_idx + 1
    while i <= end_idx and i < len(pdf_list):
        next_content = pdf_list[i]
        next_page_idx = next_content.get("page_idx", -1)

        # Rule #1: If the next paragraph is too far away, stop searching.
        if next_page_idx > cur_page_idx + max_page_gap:
            # If the next paragraph is too far away, stop searching
            break

        # Rule #2: Skip non-body content like tables, images, or titles.
        if (
            next_content.get("type") in ["table", "image"]
            or next_content.get("text_level", -1) >= 0
        ):
            i += 1
            continue

        # Rule #3: If it's an equation, it might belong to the current paragraph. Add and continue.
        if next_content.get("type") == "equation":
            candidates.append(next_content)
            i += 1
            continue

        next_text = next_content.get("text", "").strip()

        # Skip empty or too short texts
        if not next_text or len(next_text) < 3:
            i += 1
            continue

        # Rule #4 & #5: If the next paragraph starts with a lowercase letter or an acronym,
        # it's very likely part of the current paragraph. Add and continue searching.
        if next_text[0].islower() or is_first_word_acronym(next_text):
            candidates.append(next_content)
            i += 1
            continue

        # Rule #6: Handle paragraphs starting with a "normal" capital letter.
        # This usually signals a new sentence, but could be a parsing error.
        # Strategy: include the first occurrence and when not found candidate, but stop at the second.
        if next_text[0].isupper():
            normal_uppercase_count += 1
            if normal_uppercase_count == 1 and len(candidates) == 0:
                candidates.append(next_content)
                i += 1
                continue
            else:
                break

        # If we reach here, it means the next paragraph is not a continuation
        break

    return candidates


def get_json_str_text(text_candidates, max_tokens) -> List[str]:
    """
    Convert the text candidates into a list of JSON strings for LLM input.
    Each JSON string does not exceed the max_tokens limit.
    """
    pairs_lists = []
    number_of_tokens = 0
    number_of_pairs = 0
    str_list = []
    # Prepare the JSON structure for each text candidate
    columns = ["incomplete_text", "candidate_list"]
    for prev_content, candidates in text_candidates:
        incomplete_text = prev_content.get("text", "")
        candidate_list = [
            {"pdf_id": c.get("pdf_id", -1), "text": c.get("text", "")}
            for c in candidates
        ]
        current_json = {
            "incomplete_text": incomplete_text,
            "candidate_list": candidate_list,
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        current_tokens = num_tokens(cur_json_str)

        if (number_of_tokens + current_tokens < max_tokens) and number_of_pairs < 3:
            # max 3 text pairs
            pairs_lists.append(current_json)
            number_of_tokens += current_tokens
            number_of_pairs += 1
        else:
            # If adding this candidate exceeds the limit, save the current list and start a new one
            str_list.append((number_of_pairs, get_json_content(
                pairs_lists, selected_columns=columns)))
            pairs_lists = [current_json]
            number_of_tokens = current_tokens
            number_of_pairs = 1

    if pairs_lists:
        # If there are remaining candidates, add them to the list
        str_list.append((number_of_pairs, get_json_content(
            pairs_lists, selected_columns=columns)))

    return str_list


def found_remove_text(text_candidate_pairs: list[tuple[dict, list[dict]]], error_json_str: str):
    """
    Find and remove text candidates from the list based on the error JSON string.
    :param text_candidate_pairs: List of text candidates to search in
    :param error_json_str: The JSON string that caused the error
    """
    columns = ["incomplete_text", "candidate_list"]
    for i, (prev_content, candidates) in enumerate(text_candidate_pairs):
        current_json = {
            "incomplete_text": prev_content.get("text", ""),
            "candidate_list": [
                {"pdf_id": c.get("pdf_id", -1), "text": c.get("text", "")} for c in candidates
            ],
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        if cur_json_str in error_json_str:
            log.info(
                f"Found and removed text candidate at index {i}: {cur_json_str}")
            # If the error_json_str contains the current json_str, remove the pair
            text_candidate_pairs.pop(i)


def llm_text_judge(text_candidate_pairs, llm: LLM):
    """
    Judge the text stitching task using the LLM.
    """
    if not text_candidate_pairs:
        log.info("No text candidates found for LLM judgment.")
        return
    json_str_list = get_json_str_text(
        text_candidate_pairs, llm.config.max_tokens -
        num_tokens(TEXT_MERGE_PROMPT) - 400
    )
    llm_infer_results = []
    for number, json_str in json_str_list:
        success = False
        for i in range(2):
            try:
                prompt = TEXT_MERGE_PROMPT.format(json_text=json_str)
                log.info(f"number of tokens in prompt: {num_tokens(prompt)}")
                response = llm.get_json_completion(
                    prompt=prompt, schema=StitchingJudgmentsResponse)
                judgments = response.judgments
                if len(judgments) != number:
                    log.error(
                        f"LLM response length mismatch: {len(judgments)} vs {number}"
                    )
                    continue
                else:
                    llm_infer_results.extend(judgments)
                    success = True
                    break  # Exit the retry loop on success
            except Exception as e:
                log.error(f"LLM error: {e}")
                log.error(f"Prompt: {prompt}")
                continue
        if not success:
            # remove current json_str from the list if all retries failed
            log.error(
                f"Failed to process {number} table pairs with LLM judgment.")
            found_remove_text(text_candidate_pairs, json_str)

    if len(llm_infer_results) != len(text_candidate_pairs):
        log.error(
            f"LLM inference results length mismatch: {len(llm_infer_results)} vs {len(text_candidate_pairs)}"
        )
        return
    # Merge the results reversely
    i = len(llm_infer_results) - 1
    merged_cnt = 0
    while i >= 0:
        llm_res = llm_infer_results[i]
        stitched_pdf_ids = llm_res.stitched_pdf_ids

        if len(stitched_pdf_ids) == 1 and stitched_pdf_ids[0] == -1:
            # If the text should not be merged, skip it
            i -= 1
            continue

        prev_content, candidates = text_candidate_pairs[i]
        selected_candidates = [
            c for c in candidates if c.get("pdf_id", -1) in stitched_pdf_ids
        ]
        if not selected_candidates:
            log.warning(
                f"No candidates found for merging with PDF IDs: {stitched_pdf_ids} at index {i}"
            )
            i -= 1
            continue
        # Merge the text content of selected candidates into prev_content
        merge_text_and_mark_invalid(prev_content, selected_candidates)
        merged_cnt += 1
        i -= 1

    log.info(f"LLM text judgment completed. Merged {merged_cnt} texts.")


def merge_text_and_mark_invalid(prev_content: dict, merged_list: list[dict]):
    """
    Merge the text content of merged_list into prev_content and mark the merged items as invalid.
    Not mark equation as invalid, since it is not a text content.
    :param prev_content: Previous content to merge into
    :param merged_list: List of content to merge
    """
    prev_text = prev_content.get("text", "")
    merged_text = [prev_text]
    for content in merged_list:
        merged_text.append(content.get("text", ""))
        if content.get("type") == "text":
            content["invalid"] = True

    merged_str = ""
    for text in merged_text:
        # if text is end with "-", remove the last character and not add a space
        if text.endswith("-"):
            merged_str += text[:-1]
        else:
            merged_str += text + " "
    prev_content["text"] = merged_str.strip()

    print(f"Merged text in page: {prev_content.get('page_idx', -1) + 1}")
    print(f"Index in Page: {prev_content['middle_json'].get("index", -1) + 1}")
    print(f"{prev_content['text']}")  # Print first 100 chars for debug


def text_merger(pdf_list: list[Optional[str]], llm: LLM) -> list[Optional[str]]:
    incomplete_paragraphs = []
    # for循环的逻辑可以更清晰地组织
    for content in pdf_list:
        if (
            content is None
            or content.get("type") != "text"
            or content.get("text_level", -1) >= 0
        ):  # text_level >= 0
            continue

        # The logic is now direct: "if the paragraph is likely incomplete, add it."
        text = content.get("text", "")
        if is_likely_incomplete_paragraph(text):
            incomplete_paragraphs.append(content)

    if not incomplete_paragraphs:
        log.info("No incomplete paragraphs found.")
        return pdf_list

    log.info(f"Found {len(incomplete_paragraphs)} incomplete paragraphs.")
    llm_infer_candidates = []
    for i in range(len(incomplete_paragraphs)):
        cur_content = incomplete_paragraphs[i]
        start_idx = pdf_list.index(cur_content)
        end_serch_id = (
            pdf_list.index(incomplete_paragraphs[i + 1])
            if i + 1 < len(incomplete_paragraphs)
            else len(pdf_list) - 1
        )
        candidates = search_continuation_candidates(
            cur_content, pdf_list, start_idx, end_serch_id
        )
        if len(candidates) == 0:
            # If no candidates found, skip this paragraph
            continue
        llm_infer_candidates.append((cur_content, candidates))

    if not llm_infer_candidates:
        log.info("No candidates found for merging incomplete paragraphs.")
        return pdf_list
    log.info(
        f"Found {len(llm_infer_candidates)} candidates for merging incomplete paragraphs."
    )

    llm_text_judge(llm_infer_candidates, llm)
    log.info("LLM text judgment completed.")

    return pdf_list


def get_table_col_count(table_html: str) -> int:
    """
    Return the maximum number of columns in a table HTML string.
    :param table_html: HTML string of the table
    :return: maximum number of columns in the table
    """
    soup = BeautifulSoup(table_html, "html.parser")
    max_cols = 0
    for row in soup.find_all("tr"):
        cols = 0
        for cell in row.find_all(["td", "th"]):
            colspan = int(cell.get("colspan", 1))
            cols += colspan
        if cols > max_cols:
            max_cols = cols
    return max_cols


def search_previous_table(
    cur_content: dict, pdf_list: list[dict], cur_idx: int
) -> Optional[dict]:
    """
    Search for the previous table content in the PDF list.
    :param cur_content: current incomplete paragraph content
    :param pdf_list: all PDF paragraphs
    :param cur_idx: current paragraph index
    :return: the previous table content if found, otherwise None
    """
    cur_page_idx = cur_content.get("page_idx", -1)
    if cur_page_idx <= 0:
        return None

    i = cur_idx - 1
    while i >= 0:
        prev_content = pdf_list[i]
        prev_page_idx = prev_content.get("page_idx", -1)
        prev_type = prev_content.get("type", "unknown")

        if prev_page_idx < 0 or prev_page_idx < cur_page_idx - 1:
            # Invalid page index or too far back, stop searching
            break

        if prev_type == "table":
            if prev_page_idx == cur_page_idx:
                # Same page, return None immediately
                return None
            cur_table_html = cur_content.get("table_body", "")
            prev_table_html = prev_content.get("table_body", "")
            if cur_table_html and prev_table_html:
                cur_col_count = get_table_col_count(cur_table_html)
                prev_col_count = get_table_col_count(prev_table_html)
                if cur_col_count == prev_col_count and cur_col_count > 0:
                    return prev_content  # Found matching table, return immediately

            break  # Only matching current table
        else:
            i -= 1

    # If no matching table found in previous page, return None
    return None


def merge_tables_and_mark_invalid(prev_content: dict, cur_content: dict):
    """
    Merge two table HTML contents and mark the current content as invalid.
    :param prev_content: Previous table content (to be merged into)
    :param cur_content: Current table content (to be merged)
    """
    prev_html = prev_content.get("table_body", "")
    cur_html = cur_content.get("table_body", "")
    if not prev_html or not cur_html:
        return

    prev_soup = BeautifulSoup(prev_html, "html.parser")
    cur_soup = BeautifulSoup(cur_html, "html.parser")

    prev_table = prev_soup.find("table")
    prev_tbody = prev_table.find("tbody") if prev_table else None

    cur_trs = cur_soup.find_all("tr")

    if prev_tbody:
        for tr in cur_trs:
            prev_tbody.append(tr)
    elif prev_table:
        for tr in cur_trs:
            prev_table.append(tr)
    else:
        # fallback: 直接拼接字符串
        prev_content["table_body"] += cur_html
        cur_content["invalid"] = True
        return

    # 更新prev_content的table_body
    prev_content["table_body"] = str(prev_table)
    # 标记cur_content为无效
    cur_content["invalid"] = True


def get_json_str_table_pairs(
    table_pairs: list[tuple[dict, dict]], max_tokens: int = 4096
):
    """
    return a list of JSON strings of table pairs for LLM input.
    each JSON string does not exceed the max_tokens limit.
    """
    json_pairs = []
    str_list = []
    number_of_tokens = 0
    number_of_pairs = 0
    columns = ["pdf_id_1", "table_1_html", "table_2_html", "caption"]
    for table_1, table_2 in table_pairs:
        current_json = {
            "pdf_id_1": table_1.get("pdf_id", -1),
            "table_1_html": table_1.get("table_body", ""),
            "table_2_html": table_2.get("table_body", ""),
            "caption": table_1.get("caption", ""),
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        current_tokens = num_tokens(cur_json_str)
        if (number_of_tokens + current_tokens < max_tokens) and number_of_pairs < 3:
            # max 3 table pairs
            json_pairs.append(current_json)
            number_of_tokens += current_tokens
            number_of_pairs += 1
        else:
            str_list.append((number_of_pairs, get_json_content(
                json_pairs, selected_columns=columns)))
            json_pairs = [current_json]
            number_of_tokens = current_tokens
            number_of_pairs = 1

    if json_pairs:
        # If there are remaining pairs, add them to the list
        str_list.append((number_of_pairs, get_json_content(
            json_pairs, selected_columns=columns)))
    return str_list


def found_remove_table(cur_table_pairs: list[tuple[dict, dict]], error_json_str: str):
    columns = ["pdf_id_1", "table_1_html", "table_2_html", "caption"]
    for i, pairs in enumerate(cur_table_pairs):
        table_1, table_2 = pairs
        current_json = {
            "pdf_id_1": table_1.get("pdf_id", -1),
            "table_1_html": table_1.get("table_body", ""),
            "table_2_html": table_2.get("table_body", ""),
            "caption": table_1.get("caption", ""),
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        if cur_json_str in error_json_str:
            log.info(
                f"Found and removed table pair at index {i}: {cur_json_str}")
            # If the error_json_str contains the current json_str, remove the pair
            cur_table_pairs.pop(i)


def llm_table_judger(table_pairs: list[tuple[dict, dict]], llm: LLM):
    """
    Use LLM to judge a list of table pairs whether they can be merged.
    :param table_pairs: a list of table pairs, each pair is a dict with keys
    """

    json_str_list = get_json_str_table_pairs(
        table_pairs, llm.config.max_tokens -
        num_tokens(TABLE_MERGE_PROMPT) - 500
    )
    llm_infer_results = []
    for number, json_str in json_str_list:
        log.info(f"Processing {number} table pairs with LLM judgment.")
        # retry twice to ensure robustness
        success = False
        for i in range(2):
            try:
                prompt = TABLE_MERGE_PROMPT.format(json_pairs=json_str)
                log.info(f"number of tokens in prompt: {num_tokens(prompt)}")
                response = llm.get_json_completion(
                    prompt=prompt, schema=MergeJudgmentsResponse)
                judgements = response.judgments
                if len(judgements) != number:
                    log.error(
                        f"LLM response length mismatch: {len(judgements)} vs {number}"
                    )
                    continue
                else:
                    llm_infer_results.extend(judgements)
                    success = True
                    break  # Exit the retry loop on success
            except Exception as e:
                log.error(f"LLM error: {e}")
                log.error(f"Prompt: {prompt}")
                continue
        if not success:
            # remove current json_str from the list if all retries failed
            log.error(
                f"Failed to process {number} table pairs with LLM judgment.")
            found_remove_table(table_pairs, json_str)

    if len(llm_infer_results) != len(table_pairs):
        log.error(
            f"LLM inference results length mismatch: {len(llm_infer_results)} vs {len(table_pairs)}"
        )
        return

    # Merge the results reversely
    i = len(llm_infer_results) - 1
    merged_cnt = 0
    while i >= 0:
        llm_res = llm_infer_results[i]
        merged_id = llm_res.merged_id
        if merged_id == -1:
            # If the table should not be merged, skip it
            i -= 1
            continue
        prev_content, cur_content = table_pairs[i]
        if prev_content.get("pdf_id", -1) == merged_id:
            merge_tables_and_mark_invalid(prev_content, cur_content)
            merged_cnt += 1
        i -= 1

    log.info(f"LLM table judgment completed. Merged {merged_cnt} tables.")


def table_merger(pdf_list: list[Optional[str]], llm: LLM) -> list[Optional[str]]:
    possible_tables = []
    for content in pdf_list:
        content_type = content.get("type", "unknown")
        if content_type == "table":
            # If the content is a table, we need to check if it should be merged or
            # if the table does not have footnotes or captions.
            footnote = content.get("table_footnote", [])
            caption = content.get("table_caption", [])
            if len(footnote) == 0 and len(caption) == 0:
                possible_tables.append(content)
    if not possible_tables:
        log.info("No tables found to merge.")
        return pdf_list
    log.info(f"Found {len(possible_tables)} tables to merge.")

    candidate_merge_table_pairs: list[tuple[dict, dict]] = []
    for i in range(len(possible_tables)):
        cur_content = possible_tables[i]
        cur_idx = pdf_list.index(cur_content)

        # Search for the previous table
        prev_table = search_previous_table(cur_content, pdf_list, cur_idx)
        if prev_table is not None:
            candidate_merge_table_pairs.append((prev_table, cur_content))

    if not candidate_merge_table_pairs:
        log.info("No candidate table pairs found for merging.")
        return pdf_list
    log.info(
        f"Found {len(candidate_merge_table_pairs)} candidate table pairs for merging."
    )

    llm_table_judger(candidate_merge_table_pairs, llm)
    log.info("LLM table judgment completed.")
    return pdf_list


def dash_line_refiner(pdf_list: list[Optional[str]]):
    for content in pdf_list:
        content_type = content.get("type", "unknown")
        if content_type == "text":
            # If the content is a paragraph, we need to check for dash line errors
            text = content.get("text", "")
            # If a space follows a dash line, it is likely a dash line error
            # We need to remove the space after the dash line
            refined_text = re.sub(r'-\s+', '-', text)
            content["text"] = refined_text
    return pdf_list


def truncate_ocr_error_refiner(
    pdf_list: List[Optional[Dict]],
    window_size: int = 15,
    single_char_ratio_threshold: float = 0.9,
) -> List[Optional[Dict]]:
    """
    Identifies text with trailing OCR garbage and truncates it, preserving the valid part.

    This function uses a sliding window approach to find the starting point of
    OCR errors (like 't t t t') and cuts the text off just before it.

    Args:
        pdf_list: A list of dictionaries representing content blocks.
        window_size: The number of words to check in each sliding window.
        single_char_ratio_threshold: The ratio of single-character words within the
                                     window to trigger a truncation.

    Returns:
        The modified list with garbage text truncated.
    """
    if not pdf_list:
        return []

    for content in pdf_list:
        if content and isinstance(content, dict) and content.get("type") == "text":
            text = content.get("text", "")
            if not text or not isinstance(text, str):
                continue

            words = text.strip().split()

            # Only process text long enough to analyze
            if len(words) < window_size:
                continue

            garbage_start_index = -1

            # --- Sliding Window Logic ---
            # Iterate through the text in windows to find where the garbage begins.
            for i in range(len(words) - window_size + 1):
                window = words[i: i + window_size]

                # Count how many words in the current window are single characters
                single_char_count = sum(1 for word in window if len(word) == 1)

                # If the ratio is high, we've found the start of the garbage
                if single_char_count / window_size >= single_char_ratio_threshold:
                    garbage_start_index = i
                    break

            # If we found a garbage section, truncate the text
            if garbage_start_index != -1:
                # Keep the text *before* the garbage started
                refined_text = " ".join(words[:garbage_start_index])
                # Add a log or print statement to know when truncation happens
                log.info(
                    f"Truncated OCR garbage. Original length: {len(text)}, New length: {len(refined_text)}")
                content["text"] = refined_text

    return pdf_list


def pdf_info_refiner(pdf_list: list[Optional[str]], llm: LLM) -> list[Optional[str]]:
    # Heuristic refiner for "-" error in OCR
    pdf_list = dash_line_refiner(pdf_list)
    # Heuristic refiner for OCR Error
    pdf_list = truncate_ocr_error_refiner(pdf_list)

    # we first enumerate the pdf_list to ensure each content has a unique index
    pdf_list = enumerate_pdf_list(pdf_list)

    # Then we refine the PDF information by merging incomplete paragraphs and tables
    pdf_list = text_merger(pdf_list, llm)
    pdf_list = table_merger(pdf_list, llm)

    # After merging, we need to re-enumerate the pdf_list
    pdf_list = enumerate_pdf_list(pdf_list)
    log.info("PDF information refinement completed.")
    # Return the refined pdf_list
    return pdf_list


if __name__ == "__main__":

    DEBUG = False
    if DEBUG:
        logging.basicConfig(
            level=logging.INFO,  # 或 logging.DEBUG
            format="%(asctime)s %(levelname)s %(message)s",
        )
    print(
        is_likely_incomplete_paragraph('He said, "This method is the best."')
    )  # ✅ True

    # Example usage
    tmp_save_path = "/mnt/data/wangshu/mmrag/m3docrag/index/63a6b3f4-ebee-5024-b87b-84a9bcc26a63/vlm/8513db80c11ea439ab11eba406ec00d9_merged_content.json"
    # tmp_save_path = "/home/wangshu/multimodal/GBC-RAG/test/tree_index/vlm/double_paper_merged_content.json"
    with open(tmp_save_path, "rb") as f:
        pdf_list = json.load(f)
    print(f"Loaded content from {tmp_save_path}")
    from Core.configs.llm_config import LLMConfig

    llm = LLM(LLMConfig())
    pdf_list = pdf_info_refiner(pdf_list, llm)
