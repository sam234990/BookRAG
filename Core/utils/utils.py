import tiktoken
from typing import Optional, Any, Union, List, Dict
import html

import logging
import json
import re
from json_repair import repair_json


log = logging.getLogger(__name__)


def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    return len(token_encoder.encode(text))


def try_parse_json_object(input: str) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        log.info("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```json"):
        input = input[len("```json") :]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        input = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:
            result = json.loads(input)
        except json.JSONDecodeError:
            log.exception("error loading json, json=%s", input)
            return input, {}
        else:
            if not isinstance(result, dict):
                log.exception("not expected dict type. type=%s:", type(result))
                return input, {}
            return input, result
    else:
        return input, result


def get_json_content(any_list: list[Optional[str]], selected_columns: list[str]) -> str:
    """Get JSON content from the PDF list."""
    json_list = []
    for content in any_list:
        if isinstance(content, dict):
            filtered_content = {
                k: v for k, v in content.items() if k in selected_columns
            }
            json_list.append(filtered_content)
    json_list_str = json.dumps(json_list, indent=4, ensure_ascii=False)
    return json_list_str


def enumerate_pdf_list(pdf_list: list[Optional[str]]) -> list[Optional[str]]:
    """Enumerate the PDF content list."""
    enumerated_list = []
    i = 1  # pdf_id starts from 1
    for content in pdf_list:
        if content is not None and content.get("invalid", False) == False:
            content["pdf_id"] = i
            enumerated_list.append(content)
            i += 1
    return enumerated_list


def split_string_by_multi_markers(text: str, delimiters: list[str]) -> list[str]:
    """
    Split a string by multiple delimiters.

    Args:
        text (str): The string to split.
        delimiters (list[str]): A list of delimiter strings.

    Returns:
        list[str]: A list of strings, split by the delimiters.
    """
    if not delimiters:
        return [text]
    split_pattern = "|".join(re.escape(delimiter) for delimiter in delimiters)
    segments = re.split(split_pattern, text)
    return [segment.strip() for segment in segments if segment.strip()]


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

    # Remove non-alphanumeric characters and convert to lowercase
    return re.sub("[^A-Za-z0-9 ]", " ", result.lower()).strip()


def is_float_regex(value: str) -> bool:
    """
    Check if a string matches the regular expression for a float.

    Args:
        value (str): The string to check.

    Returns:
        bool: Whether the string matches the regular expression.
    """
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def get_input_tokens(prompt_or_memory: Union[str, List[Dict[str, Any]]]) -> int:
    """
    Get the number of tokens in the input string or list of strings.

    """
    if isinstance(prompt_or_memory, str):
        return num_tokens(prompt_or_memory)
    elif isinstance(prompt_or_memory, List):
        prompt_num_tokens = 0
        for message in prompt_or_memory:
            for key, value in message.items():
                prompt_num_tokens += num_tokens(key)
                # If the value is a string, count its tokens
                content = value
                if isinstance(content, str):
                    prompt_num_tokens += num_tokens(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            prompt_num_tokens += num_tokens(item)
                else:
                    raise TypeError(
                        "Message values must be strings or lists of strings."
                    )
        prompt_num_tokens += (
            3  # every reply is primed with <|start|>assistant<|message|>
        )
        return prompt_num_tokens
    else:
        raise TypeError("Input must be a string or a list of strings.")


def get_max_output_tokens(
    prompt_or_memory: Union[str, List[Dict[str, Any]]], max_model_token: int
) -> int:
    # 3 for the end token
    # max 0 for negative values
    inptput_token = get_input_tokens(prompt_or_memory)
    max_output_toekens = max(max_model_token - inptput_token - 3, 0)
    return max_output_toekens


def truncate_description(desc: str, max_words: int = 100) -> str:
    words = desc.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return desc


class TextProcessor:
    @staticmethod
    def split_text_into_chunks(text: str, max_length: int = 1000) -> List[str]:
        """
        Split the input text into chunks, each not exceeding max_length tokens.
        Args:
            text: 需要被分割的原始文本。
            max_length: 每个块的最大 token 数量。

        Returns:
            一个由文本块组成的列表。
        """
        # 1. 输入校验
        if not text:
            return []

        # 2. 使用正则表达式将文本分割成句子
        # 这个正则表达式会在. ! ? 后面跟着的空白符处进行分割，保留标点符号在句子末尾。
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        # 过滤掉可能产生的空字符串
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return (
                [text] if num_tokens(text) <= max_length else [text[:max_length]]
            )  # 简单处理无句号的短文本

        # 3. 迭代句子，组合成块
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = num_tokens(sentence)

            # --- 边缘情况处理: 单个句子超过最大长度 ---
            if sentence_tokens > max_length:
                # 如果当前有正在构建的块，先将其保存
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_chunk_tokens = 0

                # 对这个超长的句子按词进行切分
                words = sentence.split()
                current_sub_chunk_words = []
                for word in words:
                    # 检查单个单词是否就已超长 (极端情况)
                    if num_tokens(word) > max_length:
                        if current_sub_chunk_words:
                            chunks.append(" ".join(current_sub_chunk_words))
                            current_sub_chunk_words = []
                        # 强制切分这个超长单词
                        # 注意：这里是简单的字符级切分，可能会破坏单词
                        for i in range(0, len(word), max_length):
                            chunks.append(word[i : i + max_length])
                        continue

                    # 如果将当前单词加入子块会超出长度，则先将现有子块保存
                    potential_sub_chunk = " ".join(current_sub_chunk_words + [word])
                    if num_tokens(potential_sub_chunk) > max_length:
                        chunks.append(" ".join(current_sub_chunk_words))
                        current_sub_chunk_words = [word]
                    else:
                        current_sub_chunk_words.append(word)

                # 将最后一个剩余的子块加入结果列表
                if current_sub_chunk_words:
                    chunks.append(" ".join(current_sub_chunk_words))

                continue  # 处理完这个长句，继续下一个句子

            # --- 正常组合逻辑 ---
            # 如果将当前句子加入块中会超出长度，则先将现有块保存
            if (
                current_chunk_tokens + sentence_tokens > max_length
                and current_chunk_sentences
            ):
                chunks.append(" ".join(current_chunk_sentences))
                # 开始一个新的块
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
            else:
                # 否则，将当前句子加入块中
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens

        # 4. 将最后一个剩余的块加入结果列表
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    @staticmethod
    def split_texts_into_chunks(texts: List[str], max_length: int = 1000) -> List[str]:
        """
        Splits a LIST of texts into a single list of chunks.

        This method iterates through each text in the input list and applies the
        split_text_into_chunks logic to it, effectively flattening the result.

        Args:
            texts: A list of original texts to be split.
            max_length: The maximum number of tokens for each chunk.

        Returns:
            A single list containing all text chunks from all input texts.
        """
        all_chunks = []
        for text in texts:
            # Reuse the existing static method for single text processing
            chunks_from_one_text = TextProcessor.split_text_into_chunks(
                text, max_length
            )
            all_chunks.extend(chunks_from_one_text)
        return all_chunks

    # 只是为了方便演示，添加一个调用方法
    def process_and_print_chunks(self, sample_text: str, max_tokens: int):
        print(f"--- Splitting text with max_length = {max_tokens} ---")
        chunks = TextProcessor.split_text_into_chunks(
            sample_text, max_length=max_tokens
        )
        for i, chunk in enumerate(chunks):
            chunk_token_count = num_tokens(chunk)
            print(f"Chunk {i+1} (Tokens: {chunk_token_count}):")
            print(f'"{chunk}"\n')
        print("--- End of splitting ---\n")
