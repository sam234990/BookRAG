from typing import Dict, Any, Optional, List, Union, Type
from abc import ABC, abstractmethod
from PIL import Image
import base64
from io import BytesIO
import os
from pydantic import BaseModel

from Core.Common.Message import Message
from Core.Common.Memory import Memory
from Core.provider.TokenTracker import TokenTracker

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

from Core.configs import vlm_config
from Core.configs.vlm_config import VLMConfig
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class BaseVLMController(ABC):
    @abstractmethod
    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        schema: BaseModel = None,
    ) -> Dict:
        pass


class QwenVLController(BaseVLMController):
    def __init__(self, config: VLMConfig):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(config.model_name, use_fast=True)

    def _prepare_messages(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepares the message list for the Qwen-VL processor.
        - If prompt_or_memory is a string, creates a new user message with images and text.
        - If it's a list, uses it directly and prepends images to the last user message's content.
        """
        images = images or []

        # --- Case 1: Input is a simple string query ---
        if isinstance(prompt_or_memory, str):
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt_or_memory})
            return [{"role": "user", "content": content}]

        # --- Case 2: Input is a pre-structured list of messages ---
        elif isinstance(prompt_or_memory, list):
            if not prompt_or_memory:
                raise ValueError("Message list cannot be empty.")

            messages = [dict(m) for m in prompt_or_memory]  # Create a copy

            if images:
                last_message = messages[-1]
                if last_message.get("role") != "user":
                    log.warning(
                        "Images can only be added to the last message if it's from the 'user'. Skipping image attachment."
                    )
                    return messages

                # Prepend images to the content of the last user message
                image_content = [{"type": "image", "image": img} for img in images]

                if isinstance(last_message.get("content"), str):
                    # If content is a string, convert it to the list format
                    text_content = [{"type": "text", "text": last_message["content"]}]
                    last_message["content"] = image_content + text_content
                elif isinstance(last_message.get("content"), list):
                    # If content is already a list, prepend the images
                    last_message["content"] = image_content + last_message["content"]
                else:
                    log.warning(
                        f"Unsupported content type in last message: {type(last_message.get('content'))}. Skipping image attachment."
                    )

            return messages

        else:
            raise TypeError(
                f"Unsupported type for 'prompt_or_memory': {type(prompt_or_memory)}"
            )

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        # Use the helper to prepare messages payload
        messages = self._prepare_messages(prompt_or_memory, images)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images=None,
        schema=None,
    ):
        log.warning("generate_json is not implemented for QwenVLController.")
        pass


class GPTVLMController(BaseVLMController):
    def __init__(self, config: VLMConfig):
        from openai import OpenAI

        self.model_name = config.model_name or "gpt-4o"
        self.client = (
            OpenAI(api_key=config.api_key, base_url=config.api_base)
            if config.api_base
            else OpenAI(api_key=config.api_key)
        )
        self.temperature = config.temperature or 0.1

    def _encode_image(self, image_path):
        if isinstance(image_path, Image.Image):
            buffered = BytesIO()
            image_path.save(buffered, format="JPEG")
            img_data = buffered.getvalue()
            base64_encoded = base64.b64encode(img_data).decode("utf-8")
            return base64_encoded
        else:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

    def _prepare_messages(
        self,
        prompt_or_memory: Union[str, Memory],
        images: Optional[List[Union[str, Image.Image]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepares the message list for the OpenAI API call.
        - If prompt_or_memory is a string, creates a new user message with multimodal content.
        - If it's a list, it uses the list directly and intelligently adds images to the last message.
        """
        # --- Case 1: Input is a simple string query ---
        if isinstance(prompt_or_memory, str):
            content = [{"type": "text", "text": prompt_or_memory}]
            if images:
                for img in images:
                    base64_image = self._encode_image(img)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                    )
            return [{"role": "user", "content": content}]

        # --- Case 2: Input is a pre-structured list of messages ---
        elif isinstance(prompt_or_memory, Memory):
            if not prompt_or_memory:
                raise ValueError("Message list cannot be empty.")

            messages = prompt_or_memory.get()
            messages = [{"role": m.role, "content": m.content} for m in messages]

            # If images are provided, find the last message and append images to its content
            if images:
                last_message = messages[-1]
                # Add a check to ensure images are only added to a 'user' message,
                # as per the OpenAI API specification.
                if last_message.get("role") == "user":
                    # This internal logic was already correct.
                    if isinstance(last_message.get("content"), str):
                        last_message["content"] = [
                            {"type": "text", "text": last_message["content"]}
                        ]
                    elif last_message.get("content") is None:
                        last_message["content"] = []

                    if isinstance(last_message["content"], list):
                        for img in images:
                            base64_image = self._encode_image(img)
                            last_message["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                }
                            )
                    else:
                        log.warning(
                            "Could not attach images: The 'content' of the last message is not a string or list."
                        )
                else:
                    log.warning(
                        f"Could not attach images: The last message role is '{last_message.get('role')}', not 'user'."
                    )
                # --- MODIFICATION 2 END ---
            return messages

        else:
            raise TypeError(
                f"Unsupported type for 'prompt_or_memory': {type(prompt_or_memory)}"
            )

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        content = self._prepare_messages(prompt_or_memory, images)
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=content, temperature=self.temperature
        )

        if completion.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )

        return completion.choices[0].message.content

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[Union[str, Image.Image]]] = None,
        schema: Type[BaseModel] = None,
    ) -> Dict:
        if not schema:
            raise ValueError("A Pydantic schema must be provided for generate_json.")

        # Prepare messages first, which might include a system prompt
        messages = self._prepare_messages(prompt_or_memory, images)

        # Add or modify the system prompt to include JSON instructions
        json_instruction = f"\nYour output MUST conform to this JSON schema: {schema.model_json_schema()}"

        # Check if a system message already exists
        system_message_exists = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] += json_instruction
                system_message_exists = True
                break

        if not system_message_exists:
            messages.insert(0, {"role": "system", "content": json_instruction})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},  # Use modern JSON mode
        )

        if completion.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
            log.info(
                f"Prompt tokens: {completion.usage.prompt_tokens}, Completion tokens: {completion.usage.completion_tokens}"
            )

        return schema.model_validate_json(completion.choices[0].message.content)


class OllamaVLMController(BaseVLMController):
    def __init__(self, config: VLMConfig):
        import ollama

        self.model_name = config.model_name or "qwen2.5vl:latest"
        self.client = ollama.Client(host=config.api_base or "http://127.0.0.1:11434")

    def _prepare_messages(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepares the message list for the Ollama API call.
        - If prompt_or_memory is a string, creates a new user message.
        - If it's a list of dicts, uses it directly and appends images to the last message if needed.
        """
        if isinstance(prompt_or_memory, str):
            # Case 1: Input is a simple string query.
            log.debug("Preparing messages from a string query.")
            return [
                {"role": "user", "content": prompt_or_memory, "images": images or []}
            ]

        elif isinstance(prompt_or_memory, list):
            # Case 2: Input is a pre-structured list of messages.
            log.debug("Using pre-structured message list.")
            if not prompt_or_memory:
                raise ValueError("Message list cannot be empty.")

            # Make a copy to avoid modifying the original list passed by the caller.
            messages = [dict(m) for m in prompt_or_memory]

            # If images are provided, add them to the last message if it doesn't have images already.
            if images:
                last_message = messages[-1]
                if last_message.get("role") == "user" and not last_message.get(
                    "images"
                ):
                    log.debug("Attaching images to the last user message.")
                    last_message["images"] = images
                else:
                    log.warning(
                        "Could not attach images: Last message is not a user role or already contains images."
                    )

            return messages

        else:
            raise TypeError(
                f"Unsupported type for 'prompt_or_memory': {type(prompt_or_memory)}"
            )

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        images = images or []
        try:
            messages = self._prepare_messages(prompt_or_memory, images)
            response = self.client.chat(model=self.model_name, messages=messages)

            if response:
                tracker = TokenTracker.get_instance()
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                try:
                    prompt_tokens = int(prompt_tokens)
                except Exception as e:
                    logging.error(f"Error converting prompt_tokens: {e}")
                    prompt_tokens = 0

                try:
                    completion_tokens = int(completion_tokens)
                except Exception as e:
                    logging.error(f"Error converting completion_tokens: {e}")
                    completion_tokens = 0
                tracker.add_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            return response["message"]["content"]
        except Exception as e:
            log.error(f"OllamaVLMController error: {e}")
            return f"Error: Could not get a response from Ollama model '{self.model_name}'."

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        schema: BaseModel = None,
    ) -> Dict:
        images = images or []
        try:
            messages = self._prepare_messages(prompt_or_memory, images)
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                format=schema.model_json_schema(),
            )

            if response:
                tracker = TokenTracker.get_instance()
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                try:
                    prompt_tokens = int(prompt_tokens)
                except Exception as e:
                    logging.error(f"Error converting prompt_tokens: {e}")
                    prompt_tokens = 0

                try:
                    completion_tokens = int(completion_tokens)
                except Exception as e:
                    logging.error(f"Error converting completion_tokens: {e}")
                    completion_tokens = 0
                tracker.add_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            return schema.model_validate_json(response["message"]["content"])
        except Exception as e:
            log.error(f"OllamaVLMController error: {e}")
            return {
                "error": f"Could not get a JSON response from Ollama model '{self.model_name}'."
            }


class VLM:
    def __init__(self, vlm_config: Optional[Dict[str, Any]] = None):
        if vlm_config is None:
            raise ValueError("VLM config must be provided")
        if not isinstance(vlm_config, VLMConfig):
            config = VLMConfig(**vlm_config)
        else:
            config = vlm_config
        self.config = config
        backend = config.backend.lower()
        if backend == "qwen":
            self.vlm = QwenVLController(config)
        elif backend == "gpt":
            self.vlm = GPTVLMController(config)
        elif backend == "ollama":
            self.vlm = OllamaVLMController(config)
        else:
            raise ValueError(f"Unsupported VLM backend: {backend}")

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        retry = 0
        max_retries = 3
        last_exception = None
        while retry < max_retries:
            try:
                return self.vlm.generate(prompt_or_memory, images)
            except Exception as e:
                log.warning(f"Error in VLM.generate (attempt {retry + 1}): {e}")
                retry += 1
                last_exception = e
        log.error("Max retries reached for VLM.generate.")
        if last_exception:
            raise RuntimeError(
                "Failed to generate after multiple retries"
            ) from last_exception
        return ""

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        schema: BaseModel = None,
    ) -> Dict:
        retry = 0
        max_retries = 3
        last_exception = None
        while retry < max_retries:
            try:
                return self.vlm.generate_json(prompt_or_memory, images, schema)
            except Exception as e:
                log.warning(f"Error in VLM.generate_json (attempt {retry + 1}): {e}")
                retry += 1
                last_exception = e
        log.error("Max retries reached for VLM.generate_json.")
        if last_exception:
            raise RuntimeError(
                "Failed to generate JSON after multiple retries"
            ) from last_exception
        return {}

    def batch_generate(
        self, queries: list, images_list: list = None, max_workers: int = 8
    ):
        if isinstance(self.vlm, QwenVLController):
            if len(queries) > 1:
                raise RuntimeError(
                    "QwenVLController does not support parallel batch inference in a single process."
                )
            return [self.generate(queries[0], images_list[0] if images_list else None)]
        results = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.generate, queries[i], images_list[i] if images_list else None
                ): i
                for i in range(len(queries))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Error: {e}"
        return results


if __name__ == "__main__":
    vlm_config = VLMConfig()
    vlm = VLM(vlm_config)
    # llm = LLM("Qwen/Qwen2.5-VL-7B-Instruct")
    # llm = LLM('gpt-4o')

    tmp_memory = Memory()
    query = (
        "Description this image in a senence, and then list the objects in the image."
    )
    sys_temp = "You are a helpful assistant that helps people find information."
    tmp_memory.add(Message(role="system", content=sys_temp))
    tmp_memory.add(Message(role="user", content=query))
    response = vlm.generate(
        prompt_or_memory=tmp_memory,
        images=[
            "/home/wangshu/multimodal/GBC-RAG/test/tree_index/images/8f4d58edc0302540d157aa54eaabfddf7534f4b407d4c811993b60372678a274.jpg"
        ],
    )
    print(response)
