import os

import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import math
from tqdm import tqdm
import json

from abc import ABC, abstractmethod

from modelscope import AutoTokenizer, AutoModel
from transformers import AutoModel as transformer_AutoModel
import ollama
import openai
import torch.nn.functional as F
import logging
import gc

log = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    MM_EMBEDDER: bool = False

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def close(self) -> None:
        """释放模型和相关资源。"""
        pass


class GmeEmbeddingProvider(BaseEmbedder):
    """ """

    MM_EMBEDDER: bool = True

    _instance = None  # 类变量，用于存储唯一的实例

    def __new__(cls, *args, **kwargs):
        # 如果实例还不存在，就创建一个新的
        if cls._instance is None:
            log.info("Creating a new GmeEmbeddingProvider instance...")
            cls._instance = super(GmeEmbeddingProvider, cls).__new__(cls)
            # 标记一下，确保 __init__ 只被调用一次
            cls._instance._initialized = False
        else:
            log.info("Returning existing GmeEmbeddingProvider instance.")
        return cls._instance

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device: str = "auto",
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_name = model_name
        self.device_strategy = device.lower()

        log.info(
            f"Using local backend (ModelScope) with device strategy: {self.device_strategy}"
        )
        log.info(f"Loading model: {self.model_name}...")

        try:
            self.model = transformer_AutoModel.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map=self.device_strategy,
                use_fast=True,
            )
            self.model.eval()
            log.info("GME model loaded successfully.")
            self._initialized = True  # 标记为已初始化
        except Exception as e:
            log.error(
                f"Error loading GME model '{self.model_name}' from ModelScope.",
                exc_info=True,
            )
            raise e

    def _normalize(self, embeddings: torch.Tensor) -> np.ndarray:
        """对PyTorch张量进行L2归一化，并转换为numpy数组。"""
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        return normalized_embeddings.cpu().numpy()

    def embed_texts(
        self, texts: List[str], instruction: Optional[str] = None, batch_size=4
    ) -> np.ndarray:
        """
        为一批文本生成嵌入向量。

        Args:
            texts (List[str]): 需要编码的文本列表。
            instruction (Optional[str]): 用于指导文本嵌入生成的指令。如果为None，则使用模型的默认指令。

        Returns:
            np.ndarray: 返回一个 numpy 数组，形状为 (n_texts, embedding_dim)。
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input 'texts' must be a non-empty list of strings.")

        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Texts"):
                batch_texts = texts[i : i + batch_size]

                embeddings_tensor = self.model.get_text_embeddings(
                    texts=batch_texts, instruction=instruction
                )
                all_embeddings.append(embeddings_tensor.cpu())

        # 将所有批次的结果拼接成一个Tensor
        full_embeddings_tensor = torch.cat(all_embeddings, dim=0)

        return self._normalize(full_embeddings_tensor)

    def embed_images(self, images: List[str], batch_size: int = 4) -> np.ndarray:
        """
        为一批图像生成嵌入向量（已分批处理）。

        Args:
            images (List[str]): 需要编码的图像路径列表。
            batch_size (int): 每个批次处理的图像数量，默认为16。

        Returns:
            np.ndarray: 返回一个 numpy 数组，形状为 (n_images, embedding_dim)。
        """
        if not images or not isinstance(images, list):
            raise ValueError("Input 'images' must be a non-empty list of image paths.")

        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Embedding Images"):
                batch_images = images[i : i + batch_size]

                embeddings_tensor = self.model.get_image_embeddings(images=batch_images)
                all_embeddings.append(embeddings_tensor.cpu())

        full_embeddings_tensor = torch.cat(all_embeddings, dim=0)

        return self._normalize(full_embeddings_tensor)

    def embed_fused(
        self, images: List[str], texts: List[str], batch_size: int = 4
    ) -> np.ndarray:
        """
        为一一对应的文本-图像对生成融合嵌入向量（已分批处理）。

        Args:
            texts (List[str]): 文本列表。
            images (List[str]): 图像路径列表。两个列表长度必须相等。
            batch_size (int): 每个批次处理的图文对数量，默认为4

        Returns:
            np.ndarray: 返回一个 numpy 数组，形状为 (n_pairs, embedding_dim)。
        """
        if not texts or not images or len(texts) != len(images):
            raise ValueError(
                "'texts' and 'images' must be non-empty lists of the same length."
            )

        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Fused"):
                batch_texts = texts[i : i + batch_size]
                batch_images = images[i : i + batch_size]

                embeddings_tensor = self.model.get_fused_embeddings(
                    texts=batch_texts, images=batch_images
                )
                all_embeddings.append(embeddings_tensor.cpu())

        full_embeddings_tensor = torch.cat(all_embeddings, dim=0)

        return self._normalize(full_embeddings_tensor)

    def close(self) -> None:
        """关闭GmeEmbeddingProvider并释放资源。"""
        log.info(f"Closing GmeEmbeddingProvider for model: {self.model_name}...")

        if hasattr(self, "model"):
            log.info("Releasing GME model resources...")
            del self.model

            # 如果模型是加载到CUDA上的，清空缓存
            if torch.cuda.is_available():
                log.info("Embedder: Emptying CUDA cache.")
                torch.cuda.empty_cache()

            gc.collect()
            log.info("GME model resources released.")

        log.info("GmeEmbeddingProvider closed.")

    def clear_cache(self) -> None:
        """
        手动清理PyTorch的CUDA缓存和Python的垃圾回收。
        注意：这是一个同步操作，可能会影响性能，请仅在必要时（如处理完一个大任务后）调用。
        """
        log.info("Clearing CUDA cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log.info("CUDA cache cleared.")

    def rerank_documents(
        self,
        query: str,
        doc_list: List[Dict[str, str]],
        instruction: Optional[str] = None,
    ) -> List[float]:
        """
        Rerank a list of documents based on a query and an optional instruction.
        """
        if not query:
            raise ValueError("Query cannot be empty.")
        if not doc_list:
            return []

        log.info(
            f"Reranking {len(doc_list)} documents based on query and instruction..."
        )

        query_embedding = self.embed_texts([query], instruction=instruction)[0]

        # 2. 按类型对文档进行分组，同时记录原始索引
        text_only_group = []
        image_only_group = []
        fused_group = []

        for i, doc in enumerate(doc_list):
            text = doc.get("content")
            image = doc.get("img_path")

            if text and image:
                fused_group.append({"index": i, "text": text, "image": image})
            elif text:
                text_only_group.append({"index": i, "text": text})
            elif image:
                image_only_group.append({"index": i, "image": image})

        # 初始化最终分数列表
        scores = [0.0] * len(doc_list)

        # 3. 对每个分组进行批量嵌入和相似度计算
        if text_only_group:
            log.info(f"Processing {len(text_only_group)} text-only documents...")
            texts = [item["text"] for item in text_only_group]
            doc_embeddings = self.embed_texts(texts)
            similarities = (doc_embeddings @ query_embedding.T).tolist()
            for i, item in enumerate(text_only_group):
                scores[item["index"]] = similarities[i]

        if image_only_group:
            log.info(f"Processing {len(image_only_group)} image-only documents...")
            images = [item["image"] for item in image_only_group]
            doc_embeddings = self.embed_images(images)
            similarities = (doc_embeddings @ query_embedding.T).tolist()
            for i, item in enumerate(image_only_group):
                scores[item["index"]] = similarities[i]

        if fused_group:
            log.info(f"Processing {len(fused_group)} fused (text+image) documents...")
            texts = [item["text"] for item in fused_group]
            images = [item["image"] for item in fused_group]
            doc_embeddings = self.embed_fused(texts=texts, images=images)
            similarities = (doc_embeddings @ query_embedding.T).tolist()
            for i, item in enumerate(fused_group):
                scores[item["index"]] = similarities[i]

        log.info("Reranking complete.")
        return scores


class MMRerankerProvider(BaseEmbedder):
    MM_EMBEDDER: bool = True

    def __init__(
        self,
        index_path: str,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device: str = "auto",
    ):
        self.index_path = index_path
        self.doc_metadata = {}
        self.doc_embeddings = np.array([])
        self.question_embeddings = {}

        log.info(f"Initializing MMRerankerProvider from path: {self.index_path}")
        self._load_embeddings()

        self.gme_provider = GmeEmbeddingProvider(
            model_name=model_name,
            device=device,
        )

    def _load_embeddings(self):
        """
        Loads metadata and embeddings for documents and questions from disk.
        """
        # --- Load Document Embeddings ---
        doc_metadata_path = os.path.join(self.index_path, "mm_node_metadata.json")
        doc_embeddings_path = os.path.join(self.index_path, "mm_embeddings.npy")

        if os.path.exists(doc_metadata_path) and os.path.exists(doc_embeddings_path):
            log.info("Loading document metadata and embeddings...")
            with open(doc_metadata_path, "r", encoding="utf-8") as f:
                doc_metadata_list = json.load(f)

            # Create a dictionary for fast O(1) lookup by node_id
            self.doc_metadata = {item["node_id"]: item for item in doc_metadata_list}

            self.doc_embeddings = np.load(doc_embeddings_path)
            log.info(
                f"Loaded {len(self.doc_metadata)} document metadata entries and {self.doc_embeddings.shape[0]} document embeddings."
            )
        else:
            log.warning("Document metadata or embedding files not found")

        # --- Load Question Embeddings ---
        q_metadata_path = os.path.join(self.index_path, "mm_question_metadata.json")
        q_embeddings_path = os.path.join(self.index_path, "mm_question_embeddings.npy")

        if os.path.exists(q_metadata_path) and os.path.exists(q_embeddings_path):
            log.info("Loading question metadata and embeddings...")
            # Use pandas to easily read the json records
            q_metadata_df = pd.read_json(q_metadata_path, orient="records")
            q_embeddings = np.load(q_embeddings_path)

            # Create a dictionary mapping question text to its embedding for fast lookup
            for _, row in q_metadata_df.iterrows():
                question_text = row["question"]
                embedding_idx = row["question_embedding_idx"]
                self.question_embeddings[question_text] = q_embeddings[embedding_idx]
                # The query key must include the instruction, as it was used during pre-computation
                # query_key = question_text + self.RERANKER_INSTRUCTION
                # self.question_embeddings[query_key] = q_embeddings[embedding_idx]

            log.info(
                f"Loaded {len(self.question_embeddings)} pre-computed question embeddings."
            )
        else:
            log.warning("Question metadata or embedding files not found.")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError("MMRerankerProvider does not support text embedding.")

    def embed_images(self, images: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "MMRerankerProvider does not support image embedding."
        )

    def embed_fused(self, images: List[str], texts: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "MMRerankerProvider does not support fused embedding."
        )

    def rerank_documents(
        self,
        query: str,
        doc_list: List[Dict[str, str]],
        instruction: Optional[str] = None,
    ) -> List[float]:
        if not query:
            raise ValueError("Query cannot be empty.")
        if not doc_list:
            return []

        # 1. Get the pre-computed query embedding
        if query in self.question_embeddings.keys():
            query_embedding = self.question_embeddings[query]
        else:
            query_embedding = self.gme_provider.embed_texts(
                [query], instruction=instruction
            )[0]
            self.gme_provider.clear_cache()

        if query_embedding is None:
            log.error(
                f"Query '{query}' not found in pre-computed question embeddings. Returning zero scores."
            )
            return [0.0] * len(doc_list)

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        if self.doc_embeddings.size == 0:
            embedding_dim = query_embedding.shape[
                0
            ]  # Get dimension from query embedding
        else:
            embedding_dim = self.doc_embeddings.shape[1]

        # 2. Prepare document embeddings for batch calculation
        # Create a matrix to hold all document embeddings. Initialize with zeros.
        doc_embeddings_matrix = np.zeros(
            (len(doc_list), embedding_dim), dtype=query_embedding.dtype
        )

        # Collect valid embedding indices and their corresponding positions in doc_list
        valid_doc_indices = []
        valid_embedding_indices = []

        for i, doc in enumerate(doc_list):
            node_id = doc.get("node_id")
            metadata = self.doc_metadata.get(node_id)

            if metadata and metadata.get("embedding_idx") is not None:
                # skip documents without valid embedding
                embedding_idx_in_storage = metadata["embedding_idx"]
                valid_doc_indices.append(i)  # Original position in doc_list
                valid_embedding_indices.append(embedding_idx_in_storage)

        if valid_doc_indices:
            doc_embeddings_matrix[valid_doc_indices] = self.doc_embeddings[
                valid_embedding_indices
            ]

        # 3. Perform batch similarity calculation
        similarities = (doc_embeddings_matrix @ query_embedding).tolist()

        log.info(f"Successfully reranked {len(doc_list)} documents in batch mode.")
        return similarities

    def close(self) -> None:
        """
        Releases resources. In this implementation, data is in memory, so no action is needed.
        """
        log.info("MMRerankerProvider does not hold resources to release.")
        self.doc_metadata.clear()
        self.doc_embeddings = np.array([])
        self.question_embeddings.clear()


class TextEmbeddingProvider(BaseEmbedder):

    MM_EMBEDDER: bool = False

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        backend: str = "local",
        device: str = "auto",
        max_length: int = 8192,
        api_base: str = None,
    ):

        self.model_name = model_name
        self.backend = backend.lower()
        self.max_length = max_length

        if self.backend == "local":
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            log.info(f"Using local backend (ModelScope) on device: {self.device}")
            log.info(f"Loading model: {self.model_name}...")

            try:
                # 根据官方示例，使用AutoTokenizer和AutoModel，并设置padding_side
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, padding_side="left", trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=True
                ).to(self.device)
                self.model.eval()
            except Exception as e:
                log.info(f"Error loading model '{self.model_name}' from Hugging Face.")
                raise e

            log.info("Local model loaded successfully.")

        elif self.backend == "ollama":
            self.device = "ollama_service"
        elif self.backend == "openai":
            self.client = openai.OpenAI(api_key="empty", base_url=api_base)
        else:
            raise ValueError(
                f"Unsupported backend: '{self.backend}'. Choose 'local' or 'ollama'."
            )

    @classmethod
    def close_instance(cls) -> None:
        """
        作为类方法，用于在程序结束时关闭单例并释放资源。
        """
        if cls._instance is not None:
            log.info(
                f"Closing GmeEmbeddingProvider singleton for model: {cls._instance.model_name}..."
            )

            if hasattr(cls._instance, "model"):
                log.info("Releasing GME model resources...")
                del cls._instance.model

                if torch.cuda.is_available():
                    log.info("Embedder: Emptying CUDA cache.")
                    torch.cuda.empty_cache()

                gc.collect()
                log.info("GME model resources released.")

            cls._instance = None  # 销毁实例
            log.info("GmeEmbeddingProvider singleton instance closed.")

    # (旧的 close 方法可以移除或保留，但不再建议实例级别调用)
    def close(self) -> None:
        """实例级别的 close 方法现在应提醒用户使用类方法。"""
        log.warning(
            "Calling close() on a singleton instance is discouraged. Use GmeEmbeddingProvider.close_instance() at application shutdown."
        )
        # 或者直接调用类方法： GmeEmbeddingProvider.close_instance()

    def _last_token_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def _normalize(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        对PyTorch张量进行L2归一化，并转换为numpy数组。
        """
        # 使用torch.nn.functional.normalize进行归一化
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        # 移动到CPU并转换为numpy数组
        return normalized_embeddings.cpu().numpy()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        为一批文本生成嵌入向量。

        Args:
            texts (List[str]): 需要编码的文本列表。

        Returns:
            np.ndarray: 返回一个 numpy 数组，形状为 (n_texts, embedding_dim)。
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input 'texts' must be a non-empty list of strings.")

        if self.backend == "local":
            batch_size = 16
            all_embeddings = []
            n = len(texts)
            if n <= batch_size:
                batch_dict = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_dict.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                embeddings_tensor = self._last_token_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )
                all_embeddings.append(embeddings_tensor.cpu())
                embeddings_tensor = torch.cat(all_embeddings, dim=0)
            else:
                num_batches = math.ceil(n / batch_size)
                for i in tqdm(
                    range(0, n, batch_size),
                    desc="Embedding batches",
                    total=num_batches,
                ):
                    batch_texts = texts[i : i + batch_size]
                    batch_dict = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    batch_dict.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**batch_dict)
                    embeddings_tensor = self._last_token_pool(
                        outputs.last_hidden_state, batch_dict["attention_mask"]
                    )
                    all_embeddings.append(embeddings_tensor.cpu())
                embeddings_tensor = torch.cat(all_embeddings, dim=0)

        elif self.backend == "ollama":
            # --- Ollama 推理路径 ---
            all_embeddings = []
            for text in texts:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                all_embeddings.append(response["embedding"])
            embeddings_np = np.array(all_embeddings, dtype=np.float32)
            embeddings_tensor = torch.from_numpy(embeddings_np).to("cpu")  # 确保在CPU上

        elif self.backend == "openai":
            BATCH_SIZE = 8
            n = len(texts)
            all_embeddings = []
            num_batches = math.ceil(n / BATCH_SIZE)
            for i in tqdm(
                range(0, n, BATCH_SIZE), desc="Embedding texts", total=num_batches
            ):
                chunk = texts[i : i + BATCH_SIZE]
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=chunk,
                )
                embeddings_from_this_batch = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings_from_this_batch)

            embeddings_np = np.array(all_embeddings, dtype=np.float32)
            embeddings_tensor = torch.from_numpy(embeddings_np).to("cpu")  # 确保在CPU上

        # 对来自任何后端的输出都进行归一化
        return self._normalize(embeddings_tensor)

    def compute_texts_sim(self, text1: str, text2: str) -> float:
        """
        Args:
            text1 (str): The first text string.
            text2 (str): The second text string.
        Returns:
            float: Cosine similarity between the two text embeddings, in the range [-1.0, 1.0].
        """
        embeddings = self.embed_texts([text1, text2])
        vec1, vec2 = embeddings[0], embeddings[1]
        similarity = vec1 @ vec2

        return float(similarity)


if __name__ == "__main__":
    # Example usage
    text_embedder = TextEmbeddingProvider(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        # model_name="Qwen/Qwen3-Embedding-4B",
        backend="local",
        device="cuda:6",
    )
    texts = ["你好，世界！", "这是一个测试。"]
    embeddings = text_embedder.embed_texts(texts)
    print("Text embeddings shape:", embeddings.shape)
    # test the cuda storage

    mm_embedder = GmeEmbeddingProvider(
        model_name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device="cuda:6",
    )
    texts = ["你好，世界！", "这是一个测试。"]
    images = [
        "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/vlm/images/03fc225967e86ac2ca3f1dc1afe2f2657bf2298c14ad952bbf6f931574888e3c.jpg",
        "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/vlm/images/76f83ac5b4a7fd38c31a353466fec43927b5fed31f433753e8ece01795097112.jpg",
    ]
    text_embeddings = mm_embedder.embed_texts(texts)
    image_embeddings = mm_embedder.embed_images(images)
    print("Text embeddings shape:", text_embeddings.shape)
    print("Image embeddings shape:", image_embeddings.shape)
    fused_embeddings = mm_embedder.embed_fused(images=images, texts=texts)
    print("Fused embeddings shape:", fused_embeddings.shape)

    from Core.prompts.gbc_prompt import MM_RERANKER_INSTRUCTION

    docs = [
        {
            "content": "Table 1: The inference seconds over 500 sentences (run on single V100 GPU). Here LLaMA is extremely slow since we set batch size as 1 due to memory limit. <table><tr><td>Dataset (Task)</td><td>Roberta</td><td>T5</td><td>LLaMA</td><td>CODEX</td></tr><tr><td>FewNERD (NER)</td><td>2.8</td><td>39.4</td><td>1135.4</td><td>179.4</td></tr><tr><td>TACREV (RE)</td><td>1.4</td><td>45.6</td><td>1144.9</td><td>151.6</td></tr><tr><td>ACE05 (ED)</td><td>6.6</td><td>62.5</td><td>733.4</td><td>171.7</td></tr></table>",
            "img_path": "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/vlm/images/76f83ac5b4a7fd38c31a353466fec43927b5fed31f433753e8ece01795097112.jpg",
        },
        {
            "content": "Table 2: Comparative ratios of negative to positive samples across various datasets and subsets. We set fixed threshold  $\\tau$  here for simplicity.",
            "img_path": "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/vlm/images/34eaca8085bae127fb28d014cc4951fec10866cb37e58c65f4906989a5a1806a.jpg",
        },
        {
            "content": "The study highlights the significant impact of climate change on global food security, emphasizing rising temperatures, shifting precipitation patterns, and increased frequency of extreme weather events as major threats to agricultural productivity. It also underscores the vulnerability of smallholder farmers and the urgent need for adaptive strategies, including sustainable farming practices, improved water management, and investment in resilient crop varieties. The findings call for immediate and coordinated global action to mitigate risks and ensure long-term food stability."
        },
    ]
    query = "What is the inference time of LLaMA on FewNERD dataset?"
    rerank_scores = mm_embedder.rerank_documents(
        query=query,
        doc_list=docs,
        instruction=MM_RERANKER_INSTRUCTION,
    )
    print("Rerank scores:", rerank_scores)

    # clip_embedder = ChineseClipModel(model_name="ViT-B-16")
    # image_embeddings = clip_embedder.embed_images(["path/to/image1.jpg", "path/to/image2.jpg"])
    # print("Image embeddings shape:", image_embeddings.shape)
