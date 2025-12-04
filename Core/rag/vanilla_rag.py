import os

from zmq import ContextTerminated

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from Core.provider.vdb import VectorStore
from Core.provider.vlm import VLM
from Core.provider.llm import LLM
from Core.rag.base_rag import BaseRAG
from Core.configs.rag.vanilla_config import VanillaConfig
from Core.utils.bm25 import BM25
from Core.utils.utils import TextProcessor

from typing import Dict, Any, List
import json
import logging

log = logging.getLogger(__name__)


class VanillaRAG(BaseRAG):
    """
    Text-only Vanilla Retrieval Augmented Generation,
    supports vanilla, BM25, RAPTOR, PDF+Vanilla
    """

    def __init__(
        self,
        config: VanillaConfig,
        vector_store: VectorStore,
        llm: LLM,
        bm25: BM25 = None,
    ):
        super().__init__(
            llm=llm,
            name="MM RAG",
            description="Text-only Vanilla Retrieval Augmented Generation",
        )
        self.cfg = config
        self.max_tokens = self.llm.config.max_tokens - 200
        log.info("Vanilla RAG initialized.")
        self.topk = self.cfg.topk

        if self.cfg.retrieval_method == "bm25":
            self.bm25: BM25 = bm25
        else:
            self.vdb = vector_store

    def _retrieve(self, query: str, top_k: int = 3):
        if self.cfg.retrieval_method == "bm25":
            return self.bm25.search(query_text=query, top_k=top_k)
        else:
            return self.vdb.search(query_text=query, top_k=top_k)

    def _create_augmented_prompt(self, query: str, retrieved_docs=None) -> str:
        # context_text = "Please refer to the following background information to answer the question.\n\n--- Background Information ---\n"
        context_text = "Please refer to the following background information to answer the question. You should base your answer strictly on the provided information and not supplement it with outside knowledge. If the background information is insufficient to answer the question, please state that the provided information is not enough.\n\n--- Background Information ---\n"
        question_text = f"--- User Question ---\n{query}\n\n"
        context_text += question_text
        if retrieved_docs is None:
            context_text += "No relevant documents found.\n"
            return context_text

        context_text += "\n--- Retrieved Documents ---\n"
        for i, doc in enumerate(retrieved_docs):
            context_text += f"Text {i+1}: {doc['content']}\n"

        context_text = TextProcessor.split_text_into_chunks(
            text=context_text, max_length=self.max_tokens-400
        )
        context_text = context_text[0]  # take the fFirst chunk only
        return context_text

    def _save_retrieval_res(self, context_nodes, query_output_dir) -> List[Dict]:
        retrieval_ids = []
        for doc in context_nodes:
            if "metadata" in doc:
                meta = doc.get("metadata", {})
                if "node_id" in meta:
                    node_id = meta["node_id"]
                elif "chunk_id" in meta:
                    node_id = meta["chunk_id"]
                else:
                    node_id = meta.get("node_id", meta.get("id", None))
            elif "node_id" in doc:
                node_id = doc["node_id"]
            else:
                node_id = doc["id"]
            meta_info_dict = {
                "id": node_id,
                "content": doc["content"],
            }
            retrieval_ids.append(node_id)
            node_file_path = query_output_dir / f"{node_id}.json"
            with open(node_file_path, "w", encoding="utf-8") as f:
                json.dump(meta_info_dict, f, indent=2, ensure_ascii=False)

        log.info("Saved retrieval results to output directory.")

        return retrieval_ids

    def generation(self, query: str, query_output_dir: str) -> tuple:
        """
        Generates an answer for a given query and returns the answer along with the context used.
        Returns:
            Tuple[str, List[Any]]: A tuple containing the final answer string and a list of the context nodes.
        """
        retrieved_docs = self._retrieve(query, top_k=self.topk)
        if not retrieved_docs:
            # not found any relevant documents, fallback to LLM generation
            final_answer = self.llm.get_completion(query, json_response=False)
            return final_answer, []

        context_text = self._create_augmented_prompt(query, retrieved_docs)

        final_answer = self.llm.get_completion(context_text, json_response=False)

        retrieval_ids = self._save_retrieval_res(
            retrieved_docs, query_output_dir=query_output_dir
        )
        return final_answer, retrieval_ids

    def close(self):
        if self.cfg.retrieval_method == "bm25":
            log.info("Closing BM25 resources...")
            self.bm25.close()
        else:
            self.vdb.embedding_model.close()
