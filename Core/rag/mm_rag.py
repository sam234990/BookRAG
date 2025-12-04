import os
from Core.provider.vdb import VectorStore
from Core.provider.vlm import VLM
from Core.provider.llm import LLM
from Core.provider.embedding import GmeEmbeddingProvider
from Core.rag.base_rag import BaseRAG
from Core.configs.rag.mm_config import MMConfig
from typing import Dict, Any, List
import json
import logging

log = logging.getLogger(__name__)


class MMRAG(BaseRAG):
    """
    一个封装了检索、增强、生成完整流程的多模态RAG流水线。
    """

    def __init__(
        self,
        config: MMConfig,
        vector_store: VectorStore,
        llm: LLM,
        vlm: VLM,
        topk: int = 3,
    ):
        super().__init__(
            llm=llm,
            name="MM RAG",
            description="Multimodal Vanilla Retrieval Augmented Generation",
        )
        self.cfg = config
        self.vlm = vlm
        self.vdb = vector_store
        log.info("MultimodalRAGPipeline initialized.")
        self.topk = topk

    def _retrieve(self, query: str, top_k: int = 3):
        return self.vdb.search(query_text=query, top_k=top_k)

    def _create_augmented_prompt(self, query: str, retrieved_docs=None) -> str:
        context_text = "Please refer to the following background information to answer the question.\n\n--- Background Information ---\n"
        context_images = []
        question_text = f"--- User Question ---\n{query}\n\n"
        if retrieved_docs is None:
            context_text += "No relevant documents found.\n"
            context_text += question_text
            return context_text, context_images

        for i, doc in enumerate(retrieved_docs):
            content_type = doc["metadata"].get("type", "text")
            if content_type == "image":
                image_path = doc["content"]
                if os.path.exists(image_path):
                    context_images.append(image_path)
                    context_text += f"Image {i+1}: A relevant image is provided at the path: {image_path}\n"
            else:
                context_text += f"Text {i+1}: {doc['content']}\n"

        context_text += question_text

        return context_text, context_images

    def _save_retrieval_res(self, context_nodes, query_output_dir) -> List[Dict]:
        retrieval_ids = []
        for doc in context_nodes:
            content_type = doc["metadata"].get("type", "text")
            if content_type not in ["text", "image"]:
                log.warning(
                    f"Unsupported content type: {content_type}. Skipping this document."
                )
                continue
            node_id = doc["metadata"].get("node_id", -1)

            meta_info_dict = {
                "node_id": node_id,
                "type": content_type,
                "content": doc["content"],
            }
            if content_type == "image":
                img_path = doc["content"]
                meta_info_dict["img_path"] = img_path

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

        context_text, context_images = self._create_augmented_prompt(
            query, retrieved_docs
        )

        if len(context_images) > 0:
            # if there are images, use VLM to generate the answer
            if len(context_images) > 2:
                # VLM only support max 2 image input
                context_images = context_images[:2]
            final_answer = self.vlm.generate(
                prompt_or_memory=context_text, images=context_images
            )
        else:
            # if no images, fallback to LLM generation
            final_answer = self.llm.get_completion(context_text, json_response=False)
        retrieval_ids = self._save_retrieval_res(
            retrieved_docs, query_output_dir=query_output_dir
        )
        return final_answer, retrieval_ids

    def run(self, query: str) -> Dict[str, Any]:
        answer, retrieved_docs = self.generation(query)
        return {"answer": answer, "retrieved_docs": retrieved_docs}

    def close(self):
        # self.vdb.embedding_model.close()
        if isinstance(self.vdb.embedding_model, GmeEmbeddingProvider):
            self.vdb.embedding_model.clear_cache()
            log.info("Cleared GmeEmbeddingProvider cache.")
        log.info("MultimodalRAGPipeline resources have been released.")
