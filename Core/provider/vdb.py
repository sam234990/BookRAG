from Core.provider.embedding import BaseEmbedder

import chromadb
from typing import List, Dict, Any
import uuid
import logging

log = logging.getLogger(__name__)


class VectorStore:
    """
    支持多模态（如CLIP）和纯文本的向量存储和检索类。
    """

    def __init__(
        self,
        embedding_model: BaseEmbedder,  # 可以是ChineseClipModel或其他有embed_texts方法的模型
        db_path: str = "./chroma_db",
        collection_name: str = "multimodal_collection",
    ):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.metadata = {"hnsw:space": "cosine"}
        self.client = chromadb.PersistentClient(
            path=db_path, settings=chromadb.Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=self.metadata
        )
        log.info(f"ChromaDB collection '{collection_name}' loaded/created.")

    def reset(self):
        log.info(f"Resetting ChromaDB client and clearing all data...")
        self.client.reset()

        # 重新创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=self.metadata
        )
        log.info(f"Collection '{self.collection_name}' has been reset and is ready.")

    def add_texts(self, texts: List[str], metadatas: List[dict] = None):
        if not texts:
            return
        log.info(f"Adding {len(texts)} texts to the database...")
        embeddings = self.embedding_model.embed_texts(texts)
        ids = [f"text_{uuid.uuid4()}" for _ in texts]
        max_batch = 4096
        n = len(texts)
        for i in range(0, n, max_batch):
            batch_texts = texts[i : i + max_batch]
            batch_embeddings = embeddings[i : i + max_batch]
            batch_ids = ids[i : i + max_batch]
            batch_metadatas = (
                metadatas[i : i + max_batch] if metadatas else [{} for _ in batch_texts]
            )
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
        log.info("Texts added successfully.")
        return ids

    def delete_text_by_metadata(self, metadata: dict):
        """
        Deletes texts from the collection based on metadata.
        """
        if not metadata:
            return
        log.info(f"Deleting texts with metadata: {metadata}...")
        self.collection.delete(where=metadata)
        log.info("Delete by metadata finished.")
        # ChromaDB官方实现：如果没有匹配项，不会报错，只是没有任何数据被删除
        return

    def delete_text_by_ids(self, ids: List[str]):
        """
        Deletes texts from the collection based on their IDs.
        """
        if not ids:
            return
        log.info(f"Deleting texts with IDs: {ids}...")
        self.collection.delete(ids=ids)
        log.info("Delete by IDs finished.")
        # ChromaDB官方实现：如果没有匹配项，不会报错，只是没有任何数据被删除
        return

    def add_images(
        self,
        image_paths: List[str],
        metadatas: List[dict] = None,
        image_str: List[str] = None,
    ):
        if not image_paths:
            return
        if not self.embedding_model.MM_EMBEDDER:
            raise ValueError(
                "当前embedding_model不是多模态模型，无法添加图片。请使用支持图片向量化的模型。"
            )
        log.info(f"Adding {len(image_paths)} images to the database...")
        # if the embedder have embed_fused fuction, then use embed_fused
        # else use embed_images
        if hasattr(self.embedding_model, "embed_fused"):
            embeddings = self.embedding_model.embed_fused(
                images=image_paths, texts=image_str
            )
        else:
            embeddings = self.embedding_model.embed_images(image_paths)
        ids = [f"image_{uuid.uuid4()}" for _ in image_paths]
        self.collection.add(
            embeddings=embeddings,
            documents=image_paths,
            metadatas=(
                metadatas if metadatas else [{"type": "image"} for _ in image_paths]
            ),
            ids=ids,
        )
        log.info("Images added successfully.")
        return ids

    def search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Uses a text query to search for the most similar texts or images and returns the results.
        """
        # Step 1: Embed the query text
        query_embedding = self.embedding_model.embed_texts([query_text])
        # Step 2: Query the collection in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )
        # Step 3: Process and structure the results
        retrieved_results = []
        if results and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                result_item = {
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
                retrieved_results.append(result_item)
        return retrieved_results

    def batch_search(
        self, query_texts: List[str], top_k: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        批量搜索多个查询文本，返回每个查询的结果列表。
        """
        # 1. 批量embed
        query_embeddings = self.embedding_model.embed_texts(query_texts)
        # 2. 一次性批量query
        results = self.collection.query(
            query_embeddings=query_embeddings, n_results=top_k
        )
        # 3. 处理结果
        batch_results = []
        for i in range(len(query_texts)):
            single_result = []
            if results and results["ids"][i]:
                for j in range(len(results["ids"][i])):
                    result_item = {
                        "id": results["ids"][i][j],
                        "distance": results["distances"][i][j],
                        "content": results["documents"][i][j],
                        "metadata": results["metadatas"][i][j],
                    }
                    single_result.append(result_item)
            batch_results.append(single_result)
        return batch_results

