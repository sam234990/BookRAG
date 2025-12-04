import os
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from Core.provider.embedding import (
    TextEmbeddingProvider,
    GmeEmbeddingProvider,
)
from Core.provider.llm import LLM
from Core.provider.vdb import VectorStore
from Core.Index.Tree import DocumentTree, NodeType
from Core.configs.vdb_config import VDBConfig
from Core.configs.system_config import SystemConfig
from Core.utils.utils import TextProcessor
from Core.utils.raptor_utils import raptor_tree
from Core.utils.bm25 import BM25
import json
import logging

log = logging.getLogger(__name__)

save_path = "/home/wangshu/multimodal/GBC-RAG/test/sf/"


def process_tree_nodes(tree: DocumentTree) -> Tuple[Dict[str, List], Dict[str, List]]:
    text_list = []
    text_meta_data = []
    image_list = []
    image_meta_data = []
    image_str_list = []
    for node in tree.nodes:
        if node == tree.root_node:
            continue

        node_type = node.type
        meta_data = {
            "node_id": node.index_id,
            "pdf_id": node.meta_info.pdf_id,
        }

        if node_type == NodeType.IMAGE:
            image_path = node.meta_info.img_path
            image_str = node.meta_info.caption + node.meta_info.footnote
            text_list.append(image_str)
            text_meta_data.append(meta_data)

            # Check if the image path exists before adding it
            if image_path and os.path.exists(image_path):
                image_list.append(image_path)
                image_meta_data.append(meta_data)
                image_str_list.append(image_str)
        elif node_type == NodeType.TABLE:
            table_str = node.meta_info.content
            table_body = node.meta_info.table_body
            if table_body:
                table_str += table_body
            text_list.append(table_str)
            text_meta_data.append(meta_data)

            table_img = node.meta_info.img_path
            if table_img and os.path.exists(table_img):
                image_list.append(table_img)
                image_meta_data.append(meta_data)
                image_str_list.append(table_str)
        elif (
            node_type == NodeType.TEXT
            or node_type == NodeType.TITLE
            or node_type == NodeType.EQUATION
        ):
            text_content = node.meta_info.content
            if text_content:
                text_list.append(text_content)
                text_meta_data.append(meta_data)

    text_dict = {"text": text_list, "meta": text_meta_data}
    image_dict = {
        "image": image_list,
        "meta": image_meta_data,
        "image_str": image_str_list,
    }
    return text_dict, image_dict


def build_vdb_index(tree: DocumentTree, vdb_cfg: VDBConfig):
    if vdb_cfg.mm_embedding:
        embedder = GmeEmbeddingProvider(
            model_name=vdb_cfg.embedding_config.model_name,
            device=vdb_cfg.embedding_config.device,
        )
        log.info("Using GME multi-modal embedding model for vector database.")
    else:
        embedder = TextEmbeddingProvider(
            model_name=vdb_cfg.embedding_config.model_name,
            device=vdb_cfg.embedding_config.device,
            backend=vdb_cfg.embedding_config.backend,
            api_base=vdb_cfg.embedding_config.api_base,
            max_length=vdb_cfg.embedding_config.max_length,
        )
        log.info("Using text embedding model for vector database.")

    vdb = VectorStore(
        embedding_model=embedder,
        db_path=vdb_cfg.vdb_dir_name,
        collection_name=vdb_cfg.collection_name,
    )

    text_dict, image_dict = process_tree_nodes(tree)

    text, text_meta = text_dict["text"], text_dict["meta"]
    vdb.add_texts(texts=text, metadatas=text_meta)

    mm_vdb = vdb_cfg.mm_embedding
    if mm_vdb is True:
        image, img_meta, img_str = (
            image_dict["image"],
            image_dict["meta"],
            image_dict["image_str"],
        )
        vdb.add_images(image_paths=image, metadatas=img_meta, image_str=img_str)
        log.info("Images added to vector database successfully.")

    log.info("Vector database index built successfully.")

    vdb.embedding_model.close()  # Close the embedding model to free resources
    return


def get_input_text(cfg: SystemConfig) -> str:
    pdf_path = cfg.pdf_path
    file_name = str(Path(pdf_path).stem)
    md_file = os.path.join(cfg.save_path, "vlm", f"{file_name}.md")
    with open(md_file, "r", encoding="utf-8") as f:
        input_text = f.read()
    return input_text


def get_all_chunks(cfg: SystemConfig):
    corpus_text = get_input_text(cfg)
    chunks = TextProcessor.split_text_into_chunks(text=corpus_text, max_length=500)

    index_type = cfg.index_type
    if index_type == "vanilla" or index_type == "bm25":
        meta_datas = [{"source": "document", "chunk_id": i} for i in range(len(chunks))]
        return chunks, meta_datas
    elif index_type == "raptor":
        llm = LLM(cfg.llm)
        embed_cfg = cfg.vdb.embedding_config
        embedder = TextEmbeddingProvider(
            model_name=embed_cfg.model_name,
            device=embed_cfg.device,
            backend=embed_cfg.backend,
            api_base=embed_cfg.api_base,
            max_length=embed_cfg.max_length,
        )

        all_tree_text, all_meta_data = raptor_tree(chunks, embedder=embedder, llm=llm)
        embedder.close()
        return all_tree_text, all_meta_data


def build_other_vdb_index(cfg: SystemConfig):
    vdb_dir = os.path.join(cfg.save_path, cfg.vdb.vdb_dir_name)
    if os.path.exists(vdb_dir) and not cfg.vdb.force_rebuild:
        if cfg.vdb.force_rebuild:
            import shutil

            shutil.rmtree(vdb_dir)
            log.info(
                f"Vector database path already exists: {vdb_dir}. Remove and rebuild"
            )
        else:
            log.info(f"Vector database path already exists: {vdb_dir}. Skip")
            return

    os.makedirs(vdb_dir, exist_ok=True)
    all_chunks, meta_datas = get_all_chunks(cfg)

    if cfg.index_type == "bm25":
        save_path = os.path.join(vdb_dir, "bm25_index.pkl")
        bm25 = BM25(all_chunks)
        bm25.initialize()
        # test
        query = "quick"
        results = bm25.search(query, top_k=2)
        log.info(f"BM25 search results for test query {query}: {results}")

        bm25.save(save_path)
        log.info(f"BM25 index saved to {save_path}")
    else:
        vdb_config = cfg.vdb
        vdb = VectorStore(
            embedding_model=TextEmbeddingProvider(
                model_name=vdb_config.embedding_config.model_name,
                device=vdb_config.embedding_config.device,
                backend=vdb_config.embedding_config.backend,
                api_base=vdb_config.embedding_config.api_base,
                max_length=vdb_config.embedding_config.max_length,
            ),
            db_path=vdb_dir,
            collection_name=vdb_config.collection_name,
        )
        vdb.add_texts(texts=all_chunks, metadatas=meta_datas)
        log.info("Vector database index built successfully.")
        vdb.embedding_model.close()  # Close the embedding model to free resources


def load_pdf_lists_from_dir(save_dir):
    res_list = []
    pdf_list_json_files = os.listdir(save_dir)
    for pdf_list_json_file in pdf_list_json_files:
        if not pdf_list_json_file.endswith(".json"):
            continue
        pdf_list_path = os.path.join(save_dir, pdf_list_json_file)
        with open(pdf_list_path, "r", encoding="utf-8") as f:
            pdf_list = json.load(f)
        tmp_dict = {"pdf_list": pdf_list, "pdf_list_path": pdf_list_path}
        res_list.append(tmp_dict)

    return res_list


def compute_mm_embedding(cfg: SystemConfig, tree_index: DocumentTree):
    embedder_cfg = cfg.vdb.embedding_config
    embedder = GmeEmbeddingProvider(
        model_name=embedder_cfg.model_name,
        device=embedder_cfg.device,
    )

    text_only_group_data = []
    image_only_group_data = []
    fused_group_data = []

    all_node_data = []
    all_embeddings_values = []

    for i, node in enumerate(tree_index.nodes):
        if node == tree_index.root_node:
            continue

        node_id = node.index_id
        node_type = node.type
        content = node.meta_info.content
        img_path = (
            node.meta_info.img_path
            if (node_type == NodeType.IMAGE or node_type == NodeType.TABLE)
            else None
        )

        node_info = {
            "node_id": node_id,
            "node_type": node_type,
            "content": content,
            "img_path": img_path,
            "embedding_idx": None,
        }
        current_node_data_idx = len(all_node_data)
        all_node_data.append(node_info)

        if content and img_path:
            fused_group_data.append(
                {
                    "original_node_data_idx": current_node_data_idx,
                    "text": content,
                    "image": img_path,
                }
            )
        elif content:
            text_only_group_data.append(
                {"original_node_data_idx": current_node_data_idx, "text": content}
            )
        elif img_path:
            image_only_group_data.append(
                {"original_node_data_idx": current_node_data_idx, "image": img_path}
            )

    if text_only_group_data:
        texts = [item["text"] for item in text_only_group_data]
        text_embeddings = embedder.embed_texts(texts)
        for i, item in enumerate(text_only_group_data):
            original_node_data_idx = item["original_node_data_idx"]
            embedding = text_embeddings[i]

            embedding_idx = len(all_embeddings_values)
            all_embeddings_values.append(embedding)

            all_node_data[original_node_data_idx]["embedding_idx"] = embedding_idx

    if image_only_group_data:
        images = [item["image"] for item in image_only_group_data]
        image_embeddings = embedder.embed_images(images)
        for i, item in enumerate(image_only_group_data):
            original_node_data_idx = item["original_node_data_idx"]
            embedding = image_embeddings[i]

            embedding_idx = len(all_embeddings_values)
            all_embeddings_values.append(embedding)

            all_node_data[original_node_data_idx]["embedding_idx"] = embedding_idx

    if fused_group_data:
        texts = [item["text"] for item in fused_group_data]
        images = [item["image"] for item in fused_group_data]
        fused_embeddings = embedder.embed_fused(texts=texts, images=images)
        for i, item in enumerate(fused_group_data):
            original_node_data_idx = item["original_node_data_idx"]
            embedding = fused_embeddings[i]

            embedding_idx = len(all_embeddings_values)
            all_embeddings_values.append(embedding)

            all_node_data[original_node_data_idx]["embedding_idx"] = embedding_idx
    embedder.clear_cache()

    # --- 保存所有节点元数据到JSON文件 ---
    save_dir = cfg.save_path
    os.makedirs(save_dir, exist_ok=True)  # 确保保存路径存在
    metadata_filepath = os.path.join(save_dir, "mm_node_metadata.json")
    embeddings_filepath = os.path.join(save_dir, "mm_embeddings.npy")

    with open(metadata_filepath, "w", encoding="utf-8") as f:
        json.dump(all_node_data, f, ensure_ascii=False, indent=4)

    if all_embeddings_values:
        final_embeddings_array = np.array(all_embeddings_values)
        np.save(embeddings_filepath, final_embeddings_array)
        log.info(f"All embeddings saved to: {embeddings_filepath}")
    else:
        log.warning("No embeddings were computed, .npy file not saved.")

    log.info(f"All node metadata saved to: {metadata_filepath}")


def compute_mm_embedding_question(cfg: SystemConfig, group: pd.DataFrame):
    embedder_cfg = cfg.vdb.embedding_config
    embedder = GmeEmbeddingProvider(
        model_name=embedder_cfg.model_name,
        device=embedder_cfg.device,
    )

    group_dedup = group.drop_duplicates(subset=["question"], keep="first")
    questions = group_dedup["question"].tolist()
    RERANKER_INSTRUCTION = "Retrieve the most relevant document for the given query."

    # add instruction for gme model
    question_embeddings_raw = embedder.embed_texts(
        questions, instruction=RERANKER_INSTRUCTION
    )

    all_question_embeddings = []
    question_embedding_indices = []

    for i, embedding in enumerate(question_embeddings_raw):
        all_question_embeddings.append(embedding)
        question_embedding_indices.append(len(all_question_embeddings) - 1)

    group_dedup["question_embedding_idx"] = question_embedding_indices

    save_dir = cfg.save_path
    os.makedirs(save_dir, exist_ok=True)

    question_metadata_filepath = os.path.join(save_dir, "mm_question_metadata.json")
    question_embeddings_filepath = os.path.join(save_dir, "mm_question_embeddings.npy")

    group_dedup.to_json(
        question_metadata_filepath, orient="records", force_ascii=False, indent=4
    )

    if all_question_embeddings:
        final_question_embeddings_array = np.array(all_question_embeddings)
        np.save(question_embeddings_filepath, final_question_embeddings_array)
        log.info(f"All question embeddings saved to: {question_embeddings_filepath}")
    else:
        log.warning("No question embeddings were computed, .npy file not saved.")

    log.info(f"All question metadata saved to: {question_metadata_filepath}")


if __name__ == "__main__":
    # tmp_tree_path = f"{save_path}/sftree.pkl"
    # tree_index = DocumentTree.load_from_file(tmp_tree_path)
    # print(f"Loaded tree index from: {tmp_tree_path}")
    # vector_store = build_vdb_from_tree(tree_index)
    print("test")
