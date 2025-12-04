from typing import Dict, Any
from Core.configs.system_config import SystemConfig
import logging


log = logging.getLogger(__name__)


def prepare_rag_dependencies(cfg: SystemConfig) -> Dict[str, Any]:
    """
    根据配置加载并准备RAG agent所需的依赖项。
    这是一个调度函数，它知道哪种策略需要哪种资源。
    """

    rag_config = cfg.rag.strategy_config
    strategy_name = rag_config.strategy
    log.info(f"Preparing dependencies for RAG strategy: '{strategy_name}'")

    dependencies = {}

    if strategy_name == "traverse":
        from Core.Index.Tree import DocumentTree

        # 加载 TraverseAgent 需要的 tree_index
        tree_index_path = DocumentTree.get_save_path(cfg.save_path)
        tree_index = DocumentTree.load_from_file(tree_index_path)
        log.info(f"Successfully loaded tree index from {tree_index_path}")
        dependencies["tree_index"] = tree_index

    elif strategy_name == "gbc":
        from Core.Index.GBCIndex import GBC

        gbc_index = GBC.load_gbc_index(cfg)
        log.info(f"Successfully loaded GBC index from {cfg.save_path}")
        dependencies["gbc_index"] = gbc_index
    elif strategy_name == "graph":
        from Core.Index.GBCIndex import GBC

        gbc_index = GBC.load_gbc_index(cfg)
        log.info(f"Successfully loaded GBC index from {cfg.save_path}")
        dependencies["gbc_index"] = gbc_index

    elif strategy_name == "vanilla":
        import os
        from Core.configs.vdb_config import VDBConfig
        retrieval_method = rag_config.retrieval_method
        
        vdb_cfg: VDBConfig = rag_config.vdb_config
        vdb_store_path = vdb_cfg.vdb_dir_name
        if cfg.save_path not in vdb_store_path:
            vdb_store_path = os.path.join(cfg.save_path, vdb_store_path)
        
        if retrieval_method == "bm25":
            from Core.utils.bm25 import BM25
            bm25_path = os.path.join(vdb_store_path, "bm25_index.pkl")
            bm25 = BM25.load(bm25_path)
            log.info(f"Successfully loaded BM25 index from {bm25_path}")
            dependencies["bm25"] = bm25
        else:
            from Core.configs.embedding_config import EmbeddingConfig
            from Core.provider.vdb import VectorStore
            from Core.provider.embedding import TextEmbeddingProvider

            embed_cfg: EmbeddingConfig = rag_config.vdb_config.embedding_config

            embed_model = TextEmbeddingProvider(
                model_name=embed_cfg.model_name,
                backend=embed_cfg.backend,
                device=embed_cfg.device,
                max_length=embed_cfg.max_length,
                api_base=embed_cfg.api_base,
            )
            
            vdb = VectorStore(
                embedding_model=embed_model,
                db_path=vdb_store_path,
                collection_name=vdb_cfg.collection_name,
            )
            log.info(f"Successfully loaded vector store from {vdb_store_path}")
            dependencies["vector_store"] = vdb

    elif strategy_name == "mmr":
        from Core.configs.embedding_config import EmbeddingConfig
        from Core.provider.vdb import VectorStore

        embed_cfg: EmbeddingConfig = rag_config.vdb_config.embedding_config
        embed_model_type = embed_cfg.type
        if embed_model_type == "text":
            from Core.provider.embedding import TextEmbeddingProvider

            embed_model = TextEmbeddingProvider(
                model_name=embed_cfg.model_name,
                backend=embed_cfg.backend,
                device=embed_cfg.device,
                max_length=embed_cfg.max_length,
                api_base=embed_cfg.api_base,
            )
        elif embed_model_type == "gme":
            from Core.provider.embedding import GmeEmbeddingProvider

            embed_model = GmeEmbeddingProvider(
                model_name=embed_cfg.model_name,
                device=embed_cfg.device,
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {embed_model_type}")

        import os
        from Core.configs.vdb_config import VDBConfig

        vdb_cfg: VDBConfig = rag_config.vdb_config
        vdb_store_path = vdb_cfg.vdb_dir_name
        if cfg.save_path not in vdb_store_path:
            vdb_store_path = os.path.join(cfg.save_path, vdb_store_path)

        vdb = VectorStore(
            embedding_model=embed_model,
            db_path=vdb_store_path,
            collection_name=vdb_cfg.collection_name,
        )
        log.info(f"Successfully loaded vector store from {vdb_store_path}")
        dependencies["vector_store"] = vdb
    else:
        raise ValueError(f"Unknown or unsupported RAG strategy: '{strategy_name}'")

    return dependencies
