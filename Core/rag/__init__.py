from typing import Any, Union
from Core.configs.llm_config import LLMConfig
from Core.configs.vlm_config import VLMConfig
from Core.configs.rag_config import RAGConfig
from Core.rag.base_rag import BaseRAG

# Import the specific strategy config classes and the Union type
from Core.configs.rag import ALL_STRATEGY_CONFIGS
from Core.configs.rag.traverse_config import TraverseRAGConfig
from Core.configs.rag.mm_config import MMConfig 
from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.configs.rag.graph_config import GraphRAGConfig
from Core.configs.rag.vanilla_config import VanillaConfig

from Core.rag.traverse_agent import TraverseAgent
from Core.rag.gbc_rag import GBCRAG
from Core.rag.mm_rag import MMRAG
from Core.rag.graph_rag import GraphRAG
from Core.rag.vanilla_rag import VanillaRAG

from Core.provider.llm import LLM
from Core.provider.vlm import VLM

# Define the type for the strategy_config parameter
StrategyConfig = Union[*ALL_STRATEGY_CONFIGS]


def create_rag_agent(
    # The first parameter is now the specific strategy config object
    strategy_config: StrategyConfig,
    llm_config: LLMConfig,
    vlm_config: VLMConfig,
    **dependencies: Any,
) -> BaseRAG:
    """
    Factory function to create a RAG agent based on the provided strategy configuration.
    """
    # Get the strategy name directly from the config object for logging
    strategy_name = strategy_config.strategy
    print(f"INFO: Creating RAG agent with strategy: '{strategy_name}'")

    # 1. Initialize common dependencies
    llm_client = LLM(llm_config)
    vlm_client = VLM(vlm_config)

    # 2. Use isinstance for type-safe dispatching
    if isinstance(strategy_config, TraverseRAGConfig):
        tree_index = dependencies.get("tree_index")
        if not tree_index:
            raise ValueError("TraverseAgent requires a 'tree_index' in dependencies.")

        # 3. Pass the specific config object to the agent's constructor
        return TraverseAgent(
            config=strategy_config,
            llm=llm_client,
            vlm=vlm_client,
            tree_index=tree_index,
        )
    elif isinstance(strategy_config, GBCRAGConfig):
        # 4. For GBCRAG, we assume no additional dependencies are required
        gbc_index = dependencies.get("gbc_index")
        return GBCRAG(
            llm=llm_client,
            vlm=vlm_client,
            config=strategy_config,
            gbc_index=gbc_index,
        )
    elif isinstance(strategy_config, GraphRAGConfig):
        gbc_index = dependencies.get("gbc_index")
        return GraphRAG(
            llm=llm_client,
            vlm=vlm_client,
            config=strategy_config,
            gbc_index=gbc_index,
        )
    elif isinstance(strategy_config, VanillaConfig):
        if strategy_config.retrieval_method == "bm25":
            bm25 = dependencies.get("bm25")
            return VanillaRAG(
                config=strategy_config,
                llm=llm_client,
                bm25=bm25,
                vector_store=None,
            )
        else:
            vector_store = dependencies.get("vector_store")
            return VanillaRAG(
                config=strategy_config,
                llm=llm_client,
                vector_store=vector_store,
                bm25=None,
            )
    elif isinstance(strategy_config, MMConfig):
        vector_store = dependencies.get("vector_store")
        if not vector_store:
            raise ValueError("MMRAG requires a 'vector_store' in dependencies.")

        return MMRAG(
            config=strategy_config,
            llm=llm_client,
            vlm=vlm_client,
            vector_store=vector_store,
            topk=strategy_config.topk if hasattr(strategy_config, 'topk') else 3,
        )

    else:
        raise NotImplementedError(
            f"RAG agent for strategy '{strategy_name}' is not implemented."
        )