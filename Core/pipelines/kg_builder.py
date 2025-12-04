from Core.Index.Tree import DocumentTree, NodeType
from Core.Index.Graph import Graph
from Core.pipelines.kg_extractor import KGExtractor
from Core.pipelines.kg_refiner import KGRefiner
from Core.configs.system_config import SystemConfig

from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.TokenTracker import TokenTracker

import logging

log = logging.getLogger(__name__)

# print log for test
from rich.logging import RichHandler

import os
import time

# log_dir = "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/logs"
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# log_file = os.path.join(log_dir, f"kg_builder_{time.strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(
#     level="INFO",
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S",
#     handlers=[
#         RichHandler(rich_tracebacks=True),  # RichHandler 会继续使用自己的漂亮格式
#         logging.FileHandler(
#             log_file, encoding="utf-8"
#         ),  # FileHandler 会使用上面定义的 format
#     ],
# )


def build_knowledge_graph(tree: DocumentTree, cfg: SystemConfig):
    """
    Build a knowledge graph from the given document tree.

    :param tree: DocumentTree object containing the document structure.
    :param graph_config: GraphConfig object containing configuration for the graph.
    :return: A tuple containing the KGExtractor and KGRefiner instances.
    """
    llm = LLM(cfg.llm)
    vlm = VLM(cfg.vlm) if cfg.graph.image_description_force else None

    # try load_the graph if constructed before
    # graph_path = os.path.join(cfg.save_path, Graph._DATA_FILE)
    # if os.path.exists(graph_path):
    #     log.info(f"Loading existing knowledge graph from {graph_path}...")
    #     graph_index = Graph.load_from_dir(cfg.save_path)
    #     return graph_index
    # else:
    #     log.info("No existing knowledge graph found. Creating a new one...")

    if cfg.graph.refine_type == "basic":
        variant = "basic"
    else:
        variant = None

    graph_index = Graph(save_path=cfg.save_path, variant=variant)

    kg_extractor = KGExtractor(
        cfg_graph=cfg.graph, llm=llm, vlm=vlm, save_path=cfg.save_path
    )
    kg_refiner = KGRefiner(
        llm=llm,
        graph_config=cfg.graph,
        graph_index=graph_index,
        save_path=cfg.save_path,
    )

    kg_extract_res = []

    batch_process = True

    if batch_process:
        log.info("Batch processing is enabled for knowledge graph extraction.")
        batch_nodes = []
        batch_title_nodes = []
        batch_title_paths = []
        batch_sibling_nodes = []
        for node in tree.nodes:
            # for node in tree.nodes[:30]:
            if node == tree.root_node:
                # Skip the root node since it doesn't have any other information
                continue
            if node.type == NodeType.TITLE:
                # For title nodes, we collect the path and sibling nodes for batch processing
                title_path = tree.get_path_from_root(node.index_id)
                sibling_nodes = tree.get_sibling_nodes(node.index_id)
                batch_title_nodes.append(node)
                batch_title_paths.append(title_path)
                batch_sibling_nodes.append(sibling_nodes)
            else:
                # For other nodes, we collect them for batch processing
                batch_nodes.append(node)

        # Process title nodes in batches
        if batch_title_nodes:
            log.info("Processing title nodes in batches...")
            res_dict = kg_extractor.batch_extract_titles(
                nodes=batch_title_nodes,
                title_paths=batch_title_paths,
                sibling_nodes_list=batch_sibling_nodes,
            )
            kg_extract_res.extend(res_dict)

        if batch_nodes:
            log.info("Processing non-title nodes in batches...------")
            res_dict = kg_extractor.batch_extract_kg(nodes=batch_nodes)
            kg_extract_res.extend(res_dict)

        # resort the results based on node index
        kg_extract_res.sort(key=lambda x: x.get("node_idx", -1))
    else:
        for node in tree.nodes[:30]:
            # Extract entities and relationships from the node
            if node == tree.root_node:
                # Skip the root node since it doesn't have any other information
                continue
            if node.type == NodeType.TITLE:
                title_path = tree.get_path_from_root(node.index_id)
                sibling_nodes = tree.get_sibling_nodes(node.index_id)
                res_dict = kg_extractor.extract_title(node, title_path, sibling_nodes)
            else:
                res_dict = kg_extractor.extract_kg(node)
            kg_extract_res.append(res_dict)

    log.info("Knowledge graph extraction completed.")
    log.info(f"Extracted {len(kg_extract_res)} nodes from the document tree.")

    token_tracker = TokenTracker.get_instance()
    kg_extraction_cost = token_tracker.record_stage("kg_extraction")
    log.info(f"Knowledge graph extraction cost: {kg_extraction_cost}")

    for res in kg_extract_res:
        if cfg.graph.refine_type == "basic":
            log.info("Using basic KG refinement.")
            kg_refiner.basic_kg_refiner(
                entities=res.get("entities", []),
                relationships=res.get("relations", []),
                source_id=res.get("node_idx", -1),
            )
        elif cfg.graph.refine_type == "advanced":
            kg_refiner.advanced_kg_refiner(
                entities=res.get("entities", []),
                relationships=res.get("relations", []),
                source_id=res.get("node_idx", -1),
            )

    kg_refiner.refine_entities()
    kg_refiner.refine_relation()

    log.info("Knowledge graph refinement completed.")
    kg_refinement_cost = token_tracker.record_stage("kg_refinement")
    log.info(f"Knowledge graph refinement cost: {kg_refinement_cost}")

    kg_refiner.close()

    return graph_index
    # graph_index.save_graph()


if __name__ == "__main__":
    # We test the knowledge graph builder here
    from Core.configs.system_config import load_system_config

    cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")

    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(cfg.save_path))

    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    # Build the knowledge graph
    graph_index = build_knowledge_graph(tree_index, cfg)
    graph_index.save_graph()
    print("Knowledge graph built successfully.")
