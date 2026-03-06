"""Build knowledge graph from existing tree index."""
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from Core.configs.system_config import load_system_config
from Core.Index.Tree import DocumentTree
from Core.pipelines.kg_builder import build_knowledge_graph
from Core.provider.TokenTracker import TokenTracker

cfg = load_system_config("config/gbc.yaml")
cfg.pdf_path = "/Volumes/ExtMac/Projects/Exorty/BOOKRag/BOOKRAG_VLDB_2026_full.pdf"
cfg.save_path = "/Volumes/ExtMac/Projects/Exorty/BOOKRag/e2e_test_output"

print("=== Config ===")
print(f"LLM:         {cfg.llm.model_name} @ {cfg.llm.api_base}")
print(f"Extractor:   {cfg.graph.extractor_type}")
print(f"Refine type: {cfg.graph.refine_type}")
print(f"Save:        {cfg.save_path}")
print()

# Load existing tree
tree_path = DocumentTree.get_save_path(cfg.save_path)
print(f"Loading tree from: {tree_path}")
tree = DocumentTree.load_from_file(tree_path)
print(f"Tree loaded: {len(tree.nodes)} nodes")
print()

# Init token tracker
token_tracker = TokenTracker.get_instance()
token_tracker.reset()

# Build KG
start = time.time()
graph_index = build_knowledge_graph(tree, cfg)
elapsed = time.time() - start

# Save
graph_index.save_graph()

print()
print(f"=== KG Done in {elapsed:.1f}s ===")
print(f"Total KG nodes: {len(graph_index.get_all_nodes())}")
print(f"Total KG edges: {graph_index.kg.number_of_edges()}")
print(f"Tree-to-KG mappings: {len(graph_index.tree2kg)}")
print(f"Token usage: {token_tracker.stage_history}")

