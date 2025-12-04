from sympy import N
from Core.Index.Tree import *
from Core.configs.system_config import SystemConfig
from Core.provider.llm import LLM
from Core.Index.Graph import Graph
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.vdb import VectorStore


class GBC:
    """
    A class representing the index combining graph and tree structures.
    This class allows for the creation and management of a tree index, which can be used
    to organize and retrieve information in multimodal applications.
    """

    def __init__(
        self,
        config: SystemConfig,
        graph_index: Optional[Graph] = None,
        TreeIndex: Optional[DocumentTree] = None,
    ):
        """
        Initializes the TreeIndex with an optional index.

        :param index: Optional initial index for the tree.
        """
        self.save_dir = config.save_path
        self.config = config
        self.llm = LLM(config.llm)
        self.TreeIndex: DocumentTree = TreeIndex
        self.GraphIndex: Graph = graph_index

        # load the vdb of entities
        if config.graph.refine_type == "basic":
            self.entity_vdb_path = os.path.join(self.save_dir, "kg_vdb_basic")
        else:
            self.entity_vdb_path = os.path.join(self.save_dir, "kg_vdb")
        
        self.embedder = TextEmbeddingProvider(
            model_name=config.graph.embedding_config.model_name,
            backend=config.graph.embedding_config.backend,
            max_length=config.graph.embedding_config.max_length,
            device=config.graph.embedding_config.device,
            api_base=config.graph.embedding_config.api_base,
        )
        self.entity_vdb: VectorStore = VectorStore(
            db_path=self.entity_vdb_path,
            embedding_model=self.embedder,
            collection_name="kg_collection",
        )
        log.info(f"Entity VDB loaded from {self.entity_vdb_path}")

    def save_gbc_index(self):
        """
        Saves the GBC index to the specified path.

        :param save_path: The path where the index will be saved.
        """
        if self.TreeIndex:
            self.TreeIndex.save_to_file()
        if self.GraphIndex:
            self.GraphIndex.save_graph()

        # vdb is saved automatically when the entity_vdb is created

        log.info(f"GBC index saved")

    def rebuild_vdb(self):
        """
        Rebuilds the vector database for entities using the current graph index.
        """
        if not self.GraphIndex:
            raise ValueError("GraphIndex is not set. Cannot rebuild VDB.")

        self.entity_vdb.reset()

        nodes = self.GraphIndex.get_all_nodes()
        texts = []
        meta_datas = []

        for node in nodes:
            texts.append(node)

            entity = self.GraphIndex.get_entity_by_node_name(node)
            tmp_dict = {
                "entity_name": entity.entity_name,
                "entity_type": entity.entity_type,
                "description": entity.description,
            }
            meta_datas.append(tmp_dict)

        self.entity_vdb.add_texts(texts=texts, metadatas=meta_datas)
        log.info(f"Rebuilt entity VDB with {len(texts)} entries.")

    @classmethod
    def load_gbc_index(cls, config: SystemConfig):
        """
        Loads the GBC index from the specified path.

        :param config: The configuration object containing the save path.
        :return: An instance of GBC with the loaded index.
        """
        tree_index = DocumentTree.load_from_file(
            DocumentTree.get_save_path(config.save_path)
        )
        
        if config.graph.refine_type == "basic":
            variant = "basic"
        else:
            variant = None
        
        graph_index = Graph.load_from_dir(config.save_path, variant=variant)
        GBC = cls(config=config, graph_index=graph_index, TreeIndex=tree_index)
        log.info(f"GBC index loaded from {config.save_path}")
        return GBC
