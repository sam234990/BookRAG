from pydantic import BaseModel, Field
import pickle
import logging
from enum import Enum
from typing import Optional, Dict, Literal, Any, List, Set, Union
import os

log = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Enum for node types in the document tree."""

    ROOT = "root"
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    TITLE = "title"
    UNKNOWN = "unknown"


class MetaInfo(BaseModel):
    # document info
    file_name: str | None = Field(description="the name of the file", default=None)
    file_path: str | None = Field(description="the path of the file", default=None)

    # page info
    page_idx: int | None = Field(description="the page index", default=None)
    page_path: str | None = Field(
        description="the path of the page image", default=None
    )

    # item info from PDF extractor
    pdf_id: int | None = Field(
        description="the unique identifier of the item in the PDF", default=None
    )
    pdf_para_block: dict | None = Field(
        description="the paragraph block information from the PDF extractor",
        default=None,
    )

    # image and table info
    img_path: str | None = Field(
        description="the path of the image or table", default=None
    )
    image_width: int | None = Field(
        description="the width of the image or table", default=0
    )
    image_height: int | None = Field(
        description="the height of the image or table", default=0
    )
    caption: str | None = Field(
        description="the caption of the image or table", default=None
    )
    footnote: str | None = Field(
        description="the footnote of the image or table", default=None
    )

    # table info
    table_body: str | None = Field(
        description="the body content of the table", default=None
    )

    # text info, TreeNodes of Any type have the content
    content: str | None = Field(description="the content of the text", default=None)

    # title info
    title_level: int | None = Field(
        description="the level of the title, 0 is the root", default=-1
    )


class TreeNode:
    def __init__(self, meta_dict: dict = None):
        self.children: List["TreeNode"] = []
        self.parent: "TreeNode" = None
        self.type: NodeType = None
        self.meta_info: MetaInfo = MetaInfo(**meta_dict)
        self.depth = 0
        self.index_id: int = -1  # Unique identifier for the node, should be set later
        self.outline_node: bool = False  # Indicates if the node is a outline node
        self.summary: str = ""  # Summary of the node content

    def add_child(self, child_node: "TreeNode"):
        child_node.parent = self
        child_node.depth = self.depth + 1
        self.children.append(child_node)

    def get_meta_info(self):
        return self.meta_info

    def get_outline_entries(self) -> list:
        """
        returns a list of tuples containing the outline entries.

        Each tuple contains (depth, title, id) for the node.
        """
        entries = []
        if not self.outline_node:
            return entries

        title = getattr(self.meta_info, "content", "Untitled")
        entries.append((self.depth, title, self.index_id))

        for child in self.children:
            entries.extend(child.get_outline_entries())

        return entries


class DocumentTree:
    def __init__(self, meta_dict: dict = None, cfg: Optional[Dict[str, Any]] = None):
        self.nodes: list[TreeNode] = []
        self.meta_info: MetaInfo = MetaInfo(**meta_dict)
        self.root_node: Optional[TreeNode] = None
        self.init_root_node(meta_dict) if meta_dict else None
        self.save_dir = cfg.save_path
        self.pdf_id_to_index_id: Dict[int, int] = {}  # Maps pdf_id to index_id
        self.max_depth = -1

    def init_root_node(self, meta_dict: dict):
        self.root_node = TreeNode(meta_dict)
        self.root_node.index_id = 0
        self.root_node.depth = 0
        self.root_node.type = "root"
        self.root_node.meta_info.pdf_id = 0  # Root node has pdf_id 0
        self.nodes.append(self.root_node)

    def get_nodes(self, hasRoot: bool = False) -> list[TreeNode]:
        if hasRoot:
            return self.nodes
        else:
            return self.nodes[1:] if len(self.nodes) > 1 else []

    def get_outline(self):
        if self.root_node is None:
            return ""

        outline_entries = []
        for child in self.root_node.children:
            outline_entries.extend(child.get_outline_entries())

        lines = [f"{level}\t{title}\t{id_}" for level, title, id_ in outline_entries]
        return "\n".join(lines)

    def add_node(self, node: TreeNode):
        node.index_id = len(self.nodes)
        self.nodes.append(node)
        pdf_id = node.meta_info.pdf_id
        if pdf_id is not None:
            # Map the pdf_id to the index_id of the node
            self.pdf_id_to_index_id[pdf_id] = node.index_id

    def get_node_by_index_id(self, node_id: int) -> TreeNode:
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None

    def get_nodes_by_ids(self, id_list: list[int]) -> list[TreeNode]:
        return [self.nodes[i] for i in id_list if 0 <= i < len(self.nodes)]

    def get_node_by_pdf_id(self, pdf_id: int) -> TreeNode:
        """
        Returns the first node with the given pdf_id.
        If no node is found, returns None.
        """
        node_idx = self.pdf_id_to_index_id.get(pdf_id, None)
        if node_idx is not None:
            # If the pdf_id is mapped to an index_id, return the node directly
            return self.nodes[node_idx]

        # If the pdf_id is not mapped, search through the nodes
        if len(self.nodes) > pdf_id and self.nodes[pdf_id].meta_info.pdf_id == pdf_id:
            # If the pdf_id matches the index_id, return the node directly
            # This is a special case where the pdf_id is used as the index_id
            return self.nodes[pdf_id]

        for node in self.nodes:
            if node.meta_info.pdf_id == pdf_id:
                return node
        return None

    def get_max_depth(self) -> int:
        if self.max_depth != -1:
            return self.max_depth
        if not self.root_node:
            return 0
        self.max_depth = 0
        for node in self.nodes:
            if node.depth > self.max_depth:
                self.max_depth = node.depth
        return self.max_depth

    def get_path_from_root(self, node_id: int) -> List[TreeNode]:
        """
        Returns the path from the root node to the node with the given index_id.
        If the node does not exist, returns an empty list.
        """
        node = self.get_node_by_index_id(node_id)
        if not node:
            return []

        path: List[TreeNode] = []
        visited_ids: Set[int] = set()

        # 循环直到找到根节点或遇到无效节点
        while node:
            # 终止条件1: 通用的根节点判断 (没有父节点)
            # 原始逻辑是不包含根节点，所以我们在添加节点前先判断
            if node.parent is None:
                break

            # 终止条件2: 原始的根节点ID为0的判断
            if node.index_id == 0:
                break

            # 终止条件3: 检测到任何循环 (当前ID已在访问集合中)
            if node.index_id in visited_ids:
                break

            path.append(node)
            visited_ids.add(node.index_id)
            node = node.parent

        return path[::-1]

    def get_sibling_nodes(self, node_id: int) -> List[TreeNode]:
        """
        Returns a list of sibling nodes for the node with the given index_id.
        If the node does not exist or has no siblings, returns an empty list.
        """
        node = self.get_node_by_index_id(node_id)
        if not node or not node.parent:
            return []

        siblings = [
            sibling
            for sibling in node.parent.children
            if sibling.index_id != node.index_id
        ]
        return siblings

    def get_subtree_nodes(self, node_ids: Union[List[int], int]) -> List[TreeNode]:
        """
        Returns a unique list of TreeNode objects that are part of the subtrees 
        rooted at the given node_ids.
        """
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        
        unique_nodes = {}
        visited_ids = set()

        for node_id in node_ids:
            # The helper function will populate the unique_nodes dictionary
            self._get_subtree_recursive(node_id, unique_nodes, visited_ids)
        
        return list(unique_nodes.values())

    def _get_subtree_recursive(self, node_id: int, unique_nodes: Dict[int, TreeNode], visited_ids: set):
        """
        Helper function to recursively traverse the tree and collect unique nodes.
        """
        if node_id in visited_ids:
            return

        node = self.get_node_by_index_id(node_id)
        if not node:
            return

        # Mark as visited and add to our collection
        visited_ids.add(node_id)
        unique_nodes[node_id] = node

        # Recurse for children
        for child in node.children:
            self._get_subtree_recursive(child.index_id, unique_nodes, visited_ids)


    def get_ancestor_at_depth(self, node_id: int, depth: int) -> Optional[TreeNode]:
        """
        Returns the ancestor of the node with the given index_id at the specified depth.
        If the node does not exist or the depth is invalid, returns None.
        """
        node = self.get_node_by_index_id(node_id)
        if not node or depth < 0:
            return None

        while node and node.depth > depth:
            node = node.parent
        return node if node and node.depth == depth else None

    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        """
        Returns a list of nodes that are at the specified depth.
        If no nodes are found at that depth, returns an empty list.
        """
        if depth < 0:
            return []

        nodes_at_depth = [node for node in self.nodes if node.depth == depth]
        return nodes_at_depth

    def get_nodes_data(
        self, node_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries containing the data of the nodes.
        If node_ids is provided, only those nodes will be included.
        """
        if node_ids is None:
            return []

        nodes_data = []
        for node_id in node_ids:
            node = self.get_node_by_index_id(node_id)
            content = node.meta_info.content
            page_idx = node.meta_info.page_idx
            node_data = {
                "index_id": node.index_id,
                "type": node.type,
                "content": content,
                "page": page_idx,
                # "summary": node.summary,
            }
            if node.type == NodeType.IMAGE or node.type == NodeType.TABLE:
                node_data["img_path"] = node.meta_info.img_path
            if node.type is NodeType.TABLE:
                node_data["table_body"] = node.meta_info.table_body
                node_data["caption"] = node.meta_info.caption
                node_data["footnote"] = node.meta_info.footnote

            nodes_data.append(node_data)
        return nodes_data

    def get_filtered_nodes(self, node_type: Union[str, NodeType]) -> List[TreeNode]:
        """
        Input a node type (str or NodeType), str from ["text", "image", "table", "equation", "title", "root"]
        Returns a list of nodes of the specified type.
        """
        if isinstance(node_type, str):
            node_type = NodeType(node_type)
        else:
            node_type = node_type
        sel_nodes = [node for node in self.nodes if node.type == node_type]

        return sel_nodes

    def to_json_summary(self):
        """
        Dump the document tree index to a JSON-like summary format.
        This includes a list of nodes with their index_id, parent_id, type,
        meta_info, and summary.
        This json is used for visualization and debugging purposes.
        """
        node_summaries = []
        for node in self.nodes:
            node_summaries.append(
                {
                    "index_id": node.index_id,
                    "parent_id": node.parent.index_id if node.parent else None,
                    "type": str(node.type) if node.type else None,
                    "meta_info": (
                        node.meta_info.model_dump() if node.meta_info else None
                    ),
                    "summary": node.summary,
                }
            )
        return {"nodes": node_summaries, "meta_info": self.meta_info.model_dump()}

    def get_one_depth_summary(self, node_id: int) -> str:
        """
        Returns a one-depth summary of the node with the given index_id.
        If the node does not exist, returns an empty string.
        """
        node = self.get_node_by_index_id(node_id)
        if not node:
            return ""

        cur_summary = str(node.index_id) + ": " + (node.summary or "")
        if not node.children:
            # If the node has no children, return its own summary
            return cur_summary

        # Collect summaries from all children
        summaries = [cur_summary]
        summaries.append("Children summaries:")
        for child in node.children:
            child_summary = str(child.index_id) + ": " + (child.summary or "")
            if child.summary:
                summaries.append(child_summary)

        # Join summaries with a newline
        return "\n".join(summaries)

    def save_to_file(self):
        save_file_path = DocumentTree.get_save_path(self.save_dir)
        with open(save_file_path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Document tree index saved to {save_file_path}")

        import json

        # save json file for visualization
        json_path = os.path.join(self.save_dir, "tree.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_summary(), f, ensure_ascii=False, indent=2)
        log.info(f"Document tree summary saved to {json_path}")

    @staticmethod
    def get_save_path(input_dir: str) -> str:
        return os.path.join(input_dir, "tree.pkl")

    @staticmethod
    def load_from_file(filepath: str) -> "DocumentTree":
        with open(filepath, "rb") as f:
            return pickle.load(f)
