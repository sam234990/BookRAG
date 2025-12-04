from typing import Optional
from Core.Index.Tree import TreeNode, NodeType
import logging

log = logging.getLogger(__name__)


def create_node_by_type(pdf_content: Optional[str], isTitle: bool) -> TreeNode:
    content_type = pdf_content.get("type", "unknown")
    if content_type == "text":
        node_meta = {
            "content": pdf_content.get("text", ""),
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
        }
        if isTitle:
            level = pdf_content.get("text_level", -1)
            if isinstance(level, str):
                try:
                    level = int(level)
                    node_meta["title_level"] = level
                except ValueError:
                    level = -1
                    isTitle = False
            else:
                node_meta["title_level"] = level

        node = TreeNode(node_meta)
        node.type = NodeType.TITLE if isTitle else NodeType.TEXT
        node.outline_node = isTitle
    elif content_type == "image":
        caption = pdf_content.get("image_caption", [])
        caption_str = " ".join(caption) if isinstance(caption, list) else ""
        footnote = pdf_content.get("image_footnote", [])
        footnote_str = " ".join(footnote) if isinstance(footnote, list) else ""
        node_meta = {
            "img_path": pdf_content.get("img_path", ""),
            "caption": caption_str,
            "footnote": footnote_str,
            "content": caption_str + footnote_str,
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
        }
        node = TreeNode(node_meta)
        node.type = NodeType.IMAGE
    elif content_type == "table":
        caption = pdf_content.get("table_caption", [])
        caption_str = " ".join(caption) if isinstance(caption, list) else ""
        footnote = pdf_content.get("table_footnote", [])
        footnote_str = " ".join(footnote) if isinstance(footnote, list) else ""

        node_meta = {
            "img_path": pdf_content.get("img_path", ""),
            "caption": caption_str,
            "footnote": footnote_str,
            "content": caption_str + footnote_str,
            "table_body": pdf_content.get("table_body", ""),
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
        }
        node = TreeNode(node_meta)
        node.type = NodeType.TABLE
    elif content_type == "equation":
        node_meta = {
            "content": pdf_content.get("text", ""),
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
            "text_format": pdf_content.get("text_format", ""),
        }
        node = TreeNode(node_meta)
        node.type = NodeType.EQUATION
    else:
        log.warning(f"Unknown content type: {content_type}. Defaulting to text.")
        node_meta = {
            "content": pdf_content.get("text", ""),
            "pdf_id": pdf_content.get("pdf_id", -1),
            "page_idx": pdf_content.get("page_idx", -1),
            "pdf_para_block": pdf_content.get("middle_json", {}),
        }
        node = TreeNode(node_meta)
        node.type = NodeType.TEXT

    return node
