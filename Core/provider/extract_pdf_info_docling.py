"""Docling-based document parser adapter for BookRAG.

This module provides :func:`parse_doc_with_docling`, which converts a PDF (or
any Docling-supported format) into the canonical ``pdf_list`` format consumed
by the rest of the BookRAG pipeline.

``pdf_list`` schema (one dict per content block):
    - ``type``          : ``"text"`` | ``"image"`` | ``"table"`` | ``"equation"``
    - ``text``          : str — text content (text / equation nodes)
    - ``text_level``    : int — ``-1`` = body text; ``0`` = chapter; ``1`` = section; …
    - ``page_idx``      : int — 0-indexed page number
    - ``pdf_id``        : int — sequential 0-based index matching position in list
    - ``img_path``      : str — absolute path to saved PNG (image / table nodes)
    - ``image_caption`` : list[str]
    - ``image_footnote``: list[str]
    - ``table_caption`` : list[str]
    - ``table_footnote``: list[str]
    - ``table_body``    : str — markdown table string
    - ``middle_json``   : dict — raw Docling item metadata (non-critical)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Core.configs.docling_config import DoclingConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_doc_with_docling(
    pdf_path: str,
    output_dir: str,
    cfg: "DoclingConfig",
) -> list[dict]:
    """Parse *pdf_path* with Docling and return a BookRAG-compatible ``pdf_list``.

    Args:
        pdf_path:   Path to the PDF (or other supported format) to parse.
        output_dir: Root save directory; images are written to
                    ``<output_dir>/docling/images/``.
        cfg:        :class:`~Core.configs.docling_config.DoclingConfig` instance
                    that controls OCR engine, resolution, etc.

    Returns:
        A flat list of dicts matching the ``pdf_list`` schema.  The item at
        position *i* always satisfies ``item["pdf_id"] == i``.
    """
    # ------------------------------------------------------------------ setup
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem

    img_dir = os.path.join(output_dir, "docling", "images")
    os.makedirs(img_dir, exist_ok=True)

    pipeline_options = _build_pipeline_options(cfg)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    log.info(f"[Docling] Converting '{pdf_path}' …")
    conv_res = converter.convert(pdf_path)
    doc = conv_res.document
    doc_stem = Path(pdf_path).stem

    # ---------------------------------------------------------------- iterate
    pdf_list: list[dict] = []
    pdf_id = 0          # 0-based; MUST equal position in list
    pic_counter = 0
    tbl_counter = 0

    for element, _level in doc.iterate_items():
        page_idx = _get_page_idx(element)

        if isinstance(element, SectionHeaderItem):
            # SectionHeaderItem.level is 1-indexed (1 = chapter, 2 = section …)
            heading_level = getattr(element, "level", 1)
            pdf_list.append({
                "type": "text",
                "text": element.text or "",
                "text_level": max(0, heading_level - 1),  # convert to 0-indexed
                "page_idx": page_idx,
                "pdf_id": pdf_id,
                "middle_json": {"docling_label": str(element.label)},
            })

        elif isinstance(element, TableItem):
            tbl_counter += 1
            img_path = _save_element_image(
                element, doc, img_dir, f"{doc_stem}-table-{tbl_counter}.png"
            )
            captions, footnotes = _extract_captions_footnotes(element)
            pdf_list.append({
                "type": "table",
                "text": "",
                "text_level": -1,
                "page_idx": page_idx,
                "pdf_id": pdf_id,
                "img_path": img_path,
                "table_caption": captions,
                "table_footnote": footnotes,
                "table_body": element.export_to_markdown(),
                "middle_json": {"docling_label": "table"},
            })

        elif isinstance(element, PictureItem):
            pic_counter += 1
            img_path = _save_element_image(
                element, doc, img_dir, f"{doc_stem}-picture-{pic_counter}.png"
            )
            captions, footnotes = _extract_captions_footnotes(element)
            pdf_list.append({
                "type": "image",
                "text": "",
                "text_level": -1,
                "page_idx": page_idx,
                "pdf_id": pdf_id,
                "img_path": img_path,
                "image_caption": captions,
                "image_footnote": footnotes,
                "middle_json": {"docling_label": "picture"},
            })

        elif isinstance(element, TextItem):
            # Covers TextItem, ListItem, CodeItem, FormulaItem, FootnoteItem, etc.
            label_str = str(getattr(element, "label", "text")).lower()
            is_formula = "formula" in label_str or "equation" in label_str
            text = getattr(element, "text", "") or ""
            if not text.strip():
                continue  # skip empty elements — don't increment pdf_id
            pdf_list.append({
                "type": "equation" if is_formula else "text",
                "text": text,
                "text_level": -1,
                "page_idx": page_idx,
                "pdf_id": pdf_id,
                "middle_json": {"docling_label": label_str},
            })

        else:
            # Any remaining element types (e.g. page headers, key-value pairs)
            text = getattr(element, "text", "") or ""
            if not text.strip():
                continue
            pdf_list.append({
                "type": "text",
                "text": text,
                "text_level": -1,
                "page_idx": page_idx,
                "pdf_id": pdf_id,
                "middle_json": {"docling_label": str(getattr(element, "label", "unknown"))},
            })

        pdf_id += 1  # advance only when an item was appended

    log.info(f"[Docling] Extracted {len(pdf_list)} content blocks from '{doc_stem}'.")
    return pdf_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pipeline_options(cfg: "DoclingConfig"):
    """Construct :class:`PdfPipelineOptions` from *cfg*."""
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.generate_picture_images = True
    opts.images_scale = cfg.images_scale

    ocr_opts = _build_ocr_options(cfg)
    if ocr_opts is not None:
        opts.ocr_options = ocr_opts

    return opts


def _build_ocr_options(cfg: "DoclingConfig"):
    """Return an OCR-options object matching *cfg.ocr_engine*, or *None*."""
    engine = cfg.ocr_engine.lower()
    force = cfg.force_full_page_ocr
    lang = cfg.lang

    try:
        if engine == "easyocr":
            from docling.datamodel.pipeline_options import EasyOcrOptions
            return EasyOcrOptions(force_full_page_ocr=force, lang=[lang])
        elif engine == "tesseract":
            from docling.datamodel.pipeline_options import TesseractCliOcrOptions
            return TesseractCliOcrOptions(force_full_page_ocr=force, lang=lang)
        elif engine == "rapidocr":
            from docling.datamodel.pipeline_options import RapidOcrOptions
            return RapidOcrOptions(force_full_page_ocr=force)
    except ImportError as exc:
        log.warning(
            f"[Docling] Could not import OCR options for engine '{engine}': {exc}. "
            "Falling back to Docling's default OCR settings."
        )
    return None


def _get_page_idx(element) -> int:
    """Extract 0-indexed page number from a Docling element's provenance."""
    try:
        if element.prov:
            return max(0, element.prov[0].page_no - 1)  # Docling page_no is 1-indexed
    except (AttributeError, IndexError):
        pass
    return 0


def _save_element_image(element, doc, img_dir: str, filename: str) -> str:
    """Save element image as PNG and return its absolute path (or '' on failure)."""
    img_path = os.path.join(img_dir, filename)
    try:
        pil_image = element.get_image(doc)
        if pil_image is not None:
            pil_image.save(img_path, "PNG")
            return img_path
    except Exception as exc:
        log.warning(f"[Docling] Could not save image '{filename}': {exc}")
    return ""


def _extract_captions_footnotes(element) -> tuple[list[str], list[str]]:
    """Pull caption and footnote text lists from a Docling element."""
    captions: list[str] = []
    footnotes: list[str] = []
    for ref in getattr(element, "captions", []):
        text = getattr(ref, "text", None) or getattr(ref, "__str__", lambda: "")()
        if text and text.strip():
            captions.append(text.strip())
    for ref in getattr(element, "footnotes", []):
        text = getattr(ref, "text", None) or getattr(ref, "__str__", lambda: "")()
        if text and text.strip():
            footnotes.append(text.strip())
    return captions, footnotes

