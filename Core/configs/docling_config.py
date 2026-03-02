from dataclasses import dataclass, field
from typing import List


@dataclass
class DoclingConfig:
    """Configuration for the Docling document parser.

    This config is an alternative to MinerU.  Select it in the top-level YAML
    by setting ``parser: docling``.

    Attributes:
        ocr_engine: OCR back-end to use.  Accepted values:
            ``"easyocr"`` (default), ``"tesseract"``, ``"rapidocr"``.
        force_full_page_ocr: When *True* every page is passed through OCR
            even if selectable text is present.  Strongly recommended for
            scanned documents.
        images_scale: Render scale factor for page images (1.0 ≈ 72 DPI).
            Increase to 2.0 for higher-resolution figure/table crops.
        lang: ISO 639-1 language hint forwarded to the OCR engine.
    """

    ocr_engine: str = "easyocr"
    force_full_page_ocr: bool = False
    images_scale: float = 2.0
    lang: str = "en"

    def __post_init__(self):
        valid_engines = ("easyocr", "tesseract", "rapidocr")
        if self.ocr_engine not in valid_engines:
            raise ValueError(
                f"Unsupported ocr_engine: '{self.ocr_engine}'. "
                f"Choose one of {valid_engines}."
            )
        if self.images_scale <= 0:
            raise ValueError(f"images_scale must be positive, got {self.images_scale}.")

