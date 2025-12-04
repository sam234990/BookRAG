# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
from pathlib import Path
import logging

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2,
    read_fn,
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.utils.draw_bbox import draw_layout_bbox
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_result_to_middle_json,
)
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make

log = logging.getLogger(__name__)


def prepare_result_dir(output_dir, parse_method):
    """
    Prepare the output environment by creating necessary directories for images and markdown files.
    """
    local_md_dir = str(os.path.join(output_dir, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_name: str,  # Name of the PDF file to be parsed
    pdf_bytes: bytes,  # Bytes of the PDF file to be parsed
    p_lang: str,  # Language for the PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    p_formula_enable=True,  # Enable formula parsing
    p_table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-sglang-client backend
):
    local_image_dir, local_md_dir = prepare_result_dir(output_dir, parse_method)
    new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )
    image_dir = str(os.path.basename(local_image_dir))

    if backend == "pipeline":
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                [new_pdf_bytes],
                [p_lang],
                parse_method=parse_method,
                formula_enable=p_formula_enable,
                table_enable=p_table_enable,
            )
        )
        # for idx, model_list in enumerate(infer_results):
        model_list = copy.deepcopy(infer_results[0])

        images_list = all_image_lists[0]
        pdf_doc = all_pdf_docs[0]
        _lang = lang_list[0]
        _ocr_enable = ocr_enabled_list[0]
        middle_json = pipeline_result_to_middle_json(
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            _lang,
            _ocr_enable,
            p_formula_enable,
        )

        pdf_info = middle_json["pdf_info"]
        md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)

    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]
        middle_json, infer_result = vlm_doc_analyze(
            new_pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            server_url=server_url,
        )
        pdf_info = middle_json["pdf_info"]

        md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)

        model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
        md_writer.write_string(
            f"{pdf_file_name}_model_output.txt",
            model_output,
        )

    md_writer.write_string(
        f"{pdf_file_name}.md",
        md_content_str,
    )

    md_writer.write_string(
        f"{pdf_file_name}_content_list.json",
        json.dumps(content_list, ensure_ascii=False, indent=4),
    )

    md_writer.write_string(
        f"{pdf_file_name}_middle.json",
        json.dumps(middle_json, ensure_ascii=False, indent=4),
    )

    draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    log.info(f"PDF parsing completed. Output saved to {local_md_dir}")
    return middle_json, content_list


def parse_doc(
    pdf_path: Path,
    output_dir,
    lang="en",
    backend="pipeline",
    method="auto",
    server_url=None,
):
    """
    Parameter description:
    pdf_path: Path to the PDF file to be parsed.
    output_dir: Output directory for storing parsing results.
    lang: Language option, default is 'en', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
        Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
        Adapted only for the case where the backend is set to "pipeline"
    backend: the backend for parsing pdf:
        pipeline: More general.
        vlm-transformers: More general.
        vlm-sglang-engine: Faster(engine).
        vlm-sglang-client: Faster(client).
        without method specified, pipeline will be used by default.
    method: the method for parsing pdf:
        auto: Automatically determine the method based on the file type.
        txt: Use text extraction method.
        ocr: Use OCR method for image-based PDFs.
        Without method specified, 'auto' will be used by default.
        Adapted only for the case where the backend is set to "pipeline".
    server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    """
    try:
        file_name = str(Path(pdf_path).stem)
        pdf_bytes = read_fn(pdf_path)
        return do_parse(
            output_dir=output_dir,
            pdf_file_name=file_name,
            pdf_bytes=pdf_bytes,
            p_lang=lang,
            backend=backend,
            parse_method=method,
            server_url=server_url,
        )
    except Exception as e:
        log.error(f"Error parsing {pdf_path}: {e}")
        raise e


def merge_middle_content(
    middle_json, content_list, parse_dir, save_dir=None, file_name=None
):
    """Merge middle JSON content with corresponding content list.

    Args:
        middle_json (dict): The middle JSON object containing PDF information.
        content_list (list): The list of content extracted from the PDF.
        save_dir (str, optional): The directory to save the merged content. Defaults to None.
        file_name (str, optional): The name of the file to save the merged content. Defaults to None.

    Returns:
        list: A list of merged PDF information.
    """
    pdf_info = middle_json["pdf_info"]
    middle_json_para_list = []
    for info in pdf_info:
        para_blocks = info.get("para_blocks", [])
        middle_json_para_list.extend(para_blocks)
    if len(middle_json_para_list) != len(content_list):
        log.error(
            f"Error: The number of items in middle_json ({len(middle_json_para_list)}) does not match the number of content items ({len(content_list)})."
        )
        raise ValueError(
            f"The number of items in middle_json ({len(middle_json_para_list)}) does not match the number of content items ({len(content_list)})."
        )

    res_pdf_info_list = []
    for i in range(len(content_list)):
        res_pdf_info = copy.deepcopy(content_list[i])
        res_pdf_info["middle_json"] = copy.deepcopy(middle_json_para_list[i])
        if "img_path" in res_pdf_info:
            res_pdf_info["img_path"] = os.path.join(parse_dir, res_pdf_info["img_path"])
            if not os.path.exists(res_pdf_info["img_path"]):
                log.error(f"Image path does not exist: {res_pdf_info['img_path']}")
        res_pdf_info_list.append(res_pdf_info)

    log.info(f"Total {len(res_pdf_info_list)} items merged.")

    save_path = (
        os.path.join(save_dir, f"{file_name}_merged_content.json") if save_dir else None
    )
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(res_pdf_info_list, f, ensure_ascii=False, indent=4)
        log.info(f"Merged content saved to {save_path}")
    else:
        log.info("Merged content not saved, no save_dir provided.")

    return res_pdf_info_list


def batch_process_pdfs(
    input_pdf_dir,
    output_dir,
    lang="en",
    backend="pipeline",
    method="auto",
    server_url=None,
):
    """
    Batch process multiple PDF files and parse them.

    Args:
        pdf_path_list (list): List of paths to PDF files.
        output_dir (str): Output directory for storing parsing results.
        lang (str): Language option for OCR, default is 'en'.
        backend (str): Backend for parsing PDF, default is 'pipeline'.
        method (str): Method for parsing PDF, default is 'auto'.
        server_url (str, optional): Server URL for vlm-sglang-client backend.

    Returns:
        list: List of parsed results for each PDF file.
    """
    pdf_path_list = os.listdir(input_pdf_dir)
    pdf_path_list = [
        os.path.join(input_pdf_dir, pdf_path)
        for pdf_path in pdf_path_list
        if pdf_path.endswith(".pdf")
    ]
    print(f"Found {len(pdf_path_list)} PDF files in {input_pdf_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Created output directory: {output_dir}")
    results = []
    for pdf_path in pdf_path_list:
        file_name = str(Path(pdf_path).stem)
        log.info(f"Processing PDF: {file_name}")

        para_output_dir = os.path.join(output_dir, file_name)

        res_path = os.path.join(para_output_dir, f"{file_name}_merged_content.json")
        if os.path.exists(res_path):
            print(f"Skipping {pdf_path}, already processed.")
            continue

        try:
            # Parse the PDF document
            log.info(f"Parsing {pdf_path} with backend {backend} and method {method}")
            if not os.path.exists(para_output_dir):
                os.makedirs(para_output_dir, exist_ok=True)
                log.info(f"Created directory for parsed content: {para_output_dir}")
            middle_json, content_list = parse_doc(
                pdf_path=Path(pdf_path),
                output_dir=para_output_dir,
                lang=lang,
                backend=backend,
                method=method,
                server_url=server_url,
            )

            pdf_list = merge_middle_content(
                middle_json,
                content_list,
                parse_dir=os.path.join(para_output_dir, method),
                save_dir=para_output_dir,
                file_name=file_name,
            )
            results.append(pdf_list)
        except Exception as e:
            log.error(f"Error processing {pdf_path}: {e}")
            continue

    return results


if __name__ == "__main__":
    # test
    # backend = "vlm-sglang-client"  # or "vlm-transformers", "vlm-sglang-engine", "vlm-sglang-client", "pipeline"
    backend = "vlm-sglang-client"
    server_url = "http://127.0.0.1:30000" if backend == "vlm-sglang-client" else None

    method = "auto" if backend == "pipeline" else "vlm"

    # input_pdf_path = "/home/wangshu/multimodal/GBC-RAG/test/double_paper.pdf"
    # output_dir_path = "/home/wangshu/multimodal/GBC-RAG/test/mineru_output"
    input_pdf_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code/mineru/tmp_cost/COSTCO_2021_10K.pdf"
    output_dir_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code/mineru/tmp_cost"

    # Set the environment variable to use models from modelscope if you cannot download the model due to network issues.
    os.environ["MINERU_MODEL_SOURCE"] = "modelscope"

    """To enable VLM mode, change the backend to 'vlm-xxx'"""
    middle_json, content_list = parse_doc(
        input_pdf_path,
        output_dir_path,
        backend=backend,
        method=method,
        server_url=server_url,
    )  # more general.
    # parse_doc(doc_path_list, output_dir, backend="vlm-sglang-client", server_url="http://127.0.0.1:30000"）  # faster(client).

    file_name = str(Path(input_pdf_path).stem)
    save_dir = os.path.join(output_dir_path, method)
    debug = False  # Set to True to load from saved files for debugging.
    if debug:
        tmp_middle_json_path = os.path.join(save_dir, f"{file_name}_middle.json")
        tmp_content_list_path = os.path.join(save_dir, f"{file_name}_content_list.json")
        with open(tmp_middle_json_path, "r", encoding="utf-8") as f:
            middle_json = json.load(f)
        with open(tmp_content_list_path, "r", encoding="utf-8") as f:
            content_list = json.load(f)
        log.info(f"Loaded middle JSON and content list from {output_dir_path}")

    pdf_list = merge_middle_content(
        middle_json,
        content_list,
        parse_dir=os.path.join(output_dir_path, method),
        save_dir=save_dir,
        file_name=file_name,
    )  # merge middle json content with content list.
