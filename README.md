<h1 align="center">BookRAG</h1>

<p align="center">
  <strong>BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.03413">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A%202512.03413-b31b1b?style=flat-square" alt="Paper">
  </a>
  <a href="./BOOKRAG_VLDB_2026_full.pdf">
    <img src="https://img.shields.io/badge/PDF-Paper%20PDF-7c3aed?style=flat-square" alt="Paper PDF">
  </a>
  <a href="https://github.com/sam234990/BookRAG">
    <img src="https://img.shields.io/badge/Code-BookRAG-111827?style=flat-square" alt="Code">
  </a>
</p>

> **BookRAG** is the official repository for **"BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents"**, accepted by **VLDB 2026**.

BookRAG targets RAG over long, highly structured documents such as books, manuals, handbooks, reports, and technical guidebooks. Instead of flattening documents into isolated text chunks, BookRAG builds a structure-aware **BookIndex** that combines document hierarchies, entity relations, and fine-grained evidence mapping for more effective retrieval and generation.

## News

- **VLDB 2026**: BookRAG has been accepted to VLDB 2026.
- **arXiv**: The preprint is available at [arXiv:2512.03413](https://arxiv.org/abs/2512.03413).
- **Environment**: Installation files are provided through [`requirements.txt`](./requirements.txt) and [`environment.yml`](./environment.yml).

## Why BookRAG?

Complex documents are rarely just bags of chunks. They usually contain explicit logical structures, nested chapters, cross-section references, tables, figures, and entity-level dependencies. Conventional RAG pipelines often lose these signals during parsing and indexing.

BookRAG is designed around three ideas:

- **Hierarchy matters**: document trees provide natural navigation paths from coarse chapters to fine evidence.
- **Relations matter**: entity-level graphs capture cross-section dependencies that pure chunk retrieval can miss.
- **Query strategy matters**: different questions require different retrieval workflows, from local lookup to global reasoning.

<p align="center">
  <img src="./assets/alpha.png" alt="Comparison of existing RAG methods and BookRAG" width="92%">
</p>

<p align="center"><em>Comparison of existing methods and BookRAG for complex document QA.</em></p>

## Method Overview

BookRAG has two stages:

1. **Offline index construction** builds a BookIndex from each document.
2. **Online retrieval and generation** uses the BookIndex to retrieve relevant evidence and answer user questions.

The current implementation supports multiple retrieval configurations and baselines through YAML configuration files under [`config`](./config) and dataset configuration files under [`Scripts/cfg`](./Scripts/cfg).

### BookIndex Construction

BookRAG first constructs a document-native BookIndex by combining a hierarchical document tree, an entity-relation graph, and mappings between entities and document tree nodes.

<p align="center">
  <img src="./assets/index.png" alt="BookIndex construction process" width="96%">
</p>

<p align="center"><em>The BookIndex construction process.</em></p>

### Agent-based Retrieval

Given a user question, BookRAG performs query classification and planning, then dynamically composes retrieval operators over the BookIndex to locate relevant evidence and generate the final answer.

<p align="center">
  <img src="./assets/online.png" alt="Agent-based retrieval workflow in BookRAG" width="96%">
</p>

<p align="center"><em>The general workflow of agent-based retrieval in BookRAG.</em></p>

## Environment

BookRAG requires Python 3.12. The recommended setup is to create a fresh conda environment from [`environment.yml`](./environment.yml):

```bash
conda env create -f environment.yml
conda activate gbc-rag
```

Alternatively, install the Python dependencies directly:

```bash
pip install -r requirements.txt
```

BookRAG uses [MinerU](https://github.com/opendatalab/MinerU) for PDF parsing and document information extraction. The default configuration uses `vlm-sglang-client`, so PDF parsing expects a running MinerU/SGLang service and a valid `mineru.server_url` in the selected YAML config. For graph extraction, the default local NLP model is `en_core_web_sm`, which is included in [`requirements.txt`](./requirements.txt).

To check the installation without running models or parsing documents:

```bash
python main.py --help
```

If MinerU prints a warning that `sglang` is not installed, it can be ignored when using the client backend with an external service. Install the local SGLang runtime separately only if you plan to run MinerU's `sglang-engine` backend in the same environment.

## Quick Start

Before running BookRAG, update the following configuration files:

- **System configuration**: model, retriever, embedding, VLM, and runtime settings, for example [`config/default.yaml`](./config/default.yaml)
- **Dataset configuration**: input path, working directory, dataset split, and metadata settings, for example [`Scripts/cfg/example-m3docVQA.yaml`](./Scripts/cfg/example-m3docVQA.yaml)

### Offline Index Construction

Use the example script to construct the BookIndex:

```bash
bash Scripts/example-index.sh
```

### Online Retrieval

Run BookRAG on a configured dataset:

```bash
bash Scripts/example-rag.sh
```

### Evaluation

We use a strong LLM as an answer extractor for evaluating model responses. Please configure the API file before evaluation, for example [`config/api.txt`](./config/api.txt).

```bash
bash Scripts/example-eval.sh
```

## Supported Datasets

BookRAG is evaluated on widely used complex document QA benchmarks:

- **MMLongBench-Doc**: [MMLONGBENCH-DOC](https://github.com/mayubo2333/MMLongBench-Doc)
- **m3docvqa**: [M3DocRAG](https://github.com/bloomberg/m3docrag)
- **Qasper**: [Qasper](https://huggingface.co/datasets/allenai/qasper)

This repository releases the final filtered QA files used in our experiments. The original documents, raw QA files, and dataset-specific metadata should still be obtained from the official dataset repositories above.

| Dataset | Released QA file | Original source |
| --- | --- | --- |
| MMLongBench-Doc | [`datasets/MMLongBench-Doc.json`](./datasets/MMLongBench-Doc.json) | [MMLONGBENCH-DOC](https://github.com/mayubo2333/MMLongBench-Doc) |
| m3docvqa | [`datasets/m3docvqa.json`](./datasets/m3docvqa.json) | [M3DocRAG](https://github.com/bloomberg/m3docrag) |
| Qasper | [`datasets/Qasper.json`](./datasets/Qasper.json) | [Qasper](https://huggingface.co/datasets/allenai/qasper) |

Each released QA file follows the unified JSON format below:

```json
[
  {
    "question": "THE FIRST QUESTION",
    "answer": "THE ANSWER OF FIRST QUESTION",
    "doc_uuid": "UUID OF THE DOCUMENT PDF",
    "doc_path": "PATH_TO_DIR/DOCUMENT.pdf"
  },
  {
    "question": "THE SECOND QUESTION",
    "answer": "THE ANSWER OF SECOND QUESTION",
    "doc_uuid": "UUID OF THE DOCUMENT PDF",
    "doc_path": "PATH_TO_DIR/DOCUMENT.pdf"
  }
]
```

Example preprocessing notebooks and scripts are available in [`Scripts/preprocess`](./Scripts/preprocess).

## Repository Layout

```text
BookRAG/
+-- Core/                 # indexing, retrieval, generation, providers, and configs
+-- Eval/                 # evaluation scripts and answer extraction utilities
+-- Scripts/              # example scripts, dataset configs, and preprocessing notebooks
+-- assets/               # figures and static assets used in the README
+-- config/               # system and method configuration files
+-- datasets/             # filtered QA files used in the paper experiments
+-- BOOKRAG_VLDB_2026_full.pdf
+-- main.py
`-- README.md
```

## Links

- `arXiv`: [2512.03413](https://arxiv.org/abs/2512.03413)
- `Paper PDF`: [BOOKRAG_VLDB_2026_full.pdf](./BOOKRAG_VLDB_2026_full.pdf)
- `Code`: [sam234990/BookRAG](https://github.com/sam234990/BookRAG)

## License

The root [`LICENSE`](./LICENSE) provides the GNU Affero General Public License
v3.0 text for repository-level license identification. License terms for
individual source files are specified by their SPDX license identifiers; the
root license does not automatically apply to every file or non-software
artifact in this repository.

Independently authored BookRAG source components carrying the corresponding
SPDX identifier, including the Core components and the `main.py` program entry
point, are available under either Apache-2.0 or AGPL-3.0-only. Files derived
from or adapted for MinerU are available under AGPL-3.0-only. A BookRAG
distribution that includes or combines these MinerU-derived components must
comply with AGPL-3.0.

Datasets, evaluation resources, paper PDFs, figures, experimental artifacts,
scripts and configuration files outside the licensed source components, and
third-party materials are not automatically covered by the BookRAG software
licenses. They remain subject to their respective terms and existing upstream
notices unless explicitly stated otherwise.

The BookRAG paper has been accepted for publication in PVLDB Volume 19 and for
presentation at VLDB 2026. The paper is governed separately by the PVLDB
publication terms and the CC BY-NC-ND 4.0 notice stated in the paper; it is not
covered by the BookRAG software licenses.

See [`LICENSE_SCOPE.md`](./LICENSE_SCOPE.md) and
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for details.

## Citation

If you find BookRAG useful, please cite our paper:

```bibtex
@article{wang2025bookrag,
  title   = {BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents},
  author  = {Wang, Shu and Zhou, Yingli and Fang, Yixiang},
  journal = {arXiv preprint arXiv:2512.03413},
  year    = {2025},
  eprint  = {2512.03413},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2512.03413}
}
```
