# BookRAG

This is the repo for "BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents"

Our framework has based on MinerU and PDF-extract-kit-1.0 to detect PDF processing. For environment setup, please reference to [MinerU](https://github.com/opendatalab/MinerU) for more details if meets some problem related to PDF information extraction.

## Setup Environment

This project using MinerU as PDF parsing method. Please follow the MinerU's [instruction](https://github.com/opendatalab/MinerU) to install the dependency first.

Full environment of BookRAG is coming.

## Run BookRAG

Our BookRAG is two steps: offline Index construction and online query.

Before these two steps, please select and modify the system and dataset config first. For the dataset config, please set the dataset input path and working directory, example file: [dataset_config.yaml](./Scripts/cfg/example-m3docVQA.yaml). For the system config, please set the parameters related to LLM, VLM, and ..., example file: [default.yaml](./config/default.yaml).

### Offline Index

We provide a bash for constructing Book Index, please set the correct config you set before: [index.sh](./Scripts/example-index.sh).

```shell
bash Script/example-index.sh
```

### Online Retrieval

We provide a bash for online retrieval given a specific dataset, please set the correct config you set before: [online.sh](./Scripts/example-rag.sh).

```shell
bash Script/example-rag.sh
```

### Evaluate

We use powerful LLM as answer extractor from the responses of BookRAG or other method. Please set the api file first: [TXT](./Eval/utils/api.txt).

We also provide a bash for evaluate the answer: [eval.sh](./Scripts/example-eval.sh).

```shell
bash Script/example-eval.sh
```

## Dataset format

We use the following datasets:

* MMLongBench-Doc: [MMLONGBENCH-DOC](https://github.com/mayubo2333/MMLongBench-Doc)
* m3docvqa: [M3DocRAG](https://github.com/bloomberg/m3docrag)
* Qasper: [Qasper](https://huggingface.co/datasets/allenai/qasper)

We then transform these dataset into an unified format:

```json
[
    {
        "question":"THE FIRST QUESTION",
        "answer":"THE ANSWER OF FIRST QUESTION",
        "doc_uuid":"UUID OF THE DOCUMENT PDF",
        "doc_path":"PATH TO THE DOCUMENT PDF",
        "xxx":"other attributes"
    },
    {
        "question":"THE SECOND QUESTION",
        "answer":"THE ANSWER OF SECOND QUESTION",
        "doc_uuid":"UUID OF THE DOCUMENT PDF",
        "doc_path":"PATH TO THE DOCUMENT PDF",
        "xxx":"other attributes"
    }
]
```

Please see the example preprocess scripts in [Scripts](./Scripts/preprocess/process_MML.ipynb).
