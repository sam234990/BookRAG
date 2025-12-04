"""
Official script for evaluating models built for the Qasper dataset. The script
outputs Answer F1 and Evidence F1 reported in the paper.
"""

from collections import Counter
import string
import re
import json

import os
from typing import Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from Core.configs.dataset_config import DatasetConfig
from Eval.utils.extract_answer import AnswerExtractor, load_prompt
from Eval.utils.utils import get_all_cost

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat  # Helper to pass constant arguments to map


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        # The question is unanswerable and the prediction is empty.
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_answers_and_evidence(qa_info: list[dict[Any]], text_evidence_only: bool):
    references = []
    for answer_info in qa_info:
        if answer_info["unanswerable"]:
            references.append(
                {
                    "answer": "Not answerable",
                    "evidence": [],
                    "type": "none",
                    "answer_raw": "Not answerable",
                }
            )
        else:
            if answer_info["extractive_spans"]:
                answer = ", ".join(answer_info["extractive_spans"])
                answer_type = "extractive"
                answer_raw = answer_info["extractive_spans"]
            elif answer_info["free_form_answer"]:
                answer = answer_info["free_form_answer"]
                answer_type = "abstractive"
                answer_raw = answer_info["free_form_answer"]
            elif answer_info["yes_no"]:
                answer = "Yes"
                answer_type = "boolean"
                answer_raw = "Yes"
            elif answer_info["yes_no"] is not None:
                answer = "No"
                answer_type = "boolean"
                answer_raw = "No"

            if text_evidence_only:
                evidence = [
                    text
                    for text in answer_info["evidence"]
                    if "FLOAT SELECTED" not in text
                ]
            else:
                evidence = answer_info["evidence"]
            references.append(
                {
                    "answer": answer,
                    "evidence": evidence,
                    "type": answer_type,
                    "answer_raw": answer_raw,
                }
            )

    return references


def get_accuracy(prediction, ground_truth: list[str]):
    for ground_truth in ground_truth:
        norm_pred = normalize_answer(prediction)
        norm_ans = normalize_answer(ground_truth)
        if norm_ans in norm_pred:
            return 1
    return 0


def eval_single_res(pred, gold_answer: list):
    # return accuracy, f1

    accuracy_score = 0.0
    f1_score = 0.0
    for gold in gold_answer:
        answer_raw = gold.get("answer_raw", "")
        if isinstance(answer_raw, str):
            answer_raw = [answer_raw]
        if isinstance(answer_raw, int):
            answer_raw = [str(answer_raw)]
        acc = get_accuracy(pred, answer_raw)
        accuracy_score = max(accuracy_score, acc)

        f1 = token_f1_score(pred, gold.get("answer", ""))
        f1_score = max(f1_score, f1)

    return accuracy_score, f1_score


def eval_single_file(res_path: str, extractor: AnswerExtractor):
    res_file = os.path.join(res_path, "final_results.json")
    with open(res_file, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    for item in res_data:
        question = item["question"]
        output = item["output"]
        gold_answers = get_answers_and_evidence(item["answer"], text_evidence_only=True)
        correct_answer = str(gold_answers[0].get("answer", ""))
        item['gold_answers'] = gold_answers
        extracted_res, pred_ans, pred_format, llm_score = extractor.extract(
            question, output, correct_answer
        )
        item["extracted_res"] = extracted_res
        item['pred'] = pred_ans
        item["pred_format"] = pred_format
        item["llm_score"] = llm_score
        
        acc, f1 = eval_single_res(pred_ans, gold_answers)
        item["acc"] = acc
        item["f1"] = f1

    # Save results to output_dir
    save_path = os.path.join(res_path, "eval.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(res_data, f, ensure_ascii=False, indent=2)

    return res_data


def eval_qasper(
    data_df: pd.DataFrame, data_cfg: DatasetConfig, method: str, max_workers=4
):
    document_groups = data_df.groupby(["doc_uuid", "doc_path"])

    extractor = AnswerExtractor()
    result = []

    if max_workers > 1:
        # Step 1: Prepare the arguments for all the function calls. This is very fast.
        # We create a list of the 'doc_res_dir' paths that will be processed.
        doc_res_dirs = []
        for (doc_uuid, doc_path), group in document_groups:
            dir_name = f"eval_{data_cfg.dataset_name}_{method}"
            doc_res_dir = os.path.join(data_cfg.working_dir, doc_uuid, dir_name)
            doc_res_dirs.append(doc_res_dir)

        # Step 2: Execute `eval_single_file` in parallel using ThreadPoolExecutor.map
        # .map handles running the function on each item in the `doc_res_dirs` list.
        # `repeat(extractor)` and `repeat(prompt)` pass the same extractor and prompt
        # object to every function call.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # The `map` function returns results in the same order as the input iterable.
            # We wrap the iterator with tqdm for a progress bar.
            results_iterator = executor.map(
                eval_single_file,
                doc_res_dirs,  # The iterable of first arguments
                repeat(extractor),  # The constant second argument
            )

            # Step 3: Combine the results. Because .map preserves order, we can
            # simply loop through and extend our final list.
            for doc_res in tqdm(
                results_iterator, total=len(doc_res_dirs), desc="Processing Documents"
            ):
                result.extend(doc_res)
    else:
        for (doc_uuid, doc_path), group in tqdm(document_groups):
            dir_name = f"eval_{data_cfg.dataset_name}_{method}"
            doc_res_dir = os.path.join(data_cfg.working_dir, doc_uuid, dir_name)

            doc_res = eval_single_file(doc_res_dir, extractor)
            result.extend(doc_res)

    average_acc = np.mean([item["acc"] for item in result])
    average_f1 = np.mean([item["f1"] for item in result])
    average_acc = round(average_acc, 6)
    average_f1 = round(average_f1, 6)
    avg_llm_score = np.mean(
        [item["llm_score"] for item in result if "llm_score" in item]
    )
    avg_llm_score = round(avg_llm_score, 6)
    print("--------------------------------------")
    print(f"total samples: {len(result)}")
    print(f"Avg acc: {average_acc:.6f}")
    print(f"Avg f1: {average_f1:.6f}")
    print(f"Avg llm_score: {avg_llm_score:.6f}")
    score_dict = {
        "Avg acc": average_acc,
        "Avg f1": average_f1,
        "Avg llm_score": avg_llm_score,
        "Total samples": len(result),
    }
    
    # answerable average score
    answerable_acc = []
    answerable_f1 = []
    answerable_llm_score = []
    for item in result:
        gold_answers = item.get('gold_answers', [])
        if gold_answers and gold_answers[0].get("answer", "") != "Not answerable":
            answerable_acc.append(item['acc'])
            answerable_f1.append(item['f1'])
            answerable_llm_score.append(item['llm_score'])
    acc_2 = np.mean(answerable_acc) if len(answerable_acc) > 0 else 0.0
    f1_2 = np.mean(answerable_f1) if len(answerable_f1) > 0 else 0.0
    avg_llm_score_2 = np.mean(answerable_llm_score) if len(answerable_llm_score)>0 else 0.0
    avg_llm_score_2 = round(avg_llm_score_2, 6)
    print("------- Answerable answer result --------")
    print(f"total answerable samples: {len(answerable_acc)}")
    print(f"Avg acc: {acc_2:.6f}")
    print(f"Avg f1: {f1_2:.6f}")
    print(f"Avg llm_score: {avg_llm_score_2:.6f}")
    score_dict["Answerable Avg acc"] = acc_2
    score_dict["Answerable Avg f1"] = f1_2
    score_dict["Answerable Avg llm_score"] = avg_llm_score_2

    cost_dict = get_all_cost(data_df, data_cfg, method)
    for k, v in cost_dict.items():
        if k not in score_dict:
            score_dict[k] = v

    save_dir = os.path.join(data_cfg.working_dir, "0_results")
    os.makedirs(save_dir, exist_ok=True)

    priority_keys = [
        "question",
        "answer",
        "pred",
        "acc",
        "f1",
        "llm_score",
        "extracted_res",
        "output",
    ]

    # 重新排序每个字典，将优先字段放在前面
    sorted_result = []
    for item in result:
        sorted_item = {k: item[k] for k in priority_keys if k in item}
        sorted_item.update({k: v for k, v in item.items() if k not in priority_keys})
        sorted_result.append(sorted_item)

    save_path = os.path.join(
        save_dir, f"final_eval_{data_cfg.dataset_name}_{method}.json"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sorted_result, f, ensure_ascii=False, indent=2)

    score_save_path = os.path.join(
        save_dir, f"final_eval_{data_cfg.dataset_name}_{method}.score.json"
    )
    with open(score_save_path, "w", encoding="utf-8") as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved detailed results to {save_path}")
