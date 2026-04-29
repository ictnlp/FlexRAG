import datetime
import difflib
import re
from collections import Counter
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import configure
from flexrag.datasets.benchmarks import GISADataset, GISADatasetConfig
from flexrag.datasets.core import MappingDataset, QASample
from flexrag.metrics import Evaluator

from ..open_qa_base import OpenQATask, OpenQATaskConfig
from ..task_base import TASKS


class _GISAOfficialMetric:
    """Official metric wrapper for GISA."""

    def _normalize_val(self, val: str | int | float) -> str:
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ["nan", "none", "null"]:
            return ""

        clean_num = val_str.replace(",", "").replace("$", "")
        is_percent = False
        if clean_num.endswith("%"):
            is_percent = True
            clean_num = clean_num[:-1]

        try:
            f_val = float(clean_num)
            if is_percent:
                f_val /= 100.0
            if f_val.is_integer():
                return str(int(f_val))
            return "{:.6f}".format(f_val).rstrip("0").rstrip(".") or "0"
        except ValueError:
            pass

        return val_str.lower().replace(" ", "").replace("*", "").replace("\n", "")

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(col).strip().lower().replace(" ", "") for col in df.columns]
        if hasattr(df, "map"):
            return df.map(self._normalize_val)
        return df.applymap(self._normalize_val)

    def _extract_model_output(self, model_output: str) -> pd.DataFrame | None:
        pattern = r"```(?:tsv)?\s*(.*?)```"
        match = re.search(pattern, model_output, re.DOTALL)
        raw_content = match.group(1) if match else model_output
        raw_content = "\n".join(
            line for line in raw_content.split("\n") if line.strip()
        )
        if not raw_content:
            return None

        try:
            output = pd.read_csv(StringIO(raw_content), sep="\t")
        except Exception:
            return None
        return self._normalize_df(output)

    def _load_ground_truth(self, answer_csv: str, answer_type: str) -> pd.DataFrame:
        header = "infer" if answer_type == "table" else None
        df = pd.read_csv(StringIO(answer_csv), header=header)
        return self._normalize_df(df)

    def _calculate_f1(
        self, tp: int, n_pred: int, n_gt: int
    ) -> tuple[float, float, float]:
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        if precision + recall == 0:
            return precision, recall, 0.0
        return precision, recall, 2 * precision * recall / (precision + recall)

    def _flatten_table(self, df: pd.DataFrame) -> list[tuple[str, str]]:
        items = []
        for col in df.columns:
            for val in df[col]:
                items.append((col, val))
        return items

    def _evaluate_item(
        self, pred_df: pd.DataFrame | None, gt_df: pd.DataFrame
    ) -> dict[str, float]:
        if pred_df is None or pred_df.empty:
            return {"item_em": 0.0}

        pred_item = "".join(pred_df.iloc[0, :].tolist())
        gt_item = "".join(gt_df.iloc[0, :].tolist())
        return {"item_em": float(pred_item == gt_item)}

    def _evaluate_set(
        self, pred_df: pd.DataFrame | None, gt_df: pd.DataFrame
    ) -> dict[str, float]:
        if pred_df is None or pred_df.empty:
            return {"set_precision": 0.0, "set_recall": 0.0, "set_f1": 0.0}

        pred_set = set(pred_df.iloc[:, -1].tolist())
        gt_set = set(gt_df.iloc[:, -1].tolist())
        precision, recall, f1 = self._calculate_f1(
            len(pred_set.intersection(gt_set)), len(pred_set), len(gt_set)
        )
        return {"set_precision": precision, "set_recall": recall, "set_f1": f1}

    def _evaluate_list(
        self, pred_df: pd.DataFrame | None, gt_df: pd.DataFrame
    ) -> dict[str, float]:
        if pred_df is None or pred_df.empty:
            return {"list_content_f1": 0.0, "list_order_score": 0.0}

        pred_list = pred_df.iloc[:, -1].tolist()
        gt_list = gt_df.iloc[:, -1].tolist()
        gt_counter = Counter(gt_list)
        pred_counter = Counter(pred_list)
        num_common = sum((gt_counter & pred_counter).values())
        precision = num_common / len(pred_list) if pred_list else 0.0
        recall = num_common / len(gt_list) if gt_list else 0.0
        if precision + recall == 0:
            content_f1 = 0.0
        else:
            content_f1 = 2 * precision * recall / (precision + recall)
        order_score = difflib.SequenceMatcher(None, gt_list, pred_list).ratio()
        return {
            "list_content_f1": round(content_f1, 4),
            "list_order_score": round(order_score, 4),
        }

    def _evaluate_table(
        self, pred_df: pd.DataFrame | None, gt_df: pd.DataFrame
    ) -> dict[str, float]:
        default_res = {
            "table_row_f1": 0.0,
            "table_row_precision": 0.0,
            "table_row_recall": 0.0,
            "table_item_f1": 0.0,
            "table_item_precision": 0.0,
            "table_item_recall": 0.0,
        }
        if pred_df is None or pred_df.empty:
            return default_res.copy()

        common_cols = [col for col in gt_df.columns if col in pred_df.columns]
        if not common_cols:
            row_precision, row_recall, row_f1 = 0.0, 0.0, 0.0
        else:
            pred_rows = set(
                tuple(row)
                for row in pred_df[common_cols].fillna("__NAN__").astype(str).to_numpy()
            )
            gt_rows = set(
                tuple(row)
                for row in gt_df[common_cols].fillna("__NAN__").astype(str).to_numpy()
            )
            row_precision, row_recall, row_f1 = self._calculate_f1(
                len(pred_rows.intersection(gt_rows)), len(pred_rows), len(gt_rows)
            )

        pred_items = Counter(self._flatten_table(pred_df))
        gt_items = Counter(self._flatten_table(gt_df))
        item_precision, item_recall, item_f1 = self._calculate_f1(
            sum((pred_items & gt_items).values()),
            sum(pred_items.values()),
            sum(gt_items.values()),
        )

        return {
            "table_row_f1": row_f1,
            "table_row_precision": row_precision,
            "table_row_recall": row_recall,
            "table_item_f1": item_f1,
            "table_item_precision": item_precision,
            "table_item_recall": item_recall,
        }

    def _evaluate_one(
        self,
        prediction: str,
        answer_csv: str,
        answer_type: str,
    ) -> dict[str, float | str]:
        q_type = answer_type.lower()
        pred_df = self._extract_model_output(prediction)
        gt_df = self._load_ground_truth(answer_csv, q_type)

        if q_type == "item":
            metrics = self._evaluate_item(pred_df, gt_df)
        elif q_type == "set":
            metrics = self._evaluate_set(pred_df, gt_df)
        elif q_type == "list":
            metrics = self._evaluate_list(pred_df, gt_df)
        elif q_type == "table":
            metrics = self._evaluate_table(pred_df, gt_df)
        else:
            metrics = self._evaluate_item(pred_df, gt_df)

        if pred_df is None or len(pred_df.columns) == 0:
            metrics["global_em"] = 0.0
        elif q_type == "set":
            pred_set = set(pred_df.iloc[:, 0].tolist())
            gt_set = set(gt_df.iloc[:, 0].tolist())
            metrics["global_em"] = float(pred_set == gt_set)
        else:
            metrics["global_em"] = float(
                np.array_equal(pred_df.to_numpy(), gt_df.to_numpy())
            )
        metrics["question_type"] = answer_type
        return metrics

    def _gather_results(
        self, score_list: list[dict[str, float | str]]
    ) -> tuple[dict[str, float], dict]:
        if not score_list:
            return {"overall_global_em": 0.0}, {"overall_global_em": 0.0}

        df = pd.DataFrame(score_list)
        overall_em = float(df["global_em"].mean())
        flat_summary = {"overall_global_em": overall_em}
        summary: dict[str, dict | float] = {"overall_global_em": overall_em}

        for answer_type, type_df in df.groupby("question_type", sort=False):
            type_result = {"num_samples": int(len(type_df))}
            numeric_means = type_df.drop(columns=["question_type"]).mean(
                numeric_only=True
            )
            for key, value in numeric_means.round(4).items():
                if pd.isna(value):
                    continue
                type_result[f"overall_{key}"] = float(value)
                flat_summary[f"{answer_type}_overall_{key}"] = float(value)
            summary[str(answer_type)] = type_result

        return flat_summary, summary

    def __call__(
        self,
        responses: list[str],
        golden_responses: list[list[str]],
        metadatas: list[dict],
    ):
        item_scores = []
        for response, golds, metadata in zip(responses, golden_responses, metadatas):
            metrics = self._evaluate_one(
                prediction=response,
                answer_csv=golds[0] if golds else "",
                answer_type=str(metadata.get("answer_type", "item")),
            )
            metrics["id"] = str(metadata.get("id", ""))
            item_scores.append(metrics)

        flat_summary, summary = self._gather_results(item_scores)
        return flat_summary, {"summary": summary, "item_scores": item_scores}


@configure
class GISATaskConfig(OpenQATaskConfig, GISADatasetConfig):
    """Configuration for GISA Task.

    :param current_date: The date injected into the GISA answer-format prompt.
        If not specified, today's local date will be used. Default is None.
    :type current_date: Optional[str]
    """

    current_date: Optional[str] = None


@TASKS("gisa", config_class=GISATaskConfig)
class GISATask(OpenQATask):
    """The GISA Task for general information-seeking assistants."""

    template = """You are a helpful assistant. Given a user's question, your task is to think step by step and output the final answer in the format of TSV.

# Final Answer Format
You must output the final answer within <answer></answer> tags.
Inside these tags, you must strictly follow the TSV (Tab-Separated Values) format enclosed in a code block ```tsv.

Determine the nature of the answer (Item, List, or Table) and format it as follows:

1. If the answer is a Single Item (Fact/Value):
   - Use a single column with the header `Value`.
   - Example:
     ```tsv
     Value
     Kansas City Chiefs
     ```

2. If the answer is a List:
   - Use a single column with the header `Item`.
   - Example:
     ```tsv
     Item
     Apple
     Banana
     Cherry
     ```

3. If the answer is a Table (Structured Data):
   - Use standard TSV with appropriate headers for each column.
   - Example:
     ```tsv
     Name\tRole\tYear
     Alice\tEngineer\t2023
     Bob\tDesigner\t2024
     ```

Critical constraints:
- The content inside the ```tsv code block must be valid TSV.
- Always include a header row.
- Do not add markdown notes or explanations inside the code block. Put any summary text outside the code block but still inside the <answer> tags.

Current date: {current_date}
User Question: {question}
"""

    def load_dataset(self) -> MappingDataset[QASample]:
        return GISADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        return Evaluator({"official_score": _GISAOfficialMetric()})

    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        current_date = self.config.current_date or datetime.date.today().isoformat()
        prompt = self.template.format(
            current_date=current_date,
            question=sample.question,
        )
        return assistant.answer([{"role": "user", "content": prompt}])
