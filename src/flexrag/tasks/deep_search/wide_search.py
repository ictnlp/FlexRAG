import json
import re
import traceback
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd

from flexrag.assistants import AssistantProtocol, AssistantResult
from flexrag.common import ChatMessages, configure
from flexrag.datasets.benchmarks import WideSearchDataset, WideSearchDatasetConfig
from flexrag.datasets.core import MappingDataset, QASample
from flexrag.metrics import Evaluator
from flexrag.models.generators import GenerationConfig, GeneratorProtocol

from ..open_qa_base import OpenQATask, OpenQATaskConfig
from ..task_base import TASKS

_EN_TEMPLATE = """# Role
You are an expert in online search. Your task is to gather relevant information
using online search tools based on the user's query, and provide an accurate
structured answer according to the search results.

# Final Answer Format
Return the final answer as a Markdown table enclosed in a ```markdown code block.
The table must include a header row and one row per entity/item requested by the
question. Do not put explanations, notes, or citations inside the code block.

User Query:
{question}
"""

_ZH_TEMPLATE = """# 角色设定
你是一位联网信息搜索专家，需要根据用户的问题搜集相关信息，并给出准确的结构化答案。

# 最终答案格式
请将最终答案以 Markdown 表格形式输出，并放在 ```markdown 代码块中。
表格必须包含表头，并为问题要求的每个实体/项目提供一行。不要在代码块内加入解释、备注或引用。

用户问题：
{question}
"""

_PRIMARY_KEY_PREPROCESS_PROMPT = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.


The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""

_EVAL_COLUMN_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
    "idx_xxx": score,
    "idx_yyy": score,
    ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results.
"""


def _norm_column(col: str) -> str:
    return str(col).strip().lower().replace(" ", "")


def _parse_date(content: str):
    try:
        import dateparser

        return dateparser.parse(content, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        try:
            return pd.to_datetime(content).to_pydatetime()
        except Exception:
            return None


class _WideSearchOfficialMetric:
    """Official-style metric wrapper for WideSearch."""

    def __init__(self, generator: GeneratorProtocol):
        self.generator = generator
        self.gen_cfg = GenerationConfig(do_sample=False)
        return

    def _chat(self, prompt: str) -> str:
        messages = ChatMessages.from_list([{"role": "user", "content": prompt}])
        output = self.generator.chat(messages, self.gen_cfg)
        if not output or not output[0]:
            return ""
        return output[0][0].text_content or ""

    def _parse_markdown_json(self, completion: str) -> Optional[dict]:
        matches = re.findall(r"```json\s*(\{.*?\})\s*```", completion, re.DOTALL)
        if not matches:
            return None
        try:
            obj = json.loads(matches[-1])
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _primary_key_preprocess(
        self, response: list[str], reference: list[str]
    ) -> dict[str, str]:
        prompt = _PRIMARY_KEY_PREPROCESS_PROMPT.format(
            response=response, reference=reference
        )
        parsed = self._parse_markdown_json(self._chat(prompt))
        if parsed is None:
            return {}
        return {str(k): str(v) for k, v in parsed.items()}

    def _llm_judge_column(
        self, response: list[str], target: list[str], criterion: str
    ) -> tuple[list[float], list[str]]:
        response_dict = {
            f"idx_{idx}": {"response": resp, "target": tar}
            for idx, (resp, tar) in enumerate(zip(response, target))
        }
        prompt = _EVAL_COLUMN_PROMPT.format(criterion=criterion, response=response_dict)
        completion = self._chat(prompt)
        score_dict = self._parse_markdown_json(completion)
        if score_dict is None:
            return [0.0] * len(response), ["llm judge failed due to parse error"] * len(
                response
            )

        score_list = []
        for idx in range(len(response)):
            raw_score = score_dict.get(f"idx_{idx}", 0)
            score_list.append(1.0 if raw_score == 1 else 0.0)
        return score_list, [completion] * len(response)

    def _extract_dataframe(self, response: str) -> pd.DataFrame | None:
        response_df = None
        markdown_matches = re.findall(r"```markdown(.*?)```", response, re.DOTALL)
        if not markdown_matches:
            pipe_positions = [m.start() for m in re.finditer(r"\|", response)]
            if len(pipe_positions) >= 4:
                first_pipe = pipe_positions[0]
                last_pipe = pipe_positions[-1]
                start = response.rfind("\n", 0, first_pipe)
                start = 0 if start == -1 else start
                end = response.find("\n", last_pipe)
                end = len(response) if end == -1 else end
                table_candidate = response[start:end]
                markdown_matches = re.findall(r"((?:\|.*\n?)+)", table_candidate)

        if markdown_matches:
            markdown_str = markdown_matches[0].strip()
            lines = markdown_str.split("\n")
            lines[0] = lines[0].replace(" ", "").lower()
            lines = [line.strip() for line in lines]
            new_lines = []
            for line in lines:
                if set(line.strip()).issubset(set("|- :")) or "|" not in line:
                    continue
                new_lines.append("|".join([part.strip() for part in line.split("|")]))
            markdown_str = "\n".join(new_lines)
            try:
                response_df = pd.read_csv(StringIO(markdown_str), sep="|")
                response_df = response_df.loc[
                    :, ~response_df.columns.str.startswith("Unnamed")
                ]
            except Exception:
                return None
        return response_df

    def _load_answer_df(self, metadata: dict[str, Any]) -> pd.DataFrame:
        required_columns = [
            _norm_column(col) for col in metadata["evaluation"]["required"]
        ]
        gold_csv_path = metadata.get("gold_csv_path")
        if gold_csv_path is not None and Path(gold_csv_path).exists():
            answer_df = pd.read_csv(gold_csv_path)
        else:
            answer_df = pd.DataFrame(metadata["gold_rows"])

        answer_df.columns = [_norm_column(col) for col in answer_df.columns]
        return answer_df[required_columns]

    def _preprocess_call(self, content: str, preprocess_func_name: str) -> str:
        if preprocess_func_name == "extract_number":
            numbers = re.findall(
                r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?",
                str(content).replace(",", ""),
            )
            if len(numbers) == 0:
                return "NULL"
            return numbers[0]
        if preprocess_func_name == "norm_str":
            return str(content).lower().strip().replace(" ", "").replace("*", "")
        if preprocess_func_name == "norm_date":
            normalized_date = _parse_date(str(content))
            if normalized_date is None:
                return str(content)
            return normalized_date.strftime("%Y-%m-%d")
        raise ValueError(f"Unknown preprocess function: {preprocess_func_name}")

    def _metric_call(
        self,
        response: str,
        target: str,
        criterion: Any,
        metric_func_name: str,
    ) -> tuple[float, str]:
        if metric_func_name == "exact_match":
            if response.lower() == target.lower():
                return 1.0, f"exact match, response: {response}, target: {target}"
            return 0.0, f"exact not match, response: {response}, target: {target}"

        if metric_func_name == "url_match":
            url_pattern = re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            )
            response_urls = [
                urlparse(url).netloc for url in url_pattern.findall(response)
            ]
            target_urls = [urlparse(url).netloc for url in url_pattern.findall(target)]
            if set(response_urls) == set(target_urls):
                return 1.0, f"url match, response: {response}, target: {target}"
            return 0.0, f"url not match, response: {response}, target: {target}"

        if metric_func_name == "in_match":
            if response in target:
                return (
                    1.0,
                    f"response in target, response: {response}, target: {target}",
                )
            return (
                0.0,
                f"response not in target, response: {response}, target: {target}",
            )

        if metric_func_name == "number_near":
            return self._number_near(response, target, float(criterion))

        if metric_func_name == "date_near":
            response_date = _parse_date(response)
            target_date = _parse_date(target)
            if response_date is None or target_date is None:
                if response_date is None and target_date is None:
                    return 1.0, f"date near, response: {response}, target: {target}"
                return (
                    0.0,
                    f"date not convertable, response: {response}, target: {target}",
                )
            if abs((response_date - target_date).days) <= 31:
                return (
                    1.0,
                    f"date near, response: {response_date}, target: {target_date}",
                )
            return (
                0.0,
                f"date not near, response: {response_date}, target: {target_date}",
            )

        raise ValueError(f"Unknown metric function: {metric_func_name}")

    def _number_near(
        self, response: str, target: str, criterion: float
    ) -> tuple[float, str]:
        response_num = self._parse_number_for_metric(response)
        target_num = self._parse_number_for_metric(target)
        if response_num is None or target_num is None:
            if response_num is None and target_num is None and response == target:
                return 1.0, f"number equal, response: {response}, target: {target}"
            return (
                0.0,
                f"number not convertable, response: {response_num}, target: {target_num}",
            )
        if abs(response_num - target_num) <= abs(target_num) * criterion:
            return (
                1.0,
                f"number near in range {criterion * 100}%, response: {response_num}, target: {target_num}",
            )
        return 0.0, f"number not near, response: {response_num}, target: {target_num}"

    def _parse_number_for_metric(self, content: str) -> Optional[float]:
        if "%" in content:
            try:
                return float(content.replace("%", "")) / 100.0
            except (ValueError, TypeError):
                return None
        try:
            return float(content)
        except (ValueError, TypeError):
            return None

    def _calc_f1(self, precision: float, recall: float) -> float:
        if precision + recall <= 1e-9:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _evaluate_one(
        self, response: str, metadata: dict[str, Any]
    ) -> dict[str, float | str]:
        instance_id = self._get_instance_id(metadata)
        try:
            evaluation = metadata["evaluation"]
            required_columns = [_norm_column(col) for col in evaluation["required"]]
            unique_columns = [_norm_column(col) for col in evaluation["unique_columns"]]
            eval_pipeline = {
                _norm_column(col): item
                for col, item in evaluation["eval_pipeline"].items()
            }
            answer_df = self._load_answer_df(metadata)
            response_df = self._extract_dataframe(response)
            if response_df is None:
                return self._empty_result(instance_id, "response_df is None")

            answer_df.columns = [_norm_column(col) for col in answer_df.columns]
            response_df.columns = [_norm_column(col) for col in response_df.columns]
            if set(required_columns) != set(response_df.columns):
                column_map = self._primary_key_preprocess(
                    response_df.columns.tolist(), required_columns
                )
                response_df.rename(columns=column_map, inplace=True)

            if set(required_columns) != set(response_df.columns):
                return self._empty_result(
                    instance_id,
                    f"required_columns {required_columns} != response_df {response_df.columns.tolist()}",
                )

            for col in required_columns:
                try:
                    answer_type = answer_df[col].dtype
                    response_type = response_df[col].dtype
                except Exception:
                    answer_type = None
                    response_type = None
                if (response_type == float and answer_type == int) or (
                    response_type == int and answer_type == float
                ):
                    if response_type == int:
                        response_df[col] = response_df[col].astype(float)
                    elif answer_type == int:
                        answer_df[col] = answer_df[col].astype(float)
                answer_df[col] = answer_df[col].astype(str)
                response_df[col] = response_df[col].astype(str)

            response_df.drop_duplicates(subset=unique_columns, inplace=True)
            answer_df.drop_duplicates(subset=unique_columns, inplace=True)

            for col in unique_columns:
                item = eval_pipeline.get(col)
                if item is None:
                    continue
                metric_func_name_list = item.get("metric", [])
                if (
                    "llm_judge" in metric_func_name_list
                    or "exact_match" in metric_func_name_list
                ):
                    primary_key_map = self._primary_key_preprocess(
                        response_df[col].tolist(), answer_df[col].tolist()
                    )
                    response_df[col + "_before_map"] = response_df[col]
                    response_df[col] = response_df[col].apply(
                        lambda x: primary_key_map.get(x, x)
                    )

            for col, item in eval_pipeline.items():
                for preprocess_func_name in item.get("preprocess", []):
                    response_df[col] = response_df[col].apply(
                        lambda x: self._preprocess_call(x, preprocess_func_name)
                    )
                    answer_df[col] = answer_df[col].apply(
                        lambda x: self._preprocess_call(x, preprocess_func_name)
                    )

            score = 0.0
            if answer_df.shape == response_df.shape:
                gt_sorted = answer_df.sort_values(by=required_columns).reset_index(
                    drop=True
                )
                pred_sorted = response_df.sort_values(by=required_columns).reset_index(
                    drop=True
                )
                if gt_sorted.equals(pred_sorted):
                    score = 1.0

            df_inner = pd.merge(
                answer_df,
                response_df,
                on=unique_columns,
                how="inner",
                suffixes=("_query", "_response"),
            )

            df_inner_score = pd.DataFrame(index=df_inner.index)
            df_inner_msg = pd.DataFrame(index=df_inner.index)
            for col in required_columns:
                if col in unique_columns:
                    df_inner_score[f"{col}_exact_match"] = 1.0
                    df_inner_msg[f"{col}_exact_match_eval_msg"] = "key_match"
                    continue

                item = eval_pipeline[col]
                metric_func_name_list = item.get("metric", [])
                criterion = item.get("criterion")
                for metric_func_name in metric_func_name_list:
                    if metric_func_name == "llm_judge":
                        score_list, msg_list = self._llm_judge_column(
                            df_inner[col + "_response"].tolist(),
                            df_inner[col + "_query"].tolist(),
                            criterion,
                        )
                        metric_info_series = pd.Series(
                            zip(score_list, msg_list), index=df_inner.index
                        )
                    else:
                        metric_info_series = df_inner.apply(
                            lambda x: self._metric_call(
                                x[col + "_response"],
                                x[col + "_query"],
                                criterion,
                                metric_func_name,
                            ),
                            axis=1,
                        )
                    df_inner_score[f"{col}_{metric_func_name}"] = (
                        metric_info_series.apply(lambda x: x[0])
                    )
                    df_inner_msg[f"{col}_{metric_func_name}_eval_msg"] = (
                        metric_info_series.apply(lambda x: x[1])
                    )

            row_scores = df_inner_score.min(axis=1)
            tp_by_row = float(row_scores.sum())
            tp_by_item = float(df_inner_score.sum().sum())

            num_pred_rows = len(response_df)
            num_gt_rows = len(answer_df)
            num_pred_items = num_pred_rows * len(required_columns)
            num_gt_items = num_gt_rows * len(required_columns)

            precision_by_row = tp_by_row / num_pred_rows if num_pred_rows > 0 else 0.0
            recall_by_row = tp_by_row / num_gt_rows if num_gt_rows > 0 else 0.0
            precision_by_item = (
                tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
            )
            recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0
            f1_by_row = self._calc_f1(precision_by_row, recall_by_row)
            f1_by_item = self._calc_f1(precision_by_item, recall_by_item)

            msg = df_inner_score.to_string()
            if (
                precision_by_item == recall_by_item == f1_by_item == 1.0
                and precision_by_row == recall_by_row == f1_by_row == 1.0
            ):
                msg += "\nAll items match perfectly."
                score = 1.0

            return {
                "instance_id": instance_id,
                "score": score,
                "precision_by_row": precision_by_row,
                "recall_by_row": recall_by_row,
                "f1_by_row": f1_by_row,
                "precision_by_item": precision_by_item,
                "recall_by_item": recall_by_item,
                "f1_by_item": f1_by_item,
                "msg": msg,
            }
        except Exception:
            return self._empty_result(
                instance_id, f"evaluator error: \n{traceback.format_exc()}"
            )

    def _empty_result(self, instance_id: str, msg: str) -> dict[str, float | str]:
        return {
            "instance_id": instance_id,
            "score": 0.0,
            "precision_by_row": 0.0,
            "recall_by_row": 0.0,
            "f1_by_row": 0.0,
            "precision_by_item": 0.0,
            "recall_by_item": 0.0,
            "f1_by_item": 0.0,
            "msg": msg,
        }

    def _get_instance_id(self, metadata: dict[str, Any]) -> str:
        if "id" in metadata:
            return str(metadata["id"])
        gold_csv_path = metadata.get("gold_csv_path")
        if gold_csv_path is not None:
            return Path(gold_csv_path).stem
        return ""

    def _gather_results(
        self, item_scores: list[dict[str, float | str]]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        metric_names = [
            "score",
            "precision_by_row",
            "recall_by_row",
            "f1_by_row",
            "precision_by_item",
            "recall_by_item",
            "f1_by_item",
        ]
        if not item_scores:
            return {name: 0.0 for name in metric_names}, {"item_scores": []}

        df = pd.DataFrame(item_scores)
        flat_summary = {
            name: float(df[name].mean()) if name in df else 0.0 for name in metric_names
        }
        return flat_summary, {"summary": flat_summary, "item_scores": item_scores}

    def __call__(
        self,
        responses: list[str],
        golden_responses: list[list[str]],
        metadatas: list[dict],
    ):
        item_scores = [
            self._evaluate_one(response=response, metadata=metadata)
            for response, metadata in zip(responses, metadatas)
        ]
        return self._gather_results(item_scores)


@configure
class WideSearchTaskConfig(OpenQATaskConfig, WideSearchDatasetConfig):
    """Configuration for WideSearch Task."""


@TASKS("widesearch", "wide_search", config_class=WideSearchTaskConfig)
class WideSearchTask(OpenQATask):
    """The WideSearch Task for broad information-seeking assistants."""

    def __init__(
        self,
        config: WideSearchTaskConfig,
        llm_judger: GeneratorProtocol | None = None,
        *,
        assistant_factory: Callable[[], AssistantProtocol],
    ) -> None:
        """Initialize the task and its required LLM-based evaluator.

        :param config: WideSearch task configuration.
        :param llm_judger: Generator used by the official judge metric.
        :param assistant_factory: Factory returning a fresh assistant instance.
        """
        self.llm_judger = llm_judger
        super().__init__(config, assistant_factory=assistant_factory)

    def load_dataset(self) -> MappingDataset[QASample]:
        return WideSearchDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        if self.llm_judger is None:
            raise ValueError(
                "WideSearch official evaluation requires `llm_judger` to be "
                "passed to WideSearchTask because the metric uses LLM-based "
                "column/key alignment and field judging."
            )
        return Evaluator({"official_score": _WideSearchOfficialMetric(self.llm_judger)})

    async def evaluate(
        self, assistant: AssistantProtocol, sample: QASample
    ) -> AssistantResult:
        language = (sample.metadata or {}).get("language", "en")
        template = _ZH_TEMPLATE if language == "zh" else _EN_TEMPLATE
        prompt = template.format(question=sample.question)
        return await assistant.answer([{"role": "user", "content": prompt}])
