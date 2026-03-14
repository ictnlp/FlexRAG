import logging
import os
import re
from abc import abstractmethod
from dataclasses import field
from pathlib import Path
from typing import Optional

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import LOGGER_MANAGER, SimpleProgressLogger, configure
from flexrag.common.database import json_dump
from flexrag.common.dataclasses import ChatMessages, ChatTurn, RetrievedContext
from flexrag.datasets.benchmarks import (
    GutenQADataset,
    GutenQADatasetConfig,
    LiteraryQADataset,
    LiteraryQADatasetConfig,
    LongBenchDataset,
    LongBenchDatasetConfig,
    MultihopRAGDataset,
    MultihopRAGDatasetConfig,
    NarrativeQADataset,
    NarrativeQADatasetConfig,
    SQuADDataset,
    SQuADDatasetConfig,
)
from flexrag.datasets.core import ContextualQASample, MappingDataset
from flexrag.metrics import (
    F1,
    Accuracy,
    AccuracyConfig,
    Evaluator,
    ExactMatch,
    ExactMatchConfig,
    F1Config,
    MetricsBase,
    Rouge,
    RougeConfig,
)
from flexrag.models import GENERATORS, GenerationConfig, GeneratorConfig
from flexrag.models.tokenizer import TokenizerConfig

from .task_base import TASKS, TaskBase


@configure
class ContextualQATaskConfig:
    """Configuration for Contextualized QA Task."""

    log_interval: int = 10
    output_path: Optional[str] = None


class ContextualQATask(TaskBase):
    """Base class for all Contextualized QA Tasks."""

    config: ContextualQATaskConfig

    def setup(self):
        """Setup the Contextualized QA task."""
        # setup logger
        self.logger = LOGGER_MANAGER.get_logger("task.contextualized_qa")
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            log_path = Path(self.config.output_path, "log.txt")
            handler = logging.FileHandler(log_path)
            LOGGER_MANAGER.add_handler(handler)
        self.logger.debug(f"Configs:\n{self.config.dumps()}")

        # setup output paths
        if self.config.output_path is not None:
            self.details_path = Path(self.config.output_path, "details.jsonl")
            self.eval_score_path = Path(self.config.output_path, "eval_score.json")
            self.config_path = Path(self.config.output_path, "config.json")
        else:
            self.details_path = Path(os.devnull)
            self.eval_score_path = Path(os.devnull)
            self.config_path = Path(os.devnull)
        self.config.dump(self.config_path)

        # load dataset
        self.testset = self.load_dataset()

        # load metrics
        self.evaluator = self.load_evaluator()
        return

    def run(self, assistant: AssistantBase):
        """Run the Contextualized QA task."""
        # search and answer questions
        questions: list[str] = []
        golden_answers: list[list[str]] = []
        responses: list[str] = []
        contexts: list[list[RetrievedContext]] = []
        p_logger = SimpleProgressLogger(
            self.logger, interval=self.config.log_interval, total=len(self.testset)
        )
        with open(self.details_path, "w", encoding="utf-8") as f:
            for item in self.testset:
                questions.append(item.question)
                golden_answers.append(item.answers)
                response = self.evaluate(assistant=assistant, sample=item)
                if response.response.text_content is not None:
                    responses.append(response.response.text_content)
                else:
                    responses.append("")
                contexts.append(item.contexts)
                f.write(
                    json_dump(
                        {
                            "question": item.question,
                            "golden": item.answers,
                            "metadata_test": item.meta_data,
                            "response": response,
                        },
                        to_bytes=False,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                p_logger.update(desc="Inferencing")

        # Evaluate the results
        resp_score, resp_score_detail = self.evaluator.evaluate(
            questions=questions,
            responses=responses,
            golden_responses=golden_answers,
            golden_contexts=contexts,
            log=True,
        )

        # Save the evaluation results
        with open(self.eval_score_path, "w", encoding="utf-8") as f:
            f.write(
                json_dump(
                    {
                        "eval_scores": resp_score,
                        "eval_details": resp_score_detail,
                    },
                    to_bytes=False,
                    indent=4,
                    ensure_ascii=False,
                )
            )
        return

    @abstractmethod
    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        """
        Evaluate a single data sample.

        :param assistant: The assistant to evaluate.
        :type assistant: AssistantBase
        :param sample: A single data sample to be evaluated.
        :type sample: ContextualQASample
        :return: The response from the assistant.
        :rtype: AssistantResponse
        """
        return

    @abstractmethod
    def load_dataset(self) -> MappingDataset[ContextualQASample]:
        """Load the dataset for the task.

        :return: The dataset for the task.
        :rtype: MappingDataset[ContextualQASample]
        """
        return

    @abstractmethod
    def load_evaluator(self) -> Evaluator:
        """Load the evaluator for the task.

        :return: The evaluator for the task.
        :rtype: Evaluator
        """
        return


@configure
class LongBenchTaskConfig(ContextualQATaskConfig, LongBenchDatasetConfig):
    """Configuration for LongBench Task."""


@TASKS("longbench", config_class=LongBenchTaskConfig)
class LongBenchTask(ContextualQATask):
    """Contextualized QA Task on LongBench dataset."""

    instructions = {
        "narrative_qa": (
            "You are given a story, which can be either a novel or a movie script, and"
            " a question. Answer the question as concisely as you can, using a single"
            " phrase if possible. Do not provide any explanation.\n\nStory: {context}"
            "\n\nNow, answer the question based on the story as concisely as you can,"
            " using a single phrase if possible. Do not provide any explanation.\n\n"
            "Question: {input}"
        ),
        "qasper": (
            "You are given a scientific article and a question. Answer the question as"
            " concisely as you can, using a single phrase or sentence if possible. If"
            " the question cannot be answered based on the information in the article,"
            ' write "unanswerable". If the question is a yes/no question, answer "yes",'
            ' "no", or "unanswerable". Do not provide any explanation.\n\nArticle:'
            " {context}\n\n Answer the question based on the above article as concisely"
            " as you can, using a single phrase or sentence if possible. If the question"
            " cannot be answered based on the information in the article, write"
            ' "unanswerable". If the question is a yes/no question, answer "yes", "no",'
            ' or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}'
        ),
        "multifield_qa_en": (
            "Read the following text and answer briefly.\n\n{context}\n\nNow, answer"
            " the following question based on the above text, only give me the answer"
            " and do not output any other words.\n\nQuestion: {input}"
        ),
        "multifield_qa_zh": (
            "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，"
            "只告诉我答案，不要输出任何其他字词。\n\n问题：{input}"
        ),
        "hotpot_qa": (
            "Answer the question based on the given passages. Only give me the answer"
            " and do not output any other words.\n\nThe following are given passages."
            "\n{context}\n\nAnswer the question based on the given passages. Only give"
            " me the answer and do not output any other words.\n\nQuestion: {input}"
        ),
        "2wikimultihop_qa": (
            "Answer the question based on the given passages. Only give me the answer"
            " and do not output any other words.\n\nThe following are given passages."
            "\n{context}\n\nAnswer the question based on the given passages. Only give"
            " me the answer and do not output any other words.\n\nQuestion: {input}"
        ),
        "musique": (
            "Answer the question based on the given passages. Only give me the answer"
            " and do not output any other words.\n\nThe following are given passages."
            "\n{context}\n\nAnswer the question based on the given passages. Only give"
            " me the answer and do not output any other words.\n\nQuestion: {input}"
        ),
        "dureader": (
            "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。"
            "\n\n问题：{input}\n回答："
        ),
        "gov_report": (
            "You are given a report by a government agency. Write a one-page summary"
            " of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of"
            " the report."
        ),
        "qm_sum": (
            "You are given a meeting transcript and a query containing a question or"
            " instruction. Answer the query in one or more sentences.\n\nTranscript:"
            "\n{context}\n\nNow, answer the query based on the above meeting transcript"
            " in one or more sentences.\n\nQuery: {input}"
        ),
        "multi_news": (
            "You are given several news passages. Write a one-page summary of all news."
            "\n\nNews:\n{context}\n\nNow, write a one-page summary of all the news."
        ),
        "vc_sum": (
            "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}"
        ),
        "trec": (
            "Please determine the type of the question below. Here are some examples of"
            " questions.\n\n{context}\n{input}"
        ),
        "trivia_qa": (
            "Answer the question based on the given passages. Only give me the answer"
            " and do not output any other words.\n\nThe following are given passages."
            "\n{context}\n\nAnswer the question based on the given passages. Only give"
            " me the answer and do not output any other words.\n\nQuestion: {input}"
        ),
        "sam_sum": (
            "Summarize the dialogue into a few short sentences. The following are some"
            " examples.\n\n{context}\n\n{input}"
        ),
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
        "passage_count": (
            "There are some paragraphs below sourced from Wikipedia. Some of them may"
            " be duplicates. Please carefully read these paragraphs and determine how"
            " many unique paragraphs there are after removing duplicates. In other"
            " words, how many non-repeating paragraphs are there in total?\n\n{context}"
            "\n\nPlease enter the final count of unique paragraphs after removing"
            " duplicates. The output format should only contain the number, such as 1,"
            " 2, 3, and so on."
        ),
        "passage_retrieval_en": (
            "Here are 30 paragraphs from Wikipedia, along with an abstract. Please"
            " determine which paragraph the abstract is from.\n\n{context}\n\nThe"
            " following is an abstract.\n\n{input}\n\nPlease enter the number of the"
            " paragraph that the abstract is from. The answer format must be like "
            '"Paragraph 1", "Paragraph 2", etc.'
        ),
        "passage_retrieval_zh": (
            "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n"
            "{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须"
            '是"段落1"，"段落2"等格式'
        ),
        "lcc": "Please complete the code given below. \n{context}",
        "repobench_p": "Please complete the code given below. \n{context}{input}",
    }

    def load_dataset(self) -> LongBenchDataset:
        return LongBenchDataset(self.config)

    def load_evaluator(
        self, additional_metrics: dict[str, MetricsBase] | None = None
    ) -> Evaluator:
        metrics = {}
        match self.config.subset:
            case subset if subset in {
                "narrative_qa",
                "qasper",
                "multifield_qa_en",
                "hotpot_qa",
                "2wikimultihop_qa",
                "musique",
                "trivia_qa",
            }:
                # English QA subsets are evaluated with F1
                metrics["f1"] = F1(F1Config())
            case subset if subset in {"multifield_qa_zh"}:
                # Chinese QA subsets are evaluated with F1 using jieba tokenizer
                metrics["f1"] = F1(
                    F1Config(tokenizer_config=TokenizerConfig(tokenizer_type="jieba"))
                )
            case subset if subset in {"gov_report", "qm_sum", "multi_news", "sam_sum"}:
                # English summarization subsets are evaluated with ROUGE
                metrics["rouge"] = Rouge(RougeConfig())
            case subset if subset in {"dureader", "vc_sum"}:
                # Chinese summarization subsets are evaluated with ROUGE using jieba tokenizer
                metrics["rouge"] = Rouge(
                    RougeConfig(
                        tokenizer_config=TokenizerConfig(tokenizer_type="jieba")
                    )
                )
            case subset if subset in {"lsht", "trec"}:
                all_classes = [item.meta_data["all_classes"] for item in self.testset]

                # Classification subsets are evaluated with accuracy
                def classification_score(
                    *, responses: list[str], golden_responses: list[list[str]], **kwargs
                ):
                    scores = []
                    for response, golden, classes in zip(
                        responses, golden_responses, all_classes
                    ):
                        em_match_list = []
                        for class_name in classes:
                            if class_name in response:
                                em_match_list.append(class_name)
                        for match_term in em_match_list:
                            if match_term in golden[0] and match_term != golden[0]:
                                em_match_list.remove(match_term)
                        if golden[0] in em_match_list:
                            score = 1.0 / len(em_match_list)
                        else:
                            score = 0.0
                        scores.append(score)
                    final_score = sum(scores) / len(scores) if scores else 0.0
                    return {"classification_score": final_score}, {"details": scores}

                metrics["classification_score"] = classification_score
            case subset if subset in {"passage_retrieval_en"}:
                # Passage retrieval subset is evaluated with retrieval score
                def retrieval_score(
                    responses: list[str], golden_responses: list[list[str]], **kwargs
                ):
                    pattern = r"Paragraph (\d+)"
                    scores = []
                    for response, golden in zip(responses, golden_responses):
                        matches = re.findall(pattern, golden[0])
                        ground_truth_id = matches[0]
                        numbers = re.findall(r"\d+", response)
                        right_num = 0
                        for number in numbers:
                            if str(number) == str(ground_truth_id):
                                right_num += 1
                        score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
                        scores.append(score)
                    final_score = sum(scores) / len(scores) if scores else 0.0
                    return {"retrieval_score": final_score}, {"details": scores}

                metrics["retrieval_score"] = retrieval_score
            case subset if subset in {"passage_retrieval_zh"}:
                # Passage retrieval subset is evaluated with retrieval score
                def retrieval_zh_score(
                    responses: list[str], golden_responses: list[list[str]], **kwargs
                ):
                    pattern = r"段落(\d+)"
                    scores = []
                    for response, golden in zip(responses, golden_responses):
                        matches = re.findall(pattern, golden[0])
                        ground_truth_id = matches[0]
                        numbers = re.findall(r"\d+", response)
                        right_num = 0
                        for number in numbers:
                            if str(number) == str(ground_truth_id):
                                right_num += 1
                        score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
                        scores.append(score)
                    final_score = sum(scores) / len(scores) if scores else 0.0
                    return {"retrieval_score": final_score}, {"details": scores}

                metrics["retrieval_score"] = retrieval_zh_score
            case subset if subset in {"lcc", "repobench_p"}:
                # Code completion subsets are evaluated with code similarity metrics
                try:
                    from rapidfuzz import fuzz
                except ImportError:
                    raise ImportError(
                        "The 'rapidfuzz' library is required for code similarity evaluation. "
                        "Please install it with 'pip install rapidfuzz'."
                    )

                def code_sim_score(
                    responses: list[str], golden_responses: list[list[str]], **kwargs
                ):
                    scores = []
                    for response, golden in zip(responses, golden_responses):
                        all_lines = response.lstrip("\n").split("\n")
                        prediction = ""
                        for line in all_lines:
                            if (
                                ("`" not in line)
                                and ("#" not in line)
                                and ("//" not in line)
                            ):
                                prediction = line
                                break
                        score = fuzz.ratio(prediction, golden[0]) / 100
                        scores.append(score)
                    final_score = sum(scores) / len(scores) if scores else 0.0
                    return {"code_sim_score": final_score}, {"details": scores}

                metrics["code_sim_score"] = code_sim_score
            case subset if subset in {"passage_count"}:
                # Passage counting subset is evaluated with counting score
                def count_score(
                    responses: list[str], golden_responses: list[list[str]], **kwargs
                ):
                    scores = []
                    for response, golden in zip(responses, golden_responses):
                        numbers = re.findall(r"\d+", response)
                        right_num = 0
                        for number in numbers:
                            if str(number) == str(golden[0]):
                                right_num += 1
                        score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
                        scores.append(score)
                    final_score = sum(scores) / len(scores) if scores else 0.0
                    return {"count_score": final_score}, {"details": scores}

                metrics["count_score"] = count_score
            case _:
                raise ValueError(f"Unknown subset: {self.config.subset}")
        evaluator = Evaluator(metrics)
        if additional_metrics is not None:
            evaluator.update(additional_metrics)
        return evaluator

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        # construct the prompt
        subset = self.config.subset
        if subset not in {"gov_report", "multi_news", "vc_sum", "passage_count", "lcc"}:
            prompt = self.instructions[subset].format(
                context=sample.contexts[0].data["text"], input=sample.question
            )
        else:
            prompt = self.instructions[subset].format(
                context=sample.contexts[0].data["text"]
            )
        response = assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response


@configure
class NarrativeQATaskConfig(ContextualQATaskConfig, NarrativeQADatasetConfig):
    """Configuration for NarrativeQA Task."""


@TASKS("narrative_qa", config_class=NarrativeQATaskConfig)
class NarrativeQATask(ContextualQATask):
    """Contextualized QA Task on NarrativeQA dataset."""

    instruction = (
        "You are given a narrative context and a question.\n\nAnswer the question using"
        " only the information provided in the context.\nIf the answer cannot be"
        ' determined from the context, answer "Not answerable".\n\nContext:\n{context}'
        "\n\nQuestion:\n{question}"
    )

    def load_dataset(self) -> NarrativeQADataset:
        return NarrativeQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "rouge": Rouge(RougeConfig()),
            "exact_match": ExactMatch(ExactMatchConfig()),
        }
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        # construct the prompt
        prompt = self.instruction.format(
            context=sample.contexts[0].data["text"], question=sample.question
        )
        response = assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response


@configure
class SQuADTaskConfig(ContextualQATaskConfig, SQuADDatasetConfig):
    """Configuration for SQuAD Task."""


@TASKS("squad", config_class=SQuADTaskConfig)
class SQuADTask(ContextualQATask):
    """Contextualized QA Task on SQuAD dataset."""

    instructions = {
        "v1.1": (
            "Read the following passage and answer the question.\nThe answer must be a"
            " span from the passage.\n\nPassage:\n{context}\n\nQuestion:\n{question}"
        ),
        "v2.0": (
            "Read the following passage and answer the question.\nIf the answer is not"
            ' contained in the passage, output "No Answer".\nOtherwise the answer must'
            " be an exact span from the passage.\n\nPassage:\n{context}\n\nQuestion:"
            "\n{question}"
        ),
    }

    def load_dataset(self) -> SQuADDataset:
        return SQuADDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
        }
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        # construct the prompt
        prompt = self.instructions[self.config.version].format(
            context=sample.contexts[0].data["text"], question=sample.question
        )
        response = assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response


@configure
class GutenQATaskConfig(ContextualQATaskConfig, GutenQADatasetConfig):
    """Configuration for GutenQA Task."""


@TASKS("guten_qa", config_class=GutenQATaskConfig)
class GutenQATask(ContextualQATask):
    """Contextualized QA Task on GutenQA dataset."""

    instructions = {
        "book": (
            "You are given a question and the complete text of a book. Answer the"
            " question based on the information in the book. If the answer cannot be"
            ' determined from the book, output "Insufficient information".\n\n'
            "Question:\n{question}\n\nBook:\n{context}\n\nReturn only the final answer"
            " text, with no extra commentary."
        ),
        "chunk": (
            "You are given a question and several context chunks extracted from a book."
            " Answer the question based on the information in the context chunks. If"
            ' the answer cannot be determined from the contexts, output "Insufficient'
            ' information".\n\nQuestion:\n{question}\n\nContexts:\n{context}\n\nReturn'
            " only the final answer text, with no extra commentary."
        ),
    }

    def load_dataset(self) -> GutenQADataset:
        return GutenQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
        }
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        if self.config.context_mode == "book":
            context_text = sample.contexts[0].data["text"]
            template = self.instructions["book"]
        else:
            context_text = ""
            for context in sample.contexts:
                context_text += context.data["text"] + "\n"
            context_text = context_text.strip()
            template = self.instructions["chunk"]
        # construct the prompt
        prompt = template.format(context=context_text, question=sample.question)
        response = assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response


class _LiteraryQAMetric:
    """The LLM-as-a-Judge evaluation metric for LiteraryQA Task."""

    template = (
        Path(__file__).parent / "task_prompts" / "literaryqa_metric_prompt.txt"
    ).read_text(encoding="utf-8")

    def __init__(self, cfg: GeneratorConfig):
        self.generator = GENERATORS.load(cfg)
        self.gen_cfg = GenerationConfig(do_sample=False)
        assert self.generator is not None, "Generator is not loaded."
        return

    def __call__(
        self,
        questions: list[str],
        responses: list[str],
        golden_responses: list[list[str]],
        golden_contexts: list[list[RetrievedContext]],
        **kwargs,
    ):
        prompts = []
        for question, response, golden_response, ctx in zip(
            questions,
            responses,
            golden_responses,
            golden_contexts,
        ):
            ctx = ctx[0]
            prompt = self.template.format(
                title=ctx.data["title"],
                summary=ctx.data["summary"],
                question=question,
                response=response,
                reference_answer="\n".join(golden_response),
            )
            prompts.append(
                ChatMessages.from_list([{"role": "user", "content": prompt}])
            )
        outputs = self.generator.chat(prompts, self.gen_cfg)

        # compute accuracy
        scores = []
        for output in outputs:
            text = output[0].text_content or ""
            match = re.findall(r"(1|2|3|4|5)", text)
            grade = match[-1] if match else "1"
            scores.append(float(grade))
        average_score = sum(scores) / len(scores) if scores else 0.0
        return {"average_score": average_score}, {"detailed_scores": scores}


@configure
class LiteraryQATaskConfig(ContextualQATaskConfig, LiteraryQADatasetConfig):
    """Configuration for LiteraryQA Task.

    :param llm_judger: The configuration for the LLM judger used in evaluation.
        If not specified, the LLM judger will not be used and the evaluation will only
        include traditional metrics like F1 and Exact Match. Default is None.
    :type llm_judger: GeneratorConfig
    """

    llm_judger: GeneratorConfig = field(default_factory=GeneratorConfig)


@TASKS("literary_qa", config_class=LiteraryQATaskConfig)
class LiteraryQATask(ContextualQATask):
    """Contextualized QA Task on LiteraryQA dataset."""

    instruction = (
        "You are given a question and several context paragraphs extracted from a"
        " literary work. Answer the question based on the information in the context"
        " paragraphs. If the answer cannot be determined from the contexts, output"
        ' "Insufficient information".\n\nQuestion:\n{question}\n\nContexts:\n{context}'
        "\n\nReturn only the final answer text, with no extra commentary."
    )

    def load_dataset(self) -> LiteraryQADataset:
        return LiteraryQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "rouge": Rouge(RougeConfig()),
        }
        if self.config.llm_judger.generator_type is not None:
            metrics["llm_judger"] = _LiteraryQAMetric(self.config.llm_judger)
            self.logger.info("LLM judger is included in the evaluation metrics.")
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        context_text = ""
        for context in sample.contexts:
            context_text += context.data["text"] + "\n"
        context_text = context_text.strip()
        # construct the prompt
        prompt = self.instruction.format(context=context_text, question=sample.question)
        response = assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response


@configure
class MultihopRAGTaskConfig(ContextualQATaskConfig, MultihopRAGDatasetConfig):
    """Configuration for MultihopRAG Task."""


@TASKS("multihop_rag", config_class=MultihopRAGTaskConfig)
class MultihopRAGTask(ContextualQATask):
    """Contextualized QA Task on Multihop RAG dataset."""

    instruction = (
        "Below is a question followed by some context from different sources. Please"
        " answer the question based on the context. The answer to the question is a"
        " word or entity. If the provided information is insufficient to answer the"
        " question, respond 'Insufficient Information'. Answer directly without"
        " explanation.\n\nQuestion:{question}\n\nContext:{context}"
    )

    def load_dataset(self) -> MultihopRAGDataset:
        return MultihopRAGDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "accuracy": Accuracy(AccuracyConfig()),
        }
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        context_text = ""
        for context in sample.contexts:
            context_text += context.data["text"] + "\n"
        context_text = context_text.strip()
        # construct the prompt
        prompt = self.instruction.format(context=context_text, question=sample.question)
        response = assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response
