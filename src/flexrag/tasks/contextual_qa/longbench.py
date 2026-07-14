import re

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.datasets.benchmarks import LongBenchDataset, LongBenchDatasetConfig
from flexrag.datasets.core import ContextualQASample
from flexrag.metrics import (
    F1,
    Evaluator,
    F1Config,
    Rouge,
    RougeConfig,
)
from flexrag.metrics.metrics_base import MetricCallable
from flexrag.models.tokenizer import TokenizerConfig

from ..contextual_qa_base import ContextualQATask, ContextualQATaskConfig
from ..task_base import TASKS


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
        self, additional_metrics: dict[str, MetricCallable] | None = None
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
                all_classes = [item.metadata["all_classes"] for item in self.testset]

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
