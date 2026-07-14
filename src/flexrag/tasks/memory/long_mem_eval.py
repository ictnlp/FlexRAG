from collections import defaultdict

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import ChatMessages, RetrievedContext, configure
from flexrag.datasets.benchmarks import LongMemEvalDataset, LongMemEvalDatasetConfig
from flexrag.datasets.core import MultiSessionQASample
from flexrag.metrics import (
    F1,
    Evaluator,
    ExactMatch,
    ExactMatchConfig,
    F1Config,
    Rouge,
    RougeConfig,
)
from flexrag.models.generators import GenerationConfig, GeneratorProtocol

from ..multisession_qa_base import MultiSessionQATask, MultiSessionQATaskConfig
from ..task_base import TASKS

_METRIC_TEMPLATES = {
    "single-session-user": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only.",
    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
}


class _LongMemEvalMetric:
    def __init__(self, generator: GeneratorProtocol):
        self.generator = generator
        self.gen_cfg = GenerationConfig(do_sample=False)
        self.default = False
        return

    def __call__(
        self,
        questions: list[str],
        responses: list[str],
        golden_responses: list[list[str]],
        retrieved_contexts: list[list[RetrievedContext]],
        metadata: list[dict],
    ):
        prompts = []
        for question, response, golden, meta in zip(
            questions, responses, golden_responses, metadata
        ):
            if meta["abstention"]:
                template = _METRIC_TEMPLATES["abstention"]
            elif meta["question_type"] in {
                "single-session-user",
                "single-session-assistant",
                "multi-session",
            }:
                template = _METRIC_TEMPLATES["single-session-user"]
            elif meta["question_type"] == "temporal-reasoning":
                template = _METRIC_TEMPLATES["temporal-reasoning"]
            elif meta["question_type"] == "knowledge-update":
                template = _METRIC_TEMPLATES["knowledge-update"]
            elif meta["question_type"] == "single-session-preference":
                template = _METRIC_TEMPLATES["single-session-preference"]
            else:
                raise ValueError(f"Unsupported question type: {meta['question_type']}")
            prompt = template.format(
                question=question,
                response=response,
                answer=golden[0],
            )
            prompts.append([{"role": "user", "content": prompt}])
        outputs = self.generator.chat(prompts, self.gen_cfg)

        # parse scores from outputs
        labels = defaultdict(list)
        for meta, output in zip(metadata, outputs):
            if output[0].text_content is None:
                label = self.default
            else:
                label = "yes" in output[0].text_content.lower()
            if meta["abstention"]:
                labels["abstention"].append(label)
            else:
                labels[meta["question_type"]].append(label)
        overall_score = sum([sum(v) for v in labels.values()]) / sum(
            [len(v) for v in labels.values()]
        )
        scores = {k: sum(v) / len(v) if len(v) > 0 else 0.0 for k, v in labels.items()}
        scores["overall"] = overall_score
        return scores, {"details": labels}


@configure
class LongMemEvalTaskConfig(MultiSessionQATaskConfig, LongMemEvalDatasetConfig):
    """Configuration for LongMemEval Task."""


@TASKS("longmemeval", "long_mem_eval", config_class=LongMemEvalTaskConfig)
class LongMemEvalTask(MultiSessionQATask):
    """LongMemEval Task."""

    def __init__(
        self,
        config: LongMemEvalTaskConfig,
        llm_judger: GeneratorProtocol | None = None,
    ):
        self.llm_judger = llm_judger
        super().__init__(config)
        return

    def load_dataset(self) -> LongMemEvalDataset:
        return LongMemEvalDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "rouge": Rouge(RougeConfig()),
        }
        if self.llm_judger is not None:
            self.logger.info("LLM judger is enabled for evaluation.")
            metrics["llm_judger"] = _LongMemEvalMetric(self.llm_judger)
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: MultiSessionQASample
    ) -> AssistantResponse:
        metadata = sample.metadata or {}
        messages = ChatMessages.from_list(
            [{"role": "user", "content": sample.question}],
            metadata={"date": metadata["question_date"]},
        )
        return assistant.answer(messages)
