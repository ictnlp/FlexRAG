import re

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import ChatMessages, configure
from flexrag.datasets.benchmarks import BrowseCompDataset, BrowseCompDatasetConfig
from flexrag.datasets.core import MappingDataset, QASample
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

from ..open_qa_base import OpenQATask, OpenQATaskConfig
from ..task_base import TASKS

_METRIC_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()


class _BrowseCompMetric:
    """The evaluation metric for BrowseComp Task."""

    def __init__(self, generator: GeneratorProtocol):
        self.generator = generator
        self.gen_cfg = GenerationConfig(do_sample=False)
        return

    def __call__(
        self,
        questions: list[str],
        responses: list[str],
        golden_responses: list[list[str]],
    ):
        prompts = []
        for question, response, golden_response in zip(
            questions, responses, golden_responses
        ):
            prompt = _METRIC_TEMPLATE.format(
                question=question,
                response=response,
                correct_answer=golden_response[0],
            )
            prompts.append(
                ChatMessages.from_list([{"role": "user", "content": prompt}])
            )
        outputs = self.generator.chat(prompts, self.gen_cfg)
        details = []
        for output in outputs:
            text = output[0].text_content or ""
            match = re.search(r"correct: (yes|no)", text)
            # Default to "no" if no match
            details.append(match.group(0) if match else "no")
        correct_count = sum(1 for detail in details if detail == "correct: yes")
        accuracy = correct_count / len(questions)
        return {"accuracy": accuracy}, {"details": details}


@configure
class BrowseCompTaskConfig(OpenQATaskConfig, BrowseCompDatasetConfig):
    """Configuration for BrowseComp Task."""


@TASKS("browsecomp", config_class=BrowseCompTaskConfig)
class BrowseCompTask(OpenQATask):
    """The BrowseComp Task for open domain question answering."""

    template = (
        "{Question}\n\nYour response should be in the following format:\nExplanation:"
        " {{your explanation for your final answer}}\nExact Answer: {{your succinct,"
        " final answer}}\nConfidence: {{your confidence score between 0% and 100% for"
        " your answer}}"
    )

    def __init__(
        self,
        config: BrowseCompTaskConfig,
        llm_judger: GeneratorProtocol | None = None,
    ):
        self.llm_judger = llm_judger
        super().__init__(config)
        return

    def load_dataset(self) -> MappingDataset[QASample]:
        return BrowseCompDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "rouge": Rouge(RougeConfig()),
        }
        if self.llm_judger is not None:
            metrics["llm_judger"] = _BrowseCompMetric(self.llm_judger)
            self.logger.info("LLM judger is enabled for evaluation.")
        return Evaluator(metrics)

    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        prompt = self.template.format(Question=sample.question)
        response = assistant.answer([{"role": "user", "content": prompt}])
        return response
