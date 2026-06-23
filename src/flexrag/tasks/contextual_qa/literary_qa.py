import re

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn, RetrievedContext
from flexrag.datasets.benchmarks import LiteraryQADataset, LiteraryQADatasetConfig
from flexrag.datasets.core import ContextualQASample
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

from ..contextual_qa_base import ContextualQATask, ContextualQATaskConfig
from ..task_base import TASKS

_METRIC_TEMPLATE = """
### Task Description
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

### The instruction to evaluate
You are an expert narrative analyst tasked with evaluating candidate answers to questions about books. You are provided with the summary of the book to refer to as added context.
Here is the summary of {title}:
{summary}

Question:
{question}

### Response to evaluate
{response}

### Reference Answer (Score 5)
{reference_answer}

### Score Rubrics:
[How acceptable is the candidate answer EITHER compared to the reference answer OR validated against the summary?]
Score 1: The candidate answer is completely wrong.
Score 2: The answer does not answer the original question, but there is some information related to the reference answer or summary.
Score 3: The candidate answer is partially correct, but it contains some errors, omits key information or adds major extra information.
Score 4: The candidate answer is correct but it includes minor details that cannot be verified against the reference or the summary.
Score 5: The candidate answer is either exactly identical to one of the reference answers or it is a paraphrase of a reference answer that does not alter its meaning.
""".strip()


class _LiteraryQAMetric:
    """The LLM-as-a-Judge evaluation metric for LiteraryQA Task."""

    template = _METRIC_TEMPLATE

    def __init__(self, generator: GeneratorProtocol):
        self.generator = generator
        self.gen_cfg = GenerationConfig(do_sample=False)
        return

    def __call__(
        self,
        questions: list[str],
        responses: list[str],
        golden_responses: list[list[str]],
        golden_contexts: list[list[RetrievedContext]],
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
    """Configuration for LiteraryQA Task."""


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

    def __init__(
        self,
        config: LiteraryQATaskConfig,
        llm_judger: GeneratorProtocol | None = None,
    ):
        self.llm_judger = llm_judger
        super().__init__(config)
        return

    def load_dataset(self) -> LiteraryQADataset:
        return LiteraryQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "rouge": Rouge(RougeConfig()),
        }
        if self.llm_judger is not None:
            metrics["llm_judger"] = _LiteraryQAMetric(self.llm_judger)
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
