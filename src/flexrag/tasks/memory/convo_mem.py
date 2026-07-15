from flexrag.assistants import AssistantProtocol, AssistantResult
from flexrag.common import configure
from flexrag.datasets.benchmarks import ConvoMemDataset, ConvoMemDatasetConfig
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
    "factual": "I will provide you with a **Question**, a **Correct Answer**, and a **Model's Response**. Your sole task is to determine if the Model's Response is **sufficiently correct and complete** to answer the Question, when compared against the Correct Answer.\n\n**Crucial Guidelines for your judgment:**\n\n1.  **Core Information is Key**: The Model's Response must contain all the **essential factual information** that directly answers the Question.\n2.  **Equivalence Counts**: Phrasing doesn't need to be identical. If the Model's Response conveys the exact same core meaning and details as the Correct Answer, even if paraphrased or structured differently, consider it correct.\n3.  **Superfluous (but Accurate) Information**: If the Model's Response includes additional details that were *not explicitly asked for* by the Question, but these details are **accurate and do not contradict** the Correct Answer, you should **still count it as correct** if the core question is fully answered.\n4.  **Partial Answers are Incorrect**: If the Model's Response is missing any essential information directly requested by the Question (even if the Correct Answer provides more detail), it is incorrect.\n5.  **Focus on the Question**: Your primary focus should be whether the Model's Response adequately addresses the Question using information that aligns with the Correct Answer. Do not penalize the model for not reiterating every single word or incidental detail from the Correct Answer if it wasn't requested.\n\n**Answer only \"RIGHT\" or \"WRONG\". Do not provide any additional text, explanations, or reasoning.**\n\nQuestion: {question}\nCorrect Answer: {golden_answer}\nModel Response: {response}\n\nAnswer (RIGHT/WRONG):",
    "rubric": "You are a judge evaluating if a model's response meets the criteria in a rubric.\n\nQuestion asked: {question}\n\nEvaluation rubric: {golden_answer}\n\nModel's response: {response}\n\nGuidelines:\n- The model's response must fully satisfy ALL criteria in the rubric\n- Partial fulfillment of criteria should result in INCORRECT\n- The response must demonstrate clear understanding of the context\n- Missing any important aspect mentioned in the rubric means INCORRECT\n\nEvaluate whether the model's response satisfies the rubric criteria.\n\nIf the response satisfies the criteria: RIGHT\nIf the response does not satisfy the criteria: WRONG\n\nRespond with only RIGHT or WRONG.",
    "temporal": "You are a judge determining if a model's temporal answer is correct.\n\nQuestion: {question}\n\nCorrect answer: {golden_answer}\n\nModel's answer: {response}\n\nGuidelines:\n- Time calculations must be accurate within reasonable bounds\n- Accept different but equivalent time expressions (3 months = 90 days = 12 weeks)\n- The model must show correct temporal reasoning\n- Off-by-one errors in day counts are NOT acceptable\n\nTemporal answers often have legitimate variations. Consider if the model's answer represents the same time period or demonstrates correct temporal reasoning.\n\nIf the temporal answer is correct: RIGHT\nIf the temporal answer is incorrect: WRONG\n\nRespond with only RIGHT or WRONG.",
    "abstention": 'You are evaluating an ABSTENTION test where the model should recognize it lacks sufficient information to answer the question.\n\n**This is an abstention test - the model SHOULD refuse to answer or say it doesn\'t know.**\n\n**Critical Guidelines for Abstention Evaluation:**\n\n1. **Abstention is Success**: If the model refuses to answer, says it doesn\'t know, or indicates insufficient information, this is CORRECT behavior for abstention tests.\n\n2. **Various Forms of Abstention are Acceptable**:\n   - "I don\'t know" / "I\'m not sure" / "I cannot determine"\n   - "There is no information about..." / "This wasn\'t mentioned..."\n   - "I don\'t have enough information to answer..."\n   - "Based on the conversation, I cannot tell..."\n   - "This information wasn\'t provided..."\n   - Asking for clarification or more information\n\n3. **Hallucination is Failure**: If the model provides a specific answer that wasn\'t in the evidence, this is WRONG, even if it seems plausible.\n\n4. **Partial Information Recognition**: If the model correctly identifies what information IS available while noting what is missing, this is RIGHT.\n\n5. **Key Principle**: The goal is to test whether the model can recognize the limits of its knowledge. Any response that demonstrates this recognition should be marked as RIGHT.\n\n**Answer only "RIGHT" or "WRONG". Do not provide any additional text, explanations, or reasoning.**\n\nQuestion: {question}\nExpected behavior: The model should indicate it cannot answer due to insufficient information\nModel Response: {response}\n\nDoes the model appropriately abstain from answering or acknowledge insufficient information?\n\nAnswer (RIGHT/WRONG):',
    "user_facts": "You are evaluating whether a model's response correctly answers a question using information from an evidence message.\n\n**Your Task:**\nDetermine if the Model's Response is RIGHT or WRONG based on the following criteria.\n\n**Question Asked:** \n{question}\n\n**Evidence Message Available to the Model:**\n{evidence}\n\n**Correct Answer (what the response should convey):**\n{golden_answer}\n\n**Model's Response to Evaluate:**\n{response}\n\n**Evaluation Criteria:**\n\nWhen the response is RIGHT:\n- The core answer matches the Correct Answer (exact phrasing not required)\n- The response demonstrates understanding of the information from the evidence message\n- The response correctly addresses the question asked\n- If the model includes accurate additional details not in the Correct Answer, that's still RIGHT\n\nWhen the response is WRONG:\n- The core answer contradicts or significantly differs from the Correct Answer\n- Critical information from the evidence message is missing or misunderstood\n- The response shows confusion about the facts\n- The response fails to answer the question asked\n\n**Important Notes:**\n- The model doesn't need to quote the evidence message verbatim\n- Paraphrasing or summarization is expected and acceptable\n- Focus on whether the essential information has been correctly conveyed\n\n**Answer only \"RIGHT\" or \"WRONG\". Do not provide any additional text, explanations, or reasoning.**\n\nAnswer (RIGHT/WRONG):",
}


class _ConvoMemMetric:
    def __init__(self, generator: GeneratorProtocol, subset: str):
        self.generator = generator
        self.gen_cfg = GenerationConfig(do_sample=False)
        self.subset = subset
        self.default = 0
        return

    def __call__(
        self,
        questions: list[str],
        responses: list[str],
        golden_responses: list[list[str]],
        metadata: list[dict],
    ):
        match self.subset:
            case "abstention_evidence":
                template = _METRIC_TEMPLATES["abstention"]
                params = {
                    "question": questions,
                    "response": responses,
                }
            case s if s in {
                "assistant_facts_evidence",
                "changing_evidence",
                "implicit_connection_evidence",
            }:
                template = _METRIC_TEMPLATES["factual"]
                params = {
                    "question": questions,
                    "response": responses,
                    "golden_answer": [g[0] for g in golden_responses],
                }
            case "preference_evidence":
                template = _METRIC_TEMPLATES["rubric"]
                params = {
                    "question": questions,
                    "response": responses,
                    "golden_answer": [g[0] for g in golden_responses],
                }
            case "user_evidence":
                template = _METRIC_TEMPLATES["user_facts"]
                evidences = []
                for meta in metadata:
                    evidence = ""
                    for n, conv in enumerate(meta["message_evidences"]):
                        evidence += f"Message Evidence {n}: {conv['text']}\n"
                    evidences.append(evidence.rstrip())
                params = {
                    "question": questions,
                    "response": responses,
                    "golden_answer": [g[0] for g in golden_responses],
                    "evidence": evidences,
                }
            case _:
                raise ValueError(f"Unsupported subset: {self.subset}")
        prompts = []
        for row in zip(*params.values()):
            keys = {k: v for k, v in zip(params.keys(), row)}
            prompt = template.format(**keys)
            prompts.append([{"role": "user", "content": prompt}])
        outputs = self.generator.chat(prompts, self.gen_cfg)

        # compute scores
        scores = []
        for output in outputs:
            text = output[0].text_content.strip().lower() or ""
            if "right" in text:
                scores.append(1)
            elif "wrong" in text:
                scores.append(0)
            else:
                scores.append(self.default)
        accuracy = sum(scores) / len(scores) if len(scores) > 0 else 0.0
        return {"accuracy": accuracy}, {"details": scores}


@configure
class ConvoMemTaskConfig(MultiSessionQATaskConfig, ConvoMemDatasetConfig):
    """Configuration for ConvoMem Task."""


@TASKS("convomem", config_class=ConvoMemTaskConfig)
class ConvoMemTask(MultiSessionQATask):
    """ConvoMem Task."""

    def __init__(
        self,
        config: ConvoMemTaskConfig,
        llm_judger: GeneratorProtocol | None = None,
    ):
        self.llm_judger = llm_judger
        super().__init__(config)
        return

    def load_dataset(self) -> ConvoMemDataset:
        return ConvoMemDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "rouge": Rouge(RougeConfig()),
        }
        if self.llm_judger is not None:
            self.logger.info("LLM judger is enabled for evaluation.")
            metrics["llm_judger"] = _ConvoMemMetric(self.llm_judger, self.config.subset)
        return Evaluator(metrics)

    async def evaluate(
        self, assistant: AssistantProtocol, sample: MultiSessionQASample
    ) -> AssistantResult:
        return await assistant.answer([{"role": "user", "content": sample.question}])
