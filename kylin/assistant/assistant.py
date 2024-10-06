import logging
import os
from copy import deepcopy
from dataclasses import dataclass

from kylin.prompt import ChatTurn, ChatPrompt
from kylin.models import (
    GenerationConfig,
    GeneratorConfig,
    load_generator,
)
from kylin.retriever import RetrievedContext
from kylin.utils import Choices

logger = logging.getLogger("Assistant")


@dataclass
class AssistantConfig(GeneratorConfig, GenerationConfig):
    adaptive_retrieval: bool = False
    response_type: Choices(["short", "long", "original"]) = "short"  # type: ignore


class Assistant:
    def __init__(self, cfg: AssistantConfig):
        # set basic args
        self.adaptive_retrieval = cfg.adaptive_retrieval
        self.gen_cfg = cfg
        if self.gen_cfg.sample_num > 1:
            logger.warning("Sample num > 1 is not supported for Assistant")
            self.gen_cfg.sample_num = 1

        # load generator
        self.generator = load_generator(cfg)

        # load prompts
        match cfg.response_type:
            case "short":
                self.prompt_with_ctx = ChatPrompt.from_json(
                    os.path.join(
                        os.path.dirname(__file__),
                        "assistant_prompts",
                        "shortform_generate_prompt_with_context.json",
                    )
                )
                self.prompt_wo_ctx = ChatPrompt.from_json(
                    os.path.join(
                        os.path.dirname(__file__),
                        "assistant_prompts",
                        "shortform_generate_prompt_without_context.json",
                    )
                )
            case "long":
                self.prompt_with_ctx = ChatPrompt.from_json(
                    os.path.join(
                        os.path.dirname(__file__),
                        "assistant_prompts",
                        "longform_generate_prompt_with_context.json",
                    )
                )
                self.prompt_wo_ctx = ChatPrompt.from_json(
                    os.path.join(
                        os.path.dirname(__file__),
                        "assistant_prompts",
                        "longform_generate_prompt_without_context.json",
                    )
                )
            case "original":
                self.prompt_with_ctx = ChatPrompt()
                self.prompt_wo_ctx = ChatPrompt()
        return

    def answer(
        self, question: str, contexts: list[RetrievedContext]
    ) -> tuple[str, ChatPrompt]:
        """Answer question with given contexts

        Args:
            question (str): The question to answer.
            contexts (list): The contexts searched by the searcher.

        Returns:
            response (str): response to the question
            prompt (ChatPrompt): prompt used.
        """
        if self.adaptive_retrieval:
            if not self.decide_search(question):
                contexts = []

        # prepare system prompt
        if len(contexts) > 0:
            prompt = deepcopy(self.prompt_with_ctx)
        else:
            prompt = deepcopy(self.prompt_wo_ctx)

        # prepare user prompt
        usr_prompt = ""
        for n, context in enumerate(contexts):
            if context.text:
                ctx = context.text
            else:
                ctx = context.full_text
            usr_prompt += f"Context {n + 1}: {ctx}\n\n"
        usr_prompt += f"Question: {question}"

        # generate response
        prompt.update(ChatTurn(role="user", content=usr_prompt))
        response = self.generator.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return response, prompt

    def decide_search(self, question: str) -> bool:
        # TODO implement this
        raise NotImplementedError
