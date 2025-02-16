from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from flexrag.models import GENERATORS, GenerationConfig, GeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.utils import LOGGER_MANAGER

from .assistant import ASSISTANTS, AssistantBase

logger = LOGGER_MANAGER.get_logger("flexrag.assistant")


@dataclass
class BasicAssistantConfig(GeneratorConfig, GenerationConfig):
    """The configuration for the basic assistant.

    :param prompt_path: The path to the prompt file. Defaults to None.
    :type prompt_path: str, optional
    :param use_history: Whether to save the chat history for multi-turn conversation. Defaults to False.
    :type use_history: bool, optional
    """

    prompt_path: Optional[str] = None
    use_history: bool = False


@ASSISTANTS("basic", config_class=BasicAssistantConfig)
class BasicAssistant(AssistantBase):
    """A basic assistant that generates response without retrieval."""

    def __init__(self, cfg: BasicAssistantConfig):
        # set basic args
        self.gen_cfg = cfg
        if self.gen_cfg.sample_num > 1:
            logger.warning("Sample num > 1 is not supported for Assistant")
            self.gen_cfg.sample_num = 1

        # load generator
        self.generator = GENERATORS.load(cfg)

        # load prompts
        if cfg.prompt_path is not None:
            self.prompt = ChatPrompt.from_json(cfg.prompt_path)
        else:
            self.prompt = ChatPrompt()
        if cfg.use_history:
            self.history_prompt = deepcopy(self.prompt)
        else:
            self.history_prompt = None
        return

    def answer(self, question: str) -> tuple[str, None, dict[str, ChatPrompt]]:
        # prepare system prompt
        if self.history_prompt is not None:
            prompt = deepcopy(self.history_prompt)
        else:
            prompt = deepcopy(self.prompt)

        prompt.update(ChatTurn(role="user", content=question))

        # generate response
        response = self.generator.chat([prompt], generation_config=self.gen_cfg)[0][0]

        # update history prompt
        if self.history_prompt is not None:
            self.history_prompt.update(ChatTurn(role="user", content=question))
            self.history_prompt.update(ChatTurn(role="assistant", content=response))
        return response, None, {"prompt": prompt}

    def clear_history(self) -> None:
        if self.history_prompt is not None:
            self.history_prompt = deepcopy(self.prompt)
        return
