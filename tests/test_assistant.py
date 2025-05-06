import os
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from flexrag.assistant import BasicAssistant, BasicAssistantConfig
from flexrag.utils import ConfigureBase


@dataclass
class AssistantTestConfig(ConfigureBase):
    assistant_config: BasicAssistantConfig = field(default_factory=BasicAssistantConfig)


class TestAssistant:
    cfg: AssistantTestConfig = OmegaConf.merge(
        OmegaConf.structured(AssistantTestConfig),
        OmegaConf.load(
            os.path.join(os.path.dirname(__file__), "configs", "assistant.yaml")
        ),
    )
    query = "Who is Bruce Wayne?"
    # contexts = ["Bruce Wayne is Batman.", "Batman is a superhero."]

    @pytest.mark.asyncio
    async def test_answer(self):
        assistant = BasicAssistant(self.cfg.assistant_config)
        r1, _, _ = assistant.answer(self.query)
        return
