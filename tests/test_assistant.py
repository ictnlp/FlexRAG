import os
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from librarian.assistant import Assistant, BasicAssistantConfig


@dataclass
class AssistantTestConfig:
    assistant_config: BasicAssistantConfig = field(default_factory=BasicAssistantConfig)


class TestAssistant:
    cfg: AssistantTestConfig = OmegaConf.merge(
        OmegaConf.structured(AssistantTestConfig),
        OmegaConf.load(
            os.path.join(os.path.dirname(__file__), "configs", "assistant.yaml")
        ),
    )
    query = ["Who is Bruce Wayne?"] * 2
    # contexts = ["Bruce Wayne is Batman.", "Batman is a superhero."]

    def valid_result(self, r1, r2):
        pass

    @pytest.mark.asyncio
    async def test_answer(self):
        assistant = Assistant(self.cfg.assistant_config)
        r1, _ = assistant.answer(self.query)
        r2, _ = await assistant.async_answer(self.query)
        self.valid_result(r1, r2)
        return
