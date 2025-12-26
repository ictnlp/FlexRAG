import pytest

from flexrag.assistants import ModularAssistant, ModularAssistantConfig
from flexrag.models import OpenAIGeneratorConfig


class TestAssistant:
    query = "Who is Bruce Wayne?"
    # contexts = ["Bruce Wayne is Batman.", "Batman is a superhero."]

    @pytest.mark.asyncio
    async def test_modular_assistant(self, mock_openai_client):
        assistant = ModularAssistant(
            ModularAssistantConfig(
                generator_type="openai",
                openai_config=OpenAIGeneratorConfig(
                    model_name="gpt-4",
                ),
            )
        )
        r1, _, _ = assistant.answer(self.query)
        return
