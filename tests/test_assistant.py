import pytest

from flexrag.assistant import BasicAssistant, BasicAssistantConfig
from flexrag.models import OpenAIGeneratorConfig


class TestAssistant:
    query = "Who is Bruce Wayne?"
    # contexts = ["Bruce Wayne is Batman.", "Batman is a superhero."]

    @pytest.mark.asyncio
    async def test_basic_assistant(self, mock_openai_client):
        assistant = BasicAssistant(
            BasicAssistantConfig(
                generator_type="openai",
                openai_config=OpenAIGeneratorConfig(
                    model_name="gpt-4",
                ),
            )
        )
        r1, _, _ = assistant.answer(self.query)
        return
