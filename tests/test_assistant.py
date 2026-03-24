import pytest

from flexrag.assistants import ModularAssistant, ModularAssistantConfig
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.models import LiteLLMGeneratorConfig


class TestAssistant:
    query = "Who is Bruce Wayne?"
    # contexts = ["Bruce Wayne is Batman.", "Batman is a superhero."]

    @pytest.mark.asyncio
    async def test_modular_assistant(self, mock_litellm_client):
        assistant = ModularAssistant(
            ModularAssistantConfig(
                generator_type="litellm",
                litellm_config=LiteLLMGeneratorConfig(
                    provider="openai",
                    model_name="gpt-4o-mini",
                ),
            )
        )
        response = assistant.answer(
            ChatMessages(history=[ChatTurn(role="user", content=self.query)])
        )
        assert response.response.text_content == "Mocked LiteLLM chat response"
