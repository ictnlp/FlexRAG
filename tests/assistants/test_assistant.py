from flexrag.assistants import ModularAssistant, ModularAssistantConfig
from flexrag.common.dataclasses import ChatMessages, ChatTurn


class FakeGenerator:
    def chat(self, messages, generation_config=None):
        return [[ChatTurn(role="assistant", content="Mocked assistant response")]]


class TestAssistant:
    query = "Who is Bruce Wayne?"
    # contexts = ["Bruce Wayne is Batman.", "Batman is a superhero."]

    def test_modular_assistant(self):
        assistant = ModularAssistant(
            ModularAssistantConfig(),
            generator=FakeGenerator(),
        )
        response = assistant.answer(
            ChatMessages(history=[ChatTurn(role="user", content=self.query)])
        )
        assert response.response.text_content == "Mocked assistant response"
