from flexrag.models import GenerationConfig


class TestGenerationConfig:
    def test_generation_config_accepts_tools_and_reasoning_effort(self):
        cfg = GenerationConfig(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            reasoning_effort="balanced-plus",
        )

        assert cfg.tools[0]["function"]["name"] == "get_weather"
        assert cfg.reasoning_effort == "balanced-plus"
