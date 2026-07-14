import pytest

from flexrag.common.dataclasses import ChatMessages, ChatTurn


class TestChatTurn:
    def test_file_content_roundtrip(self):
        turn = ChatTurn(
            role="user",
            content=[
                {"type": "text", "text": "Check this file."},
                {
                    "type": "file",
                    "file_path": "/tmp/sample.html",
                    "mime_type": "text/html",
                    "file_name": "sample.html",
                },
            ],
        )

        restored = ChatTurn.from_dict(turn.to_dict())
        assert restored.content == turn.content

    def test_tool_result_roundtrip(self):
        turn = ChatTurn(
            role="tool",
            tool_call_id="call_1",
            name="add",
            content="98305.0",
        )

        restored = ChatTurn.from_dict(turn.to_dict())
        assert restored.role == "tool"
        assert restored.tool_call_id == "call_1"
        assert restored.name == "add"
        assert restored.content == "98305.0"

    def test_tool_calls_reasoning_and_metadata_roundtrip(self):
        tool_call = {
            "type": "tool_call",
            "id": "call_1",
            "name": "get_weather",
            "arguments": {"city": "Beijing"},
        }
        turn = ChatTurn(
            role="assistant",
            content=[
                {"type": "text", "text": "Checking the weather."},
                tool_call,
            ],
            reasoning_content="Need to call the weather tool first.",
            thinking_blocks=[
                {"type": "thinking", "thinking": "Need to call the weather tool first."}
            ],
            metadata={"finish_reason": "tool_calls", "usage": {"total_tokens": 18}},
        )

        assert turn.text_content == "Checking the weather."
        assert bool(turn.tool_calls)
        assert turn.tool_calls == [tool_call]
        assert turn.reasoning_content == "Need to call the weather tool first."
        assert turn.thinking_blocks == [
            {"type": "thinking", "thinking": "Need to call the weather tool first."}
        ]

        restored = ChatTurn.from_dict(turn.to_dict())
        assert restored.content == turn.content
        assert restored.reasoning_content == turn.reasoning_content
        assert restored.thinking_blocks == turn.thinking_blocks
        assert restored.metadata == turn.metadata
        assert restored.tool_calls == [tool_call]

    def test_text_content_ignores_tool_calls(self):
        turn = ChatTurn(
            role="assistant",
            content=[
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": {"city": "Beijing"},
                }
            ],
        )
        assert turn.text_content is None
        assert bool(turn.tool_calls)

    def test_pretty_print_is_content_first(self, capsys):
        turn = ChatTurn(
            role="assistant",
            content="The weather in Beijing is sunny.",
            reasoning_content="Need to check the weather tool output first.",
            metadata={"finish_reason": "stop", "usage": {"total_tokens": 18}},
        )

        turn.pretty_print()
        captured = capsys.readouterr().out

        assert "The weather in Beijing is sunny." in captured
        assert "reasoning=present" in captured
        assert "Need to check the weather tool output first." not in captured
        assert "legacy_reasoning=present" not in captured


class TestChatMessages:
    def test_tool_aware_sequence(self, tmp_path):
        messages = ChatMessages(
            metadata={"session_id": "weather-session", "date": "2026-07-14"},
            history=[
                ChatTurn(role="user", content="Check the weather."),
                ChatTurn(
                    role="assistant",
                    content=[
                        {
                            "type": "tool_call",
                            "id": "call_1",
                            "name": "get_weather",
                            "arguments": {"city": "Beijing"},
                        }
                    ],
                ),
                ChatTurn(
                    role="tool",
                    tool_call_id="call_1",
                    name="get_weather",
                    content='{"temperature": 24}',
                ),
                ChatTurn(role="assistant", content="It is 24 degrees in Beijing."),
            ],
        )
        assert len(messages) == 4
        assert messages.metadata["session_id"] == "weather-session"
        assert all(turn["metadata"] == {} for turn in messages.to_list(pure_text=True))

        copied = messages.copy()
        copied.metadata["session_id"] = "copied-session"
        assert messages.metadata["session_id"] == "weather-session"

        path = tmp_path / "messages.json"
        messages.to_json(path)
        loaded = ChatMessages.from_json(path)
        assert loaded == messages
        assert loaded.metadata == messages.metadata

    def test_invalid_tool_sequence_without_tool_call(self):
        with pytest.raises(ValueError):
            ChatMessages(
                history=[
                    ChatTurn(role="user", content="Check the weather."),
                    ChatTurn(role="assistant", content="I will help."),
                    ChatTurn(
                        role="tool",
                        tool_call_id="call_1",
                        name="get_weather",
                        content='{"temperature": 24}',
                    ),
                ]
            )

    def test_invalid_first_role(self):
        with pytest.raises(ValueError):
            ChatMessages(
                history=[
                    ChatTurn(
                        role="tool",
                        tool_call_id="call_1",
                        name="get_weather",
                        content='{"temperature": 24}',
                    )
                ]
            )

    def test_tool_turn_requires_tool_call_id(self):
        with pytest.raises(ValueError):
            ChatMessages(
                history=[
                    ChatTurn(role="user", content="Check the weather."),
                    ChatTurn(
                        role="assistant",
                        content=[
                            {
                                "type": "tool_call",
                                "id": "call_1",
                                "name": "get_weather",
                                "arguments": {"city": "Beijing"},
                            }
                        ],
                    ),
                    ChatTurn(
                        role="tool", name="get_weather", content='{"temperature": 24}'
                    ),
                ]
            )
