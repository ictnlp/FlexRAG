from types import SimpleNamespace

import pytest

import flexrag.models.generators.hf_generator as hf_generator_module
from flexrag.common.dataclasses import ChatTurn
from flexrag.models.generators.hf_generator import (
    HFGenerator,
    HFGeneratorConfig,
    _config_supports_multimodal,
    _content_part_to_hf,
    _turn_to_hf,
)


def test_content_part_to_hf_maps_rich_modalities():
    assert _content_part_to_hf({"type": "text", "text": "hello"}) == {
        "type": "text",
        "text": "hello",
    }
    assert _content_part_to_hf({"type": "image", "image_path": "/tmp/image.png"}) == {
        "type": "image",
        "path": "/tmp/image.png",
    }
    assert _content_part_to_hf({"type": "audio", "file_path": "/tmp/audio.mp3"}) == {
        "type": "audio",
        "path": "/tmp/audio.mp3",
    }
    assert _content_part_to_hf(
        {"type": "video", "url": "https://example.com/video.mp4"}
    ) == {
        "type": "video",
        "url": "https://example.com/video.mp4",
    }


def test_content_part_to_hf_rejects_pdf():
    with pytest.raises(ValueError, match="pdf content"):
        _content_part_to_hf({"type": "pdf", "file_path": "/tmp/sample.pdf"})


def test_turn_to_hf_force_rich_content_for_text():
    turn = ChatTurn(role="user", content="Describe the image.")
    assert _turn_to_hf(turn, force_rich_content=False) == {
        "role": "user",
        "content": "Describe the image.",
    }
    assert _turn_to_hf(turn, force_rich_content=True) == {
        "role": "user",
        "content": [{"type": "text", "text": "Describe the image."}],
    }


def test_config_supports_multimodal_detects_vision_and_audio():
    vision_cfg = SimpleNamespace(
        vision_config=object(),
        audio_config=None,
        video_config=None,
        image_config=None,
        sub_configs={},
        model_type="qwen2_5_vl",
        architectures=["Qwen2_5_VLForConditionalGeneration"],
    )
    assert _config_supports_multimodal(vision_cfg)

    audio_cfg = SimpleNamespace(
        vision_config=None,
        audio_config=object(),
        video_config=None,
        image_config=None,
        sub_configs={},
        model_type="qwen2_audio",
        architectures=["Qwen2AudioForConditionalGeneration"],
    )
    assert _config_supports_multimodal(audio_cfg)

    text_cfg = SimpleNamespace(
        vision_config=None,
        audio_config=None,
        video_config=None,
        image_config=None,
        sub_configs={},
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
    )
    assert not _config_supports_multimodal(text_cfg)


def test_hf_generator_forwards_device_map(monkeypatch):
    calls = {}

    def fake_load_hf_model(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr(
        hf_generator_module, "_resolve_model_type", lambda cfg: "causal_lm"
    )
    monkeypatch.setattr(hf_generator_module, "load_hf_model", fake_load_hf_model)
    monkeypatch.setattr(HFGenerator, "_patch_model", lambda self: None)

    HFGenerator(
        HFGeneratorConfig(
            model_path="Qwen/Qwen3-0.6B",
            device_map={"": 0},
        )
    )

    assert calls["device_map"] == {"": 0}
