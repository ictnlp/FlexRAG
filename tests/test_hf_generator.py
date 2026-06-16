from types import SimpleNamespace

import pytest

from flexrag.common.dataclasses import ChatTurn
from flexrag.models.generators.hf_generator import (
    HFGenerator,
    HFGeneratorConfig,
    _config_supports_multimodal,
    _content_part_to_hf,
    _turn_to_hf,
)
from flexrag.runtime.process_worker import build_worker_config


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


def test_build_worker_config_normalizes_multi_gpu_device_ids():
    cfg = build_worker_config(
        "flexrag.models.generators.hf_generator:HFGeneratorConfig",
        {
            "model_path": "Qwen/Qwen3-0.6B",
            "tokenizer_path": None,
            "trust_remote_code": False,
            "device_id": [2, 5],
            "load_dtype": "auto",
            "parallel_mode": "pipeline",
            "model_type": "auto",
        },
        [2, 5],
    )
    assert cfg.device_id == [0, 1]


def test_hf_generator_build_worker_device_groups_data_mode():
    generator = HFGenerator(
        HFGeneratorConfig(
            model_path="Qwen/Qwen3-0.6B",
            device_id=[0, 1, 2],
            parallel_mode="data",
        )
    )
    try:
        groups = generator._build_worker_device_groups(generator._config)
        assert groups == [[0], [1], [2]]
    finally:
        generator.close()


def test_hf_generator_build_worker_device_groups_pipeline_mode():
    generator = HFGenerator(
        HFGeneratorConfig(
            model_path="Qwen/Qwen3-0.6B",
            device_id=[0, 1, 2],
            parallel_mode="pipeline",
        )
    )
    try:
        groups = generator._build_worker_device_groups(generator._config)
        assert groups == [[0, 1, 2]]
    finally:
        generator.close()
