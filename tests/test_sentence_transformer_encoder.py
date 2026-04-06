from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from flexrag.models.encoders.sentence_transformers_model import (
    SentenceTransformerEncoderConfig,
    SentenceTransformerEncoderImpl,
)


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []

    def encode(self, *, sentences, **kwargs):
        modality = "text" if sentences and isinstance(sentences[0], str) else "image"
        self.calls.append(
            {
                "modality": modality,
                "sentences": sentences,
                "kwargs": kwargs,
            }
        )
        base = 1.0 if modality == "text" else 101.0
        return np.array(
            [[base + idx, base + idx + 0.5] for idx in range(len(sentences))],
            dtype=np.float32,
        )

    def get_sentence_embedding_dimension(self):
        return 2


@pytest.fixture
def fake_sentence_transformer(mocker):
    import sentence_transformers

    holder: dict[str, FakeSentenceTransformer] = {}

    def factory(*args, **kwargs):
        instance = FakeSentenceTransformer(*args, **kwargs)
        holder["instance"] = instance
        return instance

    mocker.patch.object(
        sentence_transformers, "SentenceTransformer", side_effect=factory
    )
    return holder


def test_sentence_transformer_impl_mixed_text_image_batch(
    fake_sentence_transformer, tmp_path
):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color="red").save(image_path)

    impl = SentenceTransformerEncoderImpl(
        SentenceTransformerEncoderConfig(
            model_path="mock-model",
            task="retrieval",
            prompt_name="query",
            prompt="Represent this sentence",
        )
    )
    mixed_embeddings = impl.encode(
        [
            {"type": "text", "text": "Bruce Wayne"},
            {"type": "image", "image_path": str(image_path)},
            {"type": "text", "text": "Thomas Wayne"},
        ]
    )

    assert mixed_embeddings.shape == (3, 2)
    fake_model = fake_sentence_transformer["instance"]
    assert len(fake_model.calls) == 2
    assert fake_model.calls[0]["modality"] == "text"
    assert fake_model.calls[0]["kwargs"]["task"] == "retrieval"
    assert fake_model.calls[0]["kwargs"]["prompt_name"] == "query"
    assert fake_model.calls[0]["kwargs"]["prompt"] == "Represent this sentence"
    assert fake_model.calls[1]["modality"] == "image"
    assert "task" not in fake_model.calls[1]["kwargs"]
    assert np.allclose(mixed_embeddings[0], np.array([1.0, 1.5], dtype=np.float32))
    assert np.allclose(mixed_embeddings[1], np.array([101.0, 101.5], dtype=np.float32))
    assert np.allclose(mixed_embeddings[2], np.array([2.0, 2.5], dtype=np.float32))


def test_sentence_transformer_impl_loads_url_images_in_memory(
    fake_sentence_transformer, mocker
):
    image = Image.new("RGB", (4, 4), color="blue")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    mocker.patch(
        "flexrag.models.encoders.sentence_transformers_model.requests.get",
        return_value=SimpleNamespace(
            content=buffer.getvalue(),
            raise_for_status=lambda: None,
        ),
    )

    impl = SentenceTransformerEncoderImpl(
        SentenceTransformerEncoderConfig(model_path="mock-model")
    )
    embeddings = impl.encode([{"type": "image", "url": "https://example.com/a.png"}])
    assert embeddings.shape == (1, 2)
    fake_model = fake_sentence_transformer["instance"]
    assert fake_model.calls[0]["modality"] == "image"
    assert fake_model.calls[0]["sentences"][0].size == (4, 4)


def test_sentence_transformer_impl_rejects_non_text_image_content(
    fake_sentence_transformer,
):
    impl = SentenceTransformerEncoderImpl(
        SentenceTransformerEncoderConfig(model_path="mock-model")
    )
    with pytest.raises(
        ValueError,
        match="SentenceTransformerEncoder only supports text and image content blocks",
    ):
        impl.encode([{"type": "audio", "url": "https://example.com/a.mp3"}])
