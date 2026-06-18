import math
from dataclasses import field
from io import BytesIO
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image

from flexrag.common import ContentPart, configure, trace

from .encoder_base import ENCODERS, LocalEncoderBase


@configure
class SentenceTransformerEncoderConfig:
    """Configuration for SentenceTransformerEncoder.

    :param model_path: The path to the model. Required.
    :type model_path: str
    :param device_id: The device id to use. [] for CPU. Defaults to [].
    :type device_id: list[int]
    :param trust_remote_code: Whether to trust remote code. Defaults to False.
    :type trust_remote_code: bool
    :param task: The task to use. Defaults to None.
    :type task: Optional[str]
    :param prompt_name: The prompt name to use. Defaults to None.
    :type prompt_name: Optional[str]
    :param prompt: The prompt to use. Defaults to None.
    :type prompt: Optional[str]
    :param prompt_dict: The prompt dictionary to use. Defaults to None.
    :type prompt_dict: Optional[dict]
    :param normalize: Whether to normalize embeddings. Defaults to False.
    :type normalize: bool
    :param batch_size: Maximum direct-use batch size. This value is ignored
        when the encoder is created through Runtime or ResourceManager;
        configure the runtime or resource batch size instead.
    :param model_kwargs: Additional keyword arguments for loading the model. Defaults to {}.
    :type model_kwargs: dict[str, Any]
    """

    model_path: Optional[str] = None
    device_id: list[int] = field(default_factory=list)
    trust_remote_code: bool = False
    task: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt: Optional[str] = None
    prompt_dict: Optional[dict] = None
    normalize: bool = False
    batch_size: int = 32
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@ENCODERS("sentence_transformer", config_class=SentenceTransformerEncoderConfig)
class SentenceTransformerEncoder(LocalEncoderBase):
    def __init__(self, config: SentenceTransformerEncoderConfig) -> None:
        from sentence_transformers import SentenceTransformer

        super().__init__(batch_size=config.batch_size)

        self.devices = config.device_id
        assert config.model_path is not None, "`model_path` must be provided"
        self.model = SentenceTransformer(
            model_name_or_path=config.model_path,
            device=f"cuda:{config.device_id[0]}" if config.device_id else "cpu",
            trust_remote_code=config.trust_remote_code,
            backend="torch",
            prompts=config.prompt_dict,
            model_kwargs=config.model_kwargs,
        )

        # set args
        self.prompt_name = config.prompt_name
        self.task = config.task
        self.prompt = config.prompt
        self.normalize = config.normalize
        return

    def _resolve_image_part(self, content_part: ContentPart) -> Image.Image:
        if content_part.get("type") != "image":
            raise ValueError(
                "SentenceTransformerEncoder only supports text and image content blocks."
            )
        if content_part.get("image") is not None:
            return content_part["image"]  # type: ignore
        if content_part.get("image_path") is not None:
            image = Image.open(content_part["image_path"])  # type: ignore
            image.load()
            return image
        if content_part.get("url") is not None:
            response = requests.get(content_part["url"], timeout=30)  # type: ignore
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.load()
            return image
        raise ValueError(
            "Image content must have either 'image', 'image_path', or 'url' "
            "for SentenceTransformerEncoder."
        )

    def _encode_texts(self, texts: list[str], **kwargs) -> np.ndarray:
        args = {
            "sentences": texts,
            "batch_size": math.ceil(len(texts) / max(1, len(self.devices))),
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize,
        }
        if kwargs.get("task", self.task) is not None:
            args["task"] = self.task
        if kwargs.get("prompt_name", self.prompt_name) is not None:
            args["prompt_name"] = self.prompt_name
        if kwargs.get("prompt", self.prompt) is not None:
            args["prompt"] = self.prompt
        return self.model.encode(**args)

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        args = {
            "sentences": images,
            "batch_size": math.ceil(len(images) / max(1, len(self.devices))),
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize,
        }
        return self.model.encode(**args)

    @trace("encoder.st_encode")
    def _encode_batch(self, inputs: list[ContentPart], **kwargs) -> np.ndarray:
        text_indices: list[int] = []
        texts: list[str] = []
        image_indices: list[int] = []
        images: list[Image.Image] = []

        for idx, part in enumerate(inputs):
            part_type = part.get("type")
            if part_type == "text":
                text_indices.append(idx)
                texts.append(part.get("text", ""))
                continue
            if part_type == "image":
                image_indices.append(idx)
                images.append(self._resolve_image_part(part))
                continue
            raise ValueError(
                "SentenceTransformerEncoder only supports text and image content blocks, "
                f"but got '{part_type}'."
            )

        results: list[np.ndarray | None] = [None] * len(inputs)
        if texts:
            text_embeddings = self._encode_texts(texts, **kwargs)
            for idx, embedding in zip(text_indices, text_embeddings, strict=True):
                results[idx] = embedding
        if images:
            image_embeddings = self._encode_images(images)
            for idx, embedding in zip(image_indices, image_embeddings, strict=True):
                results[idx] = embedding

        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError(
                "Some SentenceTransformerEncoder inputs did not produce embeddings."
            )
        return np.stack(ready_results, axis=0)

    @property
    def embedding_size(self) -> int:
        return self.model.get_embedding_dimension()
