from dataclasses import field
from functools import cached_property
from typing import Annotated, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ImageFile import ImageFile
from transformers import CLIPModel, PreTrainedTokenizer

from flexrag.common import LOGGER_MANAGER, Choices, ContentPart, configure, trace

from ..hf_utils import HFModelConfig, load_hf_model
from .encoder_base import ENCODERS, LocalEncoderBase

logger = LOGGER_MANAGER.get_logger("flexrag.models.hf_model")


_HF_ENCODE_METHODS = ("cls", "mean", "mean_without_prefix", "last")


@configure
class HFEncoderConfig(HFModelConfig):
    """Configuration for HFEncoder.

    :param max_encode_length: The maximum length of the input sequence. Default is None.
    :param encode_method: The method to get the embedding. Default is "mean".
        Available choices:

        - `cls`: Use the [CLS] token representation.
        - `mean`: Use the mean pooling of all token representations.
        - `mean_without_prefix`: Use the mean pooling of all token representations without
          considering the prefix tokens (only for models with prefix).
        - `last`: Use the last token representation (usually used in decoder-only models).
    :param normalize: Whether to normalize the embedding. Default is False.
    :param batch_size: Maximum direct-use batch size. This value is ignored
        when the encoder is created through Runtime or ResourceManager;
        configure the runtime or resource batch size instead.
    :param prefix: A string prefix before encoding query / passage. Default is None.
    :param task: The task to use. Default is None.
    :param other_tokenizer_kwargs: Other keyword arguments for tokenizer. Default is empty dict.

    For example, if you want to use the Qwen3-Embedding-0.6B model as an query encoder,
    you can use the following code:

    .. code-block:: python
        from flexrag.models import HFEncoder, HFEncoderConfig

        prefix = (
            'Instruct: Given a web search query, retrieve'
            'relevant passages that answer the query\nQuery:'
        )

        query_encoder = HFEncoder(
            HFEncoderConfig(
                model_path="Qwen/Qwen3-Embedding-0.6B",
                device_map=0,
                prefix=prefix,
                normalize=True,
                encode_method="last",
            )
        )
        emb = query_encoder.encode(["Who is Bruce Wayne?"])
    """

    max_encode_length: Optional[int] = None
    encode_method: Annotated[
        str,
        Choices(*_HF_ENCODE_METHODS),
    ] = "mean"
    normalize: bool = False
    batch_size: int = 32
    prefix: Optional[str] = None  # used in nomic-text-embedding
    task: Optional[str] = None  # used in jina-embedding
    other_tokenizer_kwargs: dict = field(default_factory=dict)


@ENCODERS("hf", config_class=HFEncoderConfig)
class HFEncoder(LocalEncoderBase):
    def __init__(self, cfg: HFEncoderConfig):
        if cfg.encode_method not in _HF_ENCODE_METHODS:
            available_methods = ", ".join(_HF_ENCODE_METHODS)
            raise ValueError(
                f"Unsupported HFEncoder encode_method '{cfg.encode_method}'. "
                f"Available choices are: {available_methods}."
            )
        super().__init__(batch_size=cfg.batch_size)

        # load model
        self.model, self.tokenizer = load_hf_model(
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            load_dtype=cfg.load_dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
            other_tokenizer_kwargs=cfg.other_tokenizer_kwargs,
        )

        # setup arguments
        self.encode_method = cfg.encode_method
        self.normalize = cfg.normalize
        self.prefix = cfg.prefix
        self.task = cfg.task
        self.max_encoding_length = cfg.max_encode_length
        return

    def get_embedding(
        self, hidden: torch.Tensor, attn_mask: torch.Tensor
    ) -> np.ndarray:
        match self.encode_method:
            case "mean":
                attn_mask = attn_mask.to(hidden.device)
                embeddings = hidden.masked_fill(~attn_mask[..., None].bool(), 0.0)
                embeddings = embeddings.sum(dim=1) / torch.clamp(
                    attn_mask.sum(dim=1)[..., None], min=1e-6
                )
            case "mean_without_prefix":
                attn_mask = attn_mask.to(hidden.device)
                if self.prefix_length > 0:
                    attn_mask[:, : self.prefix_length] = 0
                embeddings = hidden.masked_fill(~attn_mask[..., None].bool(), 0.0)
                embeddings = embeddings.sum(dim=1) / torch.clamp(
                    attn_mask.sum(dim=1)[..., None], min=1e-6
                )
            case "cls":
                embeddings = hidden[:, 0]
            case "last":
                left_padding = attn_mask[:, -1].sum() == attn_mask.shape[0]
                if left_padding:
                    embeddings = hidden[:, -1]
                else:
                    sequence_lengths = attn_mask.sum(dim=1) - 1
                    batch_size = hidden.shape[0]
                    embeddings = hidden[
                        torch.arange(batch_size, device=hidden.device),
                        sequence_lengths,
                    ]
            case _:
                raise ValueError(f"Unsupported encode method: {self.encode_method}")
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings.float().cpu().numpy()

    @trace("encoder.hf_encode")
    @torch.no_grad()
    def _encode_batch(self, inputs: list[ContentPart]) -> np.ndarray:
        texts: list[str] = []
        for part in inputs:
            if part.get("type") != "text":
                raise ValueError(
                    "HFEncoder only supports text content blocks, "
                    f"but got '{part.get('type')}'."
                )
            text = part.get("text", "")
            if not isinstance(text, str):
                raise ValueError("HFEncoder text content must be a string.")
            texts.append(text)

        # add prefix if needed
        if self.prefix:
            texts = [self.prefix + text for text in texts]

        # prepare input_dict
        # PERFORMANCE NOTE: tokenize takes significant time for encoding.
        encoding_args = {"padding": True}
        if self.max_encoding_length is not None:
            encoding_args["max_length"] = self.max_encoding_length
            encoding_args["truncation"] = True
        input_dict = self.tokenizer(
            texts,
            return_tensors="pt",
            **encoding_args,
        )
        # for jina-embedding v3
        if hasattr(self.model, "_adaptation_map") and (self.task is not None):
            task_id = self.model._adaptation_map.get(self.task, None)
            if task_id is not None:
                input_dict["adapter_mask"] = torch.full(
                    (len(texts),), task_id, dtype=torch.int32
                )
        input_dict = input_dict.to(self.model.device)

        # get hidden states
        mask = input_dict["attention_mask"]
        output = self.model(**input_dict).last_hidden_state

        # get embeddings
        embeddings = self.get_embedding(output, mask)
        return embeddings

    @property
    def embedding_size(self) -> int:
        return self.model.config.hidden_size

    @cached_property
    def prefix_length(self) -> int:
        if self.prefix:
            prefix_toks = self.tokenizer(self.prefix.rstrip())["input_ids"]
            if (
                hasattr(self.tokenizer, "all_special_ids")
                and prefix_toks[-1] in self.tokenizer.all_special_ids
            ):
                prefix_lengths = len(prefix_toks) - 1
            else:
                prefix_lengths = len(prefix_toks)
        else:
            prefix_lengths = 0
        return prefix_lengths


@configure
class HFClipEncoderConfig(HFModelConfig):
    """Configuration for HFClipEncoder.

    :param max_encode_length: The maximum length of the input sequence. Default is 512.
    :type max_encode_length: int
    :param normalize: Whether to normalize the embedding. Default is False.
    :type normalize: bool
    :param batch_size: Maximum direct-use batch size. This value is ignored
        when the encoder is created through Runtime or ResourceManager;
        configure the runtime or resource batch size instead.
    :param convert_to_rgb: Whether to convert the image to RGB. Default is False.
    :type convert_to_rgb: bool
    """

    max_encode_length: int = 512
    normalize: bool = False
    batch_size: int = 32
    convert_to_rgb: bool = False


@ENCODERS("hf_clip", config_class=HFClipEncoderConfig)
class HFClipEncoder(LocalEncoderBase):
    model: CLIPModel
    tokenizer: PreTrainedTokenizer

    def __init__(self, cfg: HFClipEncoderConfig):
        super().__init__(batch_size=cfg.batch_size)

        self.model, (self.tokenizer, self.processor) = load_hf_model(
            model_type="clip",
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            load_dtype=cfg.load_dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
        )

        # setup arguments
        self.max_encode_length = cfg.max_encode_length
        self.normalize = cfg.normalize
        self.convert_to_rgb = cfg.convert_to_rgb
        return

    def _extract_feature_tensor(self, embeddings):
        if isinstance(embeddings, torch.Tensor):
            return embeddings
        pooler_output = getattr(embeddings, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        raise TypeError(f"Unsupported CLIP feature output type: {type(embeddings)}")

    def _resolve_image_part(self, content_part: ContentPart) -> Image.Image:
        if content_part.get("type") != "image":
            raise ValueError(
                "HFClipEncoder only supports text and image content blocks."
            )
        if content_part.get("image") is not None:
            return content_part["image"]  # type: ignore
        if content_part.get("image_path") is not None:
            image = Image.open(content_part["image_path"])  # type: ignore
            image.load()
            return image
        if content_part.get("url") is not None:
            raise ValueError("HFClipEncoder does not support remote image URLs.")
        raise ValueError(
            "Image content must have either 'image' or 'image_path' for HFClipEncoder."
        )

    def _encode_batch(self, inputs: list[ContentPart]) -> np.ndarray:
        if not inputs:
            return np.empty((0, self.embedding_size), dtype=np.float32)

        text_indices: list[int] = []
        texts: list[str] = []
        image_indices: list[int] = []
        images: list[ImageFile] = []

        for idx, part in enumerate(inputs):
            part_type = part.get("type")
            if part_type == "text":
                text_indices.append(idx)
                texts.append(part.get("text", ""))
            elif part_type == "image":
                image_indices.append(idx)
                images.append(self._resolve_image_part(part))
            else:
                raise ValueError(
                    f"HFClipEncoder only supports text and image content blocks, "
                    f"but got '{part_type}'."
                )

        results: list[np.ndarray | None] = [None] * len(inputs)
        if texts:
            text_embeddings = self._encode_text(texts)
            for idx, embedding in zip(text_indices, text_embeddings, strict=True):
                results[idx] = embedding
        if images:
            image_embeddings = self._encode_image(images)
            for idx, embedding in zip(image_indices, image_embeddings, strict=True):
                results[idx] = embedding

        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError("Some HFClipEncoder inputs did not produce embeddings.")
        return np.stack(ready_results, axis=0)

    @trace("encoder.hf_clip_encode")
    @torch.no_grad()
    def _encode_image(self, images: list[Image.Image]) -> np.ndarray:
        if self.convert_to_rgb:
            images = [img.convert("RGB") for img in images]
        input_dict = self.processor(images=images, return_tensors="pt")
        input_dict = input_dict.to(self.model.device)
        embeddings = self.model.get_image_features(**input_dict)
        embeddings = self._extract_feature_tensor(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings.float().cpu().numpy()

    @trace("encoder.hf_clip_encode")
    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> np.ndarray:
        input_dict = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_encode_length,
            padding=True,
            truncation=True,
        )
        input_dict = input_dict.to(self.model.device)
        embeddings = self.model.get_text_features(**input_dict)
        embeddings = self._extract_feature_tensor(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings.float().cpu().numpy()

    @property
    def embedding_size(self) -> int:
        if hasattr(self.model.config, "projection_dim"):
            return self.model.config.projection_dim
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        raise ValueError("Cannot determine embedding size from model config.")
