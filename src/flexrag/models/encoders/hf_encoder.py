import asyncio
from typing import Annotated, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL.ImageFile import ImageFile
from torch.nn.parallel import DataParallel as DP
from transformers import CLIPModel, PreTrainedTokenizer

from flexrag.common import LOGGER_MANAGER, TIME_METER, Choices, configure

from ..hf_utils import HFModelConfig, load_hf_model
from .encoder_base import ENCODERS, EncoderBase

logger = LOGGER_MANAGER.get_logger("flexrag.models.hf_model")


@configure
class HFEncoderConfig(HFModelConfig):
    """Configuration for HFEncoder.

    :param max_encode_length: The maximum length of the input sequence. Default is 512.
    :type max_encode_length: int
    :param encode_method: The method to get the embedding. Default is "mean". Available choices are "cls", "mean".
    :type encode_method: str
    :param normalize: Whether to normalize the embedding. Default is False.
    :type normalize: bool
    :param prefix: A Python prefix before encoding query / passage. Default is None.
        The `query` variable will be replaced with the input text.
    :type prefix: Optional[str]
    :param task: The task to use. Default is None.
    :type task: Optional[str]


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
                device_id=[0],
                prefix=prefix,
                normalize=True,
                encode_method="last",
            )
        )
        emb = query_encoder.encode(["Who is Bruce Wayne?"])
    """

    max_encode_length: int = 512
    encode_method: Annotated[str, Choices("cls", "mean", "last")] = "mean"
    normalize: bool = False
    prefix: Optional[str] = None  # used in nomic-text-embedding
    task: Optional[str] = None  # used in jina-embedding


@ENCODERS("hf", config_class=HFEncoderConfig)
class HFEncoder(EncoderBase):
    def __init__(self, cfg: HFEncoderConfig):
        # load model
        self.model, self.tokenizer = load_hf_model(
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            load_dtype=cfg.load_dtype,
            device_id=cfg.device_id,
            trust_remote_code=cfg.trust_remote_code,
        )
        # setup model
        self.devices = cfg.device_id
        if len(self.devices) > 1:
            if self.is_jina:
                logger.warning("Data parallel does not support self implemented model.")
                self.dp_model = None
            elif not torch.cuda.is_available():
                logger.warning("Data parallel is not supported on CPU.")
                self.dp_model = None
            elif torch.cuda.device_count() <= max(self.devices):
                logger.warning(
                    f"Invalid device ids: {self.devices}. Using single device mode."
                )
                self.dp_model = None
            else:
                self.dp_model = DP(self.model, device_ids=self.devices)
        else:
            self.dp_model = None

        # setup arguments
        self.max_encode_length = cfg.max_encode_length
        self.encode_method = cfg.encode_method
        self.normalize = cfg.normalize
        self.prefix = cfg.prefix
        self.task = cfg.task
        return

    def get_embedding(
        self, hidden: torch.Tensor, attn_mask: torch.Tensor
    ) -> np.ndarray:
        match self.encode_method:
            case "mean":
                attn_mask = attn_mask.to(hidden.device)
                embeddings = hidden.masked_fill(~attn_mask[..., None].bool(), 0.0)
                embeddings = embeddings.sum(dim=1) / attn_mask.sum(dim=1)[..., None]
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

    @TIME_METER("encoder.hf_encode")
    @torch.no_grad()
    def encode(self, texts: str | list[str]) -> np.ndarray:
        texts = texts if isinstance(texts, list) else [texts]
        if self.is_jina:  # for jina-embedding
            return self.model.encode(
                texts,
                task=self.task,
                max_length=self.max_encode_length,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # add prompt if needed
        if self.prefix:
            texts = [self.prefix + text for text in texts]

        # prepare encoder
        if (len(texts) >= len(self.devices) * 8) and (self.dp_model is not None):
            encoder = self.dp_model
        else:
            encoder = self.model

        # encode
        # PERFORMANCE NOTE: tokenize takes significant time for encoding.
        input_dict = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=self.max_encode_length,
            padding=True,
            truncation=True,
        )
        if not isinstance(encoder, DP):
            input_dict = input_dict.to(encoder.device)
        mask = input_dict["attention_mask"]
        output = encoder(**input_dict).last_hidden_state
        embeddings = self.get_embedding(output, mask)
        return embeddings

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self.encode, texts)

    @property
    def embedding_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def is_jina(self) -> bool:
        return self.model.__class__.__name__ == "XLMRobertaLoRA" and hasattr(
            self.model, "encode"
        )


@configure
class HFClipEncoderConfig(HFModelConfig):
    """Configuration for HFClipEncoder.

    :param max_encode_length: The maximum length of the input sequence. Default is 512.
    :type max_encode_length: int
    :param normalize: Whether to normalize the embedding. Default is False.
    :type normalize: bool
    :param convert_to_rgb: Whether to convert the image to RGB. Default is False.
    :type convert_to_rgb: bool
    """

    max_encode_length: int = 512
    normalize: bool = False
    convert_to_rgb: bool = False


@ENCODERS("hf_clip", config_class=HFClipEncoderConfig)
class HFClipEncoder(EncoderBase):
    model: CLIPModel
    tokenizer: PreTrainedTokenizer

    def __init__(self, cfg: HFClipEncoderConfig):
        super().__init__(cfg)
        self.devices = cfg.device_id
        # load model
        self.model, (self.tokenizer, self.processor) = load_hf_model(
            model_type="clip",
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            load_dtype=cfg.load_dtype,
            device_id=cfg.device_id,
            trust_remote_code=cfg.trust_remote_code,
        )

        # setup arguments
        self.max_encode_length = cfg.max_encode_length
        self.normalize = cfg.normalize
        self.convert_to_rgb = cfg.convert_to_rgb
        return

    def _encode(self, data: list[str | ImageFile]) -> np.ndarray:
        if isinstance(data[0], str):
            assert all(isinstance(d, str) for d in data)
            return self.encode_text(data)
        assert all(isinstance(d, ImageFile) for d in data)
        return self.encode_image(data)

    @TIME_METER("encoder.hf_clip_encode")
    @torch.no_grad()
    def encode_image(self, images: list[ImageFile]) -> np.ndarray:
        if self.convert_to_rgb:
            images = [img.convert("RGB") for img in images]
        input_dict = self.processor(images=images, return_tensors="pt")
        input_dict = input_dict.to(self.model.device)
        embeddings = self.model.get_image_features(**input_dict)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings.float().cpu().numpy()

    @TIME_METER("encoder.hf_clip_encode")
    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        input_dict = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=self.max_encode_length,
            padding=True,
            truncation=True,
        )
        input_dict = input_dict.to(self.model.device)
        embeddings = self.model.get_text_features(**input_dict)
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
