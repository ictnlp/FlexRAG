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

    :param max_encode_length: The maximum length of the input sequence. Default is None.
    :type max_encode_length: Optional[int]
    :param encode_method: The method to get the embedding. Default is "mean".
        Available choices:

        - `cls`: Use the [CLS] token representation.
        - `mean`: Use the mean pooling of all token representations.
        - `last`: Use the last token representation (usually used in decoder-only models).
        - `late`: Use `Late Chunking <https://arxiv.org/abs/2409.04701>`_ to get the embeddings.
          If this method is chosen, the input texts will be concatenated as a single document.
          The final embeddings will be computed by mean pooling over each text chunk hiddens.
    :type encode_method: str
    :param normalize: Whether to normalize the embedding. Default is False.
    :type normalize: bool
    :param prefix: A string prefix before encoding query / passage. Default is None.
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

    max_encode_length: Optional[int] = None
    encode_method: Annotated[str, Choices("cls", "mean", "last", "late")] = "mean"
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

        # setup arguments
        self.encode_method = cfg.encode_method
        self.normalize = cfg.normalize
        self.prefix = cfg.prefix
        self.task = cfg.task
        self.encoding_args = {
            "padding": True,
        }
        if cfg.max_encode_length is not None:
            self.encoding_args["max_length"] = cfg.max_encode_length
            self.encoding_args["truncation"] = True
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

        # for late chunking
        if self.encode_method == "late":
            return self.contextual_encode(texts)

        # add prefix if needed
        if self.prefix:
            texts = [self.prefix + text for text in texts]

        # prepare input_dict
        # PERFORMANCE NOTE: tokenize takes significant time for encoding.
        input_dict = self.tokenizer(
            texts,
            return_tensors="pt",
            **self.encoding_args,
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

    @torch.no_grad()
    def contextual_encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using late chunking method.

        :param texts: A list of input texts.
            All inputs will be considered as chunks of a single document.
        :type texts: list[str]
        :return: The encoded embeddings.
        :rtype: np.ndarray
        """
        # prepare input text with prefix
        chunk_boundaries = []
        full_text = ""
        for idx, text in enumerate(texts):
            if idx == 0 and self.prefix:
                full_text += self.prefix + " " + text
                chunk_boundaries.append((len(self.prefix) + 1, len(full_text)))
            elif idx == 0:
                full_text += text
                chunk_boundaries.append((0, len(full_text)))
            else:
                full_text += " " + text
                chunk_boundaries.append(
                    (len(full_text) - len(text) - 1, len(full_text))
                )

        # tokenize full text
        encoding = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            **self.encoding_args,
        )

        # prepare chunk ids
        chunk_ids = []
        current_chunk_idx = 0
        for start, end in encoding["offset_mapping"]:
            if start == end == 0:
                chunk_ids.append(0)  # special token
                continue
            while (start > chunk_boundaries[current_chunk_idx][1]) and (
                current_chunk_idx < len(chunk_boundaries)
            ):
                current_chunk_idx += 1
            if (
                (current_chunk_idx < len(chunk_boundaries))
                and (end >= chunk_boundaries[current_chunk_idx][0])
                and (start <= chunk_boundaries[current_chunk_idx][1])
            ):
                chunk_ids.append(current_chunk_idx + 1)  # chunk_id starts from 1
            else:
                chunk_ids.append(0)  # space between chunks or prefix or suffix tokens.
        chunk_ids = torch.tensor(chunk_ids, dtype=torch.long, device=self.model.device)
        input_ids = torch.tensor(
            encoding["input_ids"], device=self.model.device, dtype=torch.long
        ).unsqueeze(0)
        attn_mask = torch.tensor(
            encoding["attention_mask"], device=self.model.device, dtype=torch.long
        ).unsqueeze(0)
        input_dict = {"input_ids": input_ids, "attention_mask": attn_mask}

        # for jina embedding v3
        if hasattr(self.model, "_adaptation_map") and (self.task is not None):
            task_id = self.model._adaptation_map.get(self.task, None)
            if task_id is not None:
                input_dict["adapter_mask"] = torch.full(
                    (1,), task_id, dtype=torch.int32, device=self.model.device
                )

        # forward for hiddens
        outputs = self.model(**input_dict)
        hidden_states = outputs.last_hidden_state.squeeze(0)  # [seq, hidden_dim]

        # mean pooling for each chunk
        embeddings = torch.zeros(
            (len(texts) + 1, hidden_states.size(-1)),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        # index_reduce_ is in beta, use index_add_ for stability.
        embeddings.index_add_(0, chunk_ids, hidden_states)
        token_counts = torch.bincount(chunk_ids)[1:]
        embeddings = embeddings[1:] / torch.clamp(token_counts, min=1).unsqueeze(1)

        # normalize
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings.float().cpu().numpy()

    @property
    def embedding_size(self) -> int:
        return self.model.config.hidden_size


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
