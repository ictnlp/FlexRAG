import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from omegaconf import MISSING

from flexrag.utils import TIME_METER

from .model_base import ENCODERS, EncoderBase


@dataclass
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
    :param model_kwargs: Additional keyword arguments for loading the model. Defaults to {}.
    :type model_kwargs: dict[str, Any]
    """

    model_path: str = MISSING
    device_id: list[int] = field(default_factory=list)
    trust_remote_code: bool = False
    task: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt: Optional[str] = None
    prompt_dict: Optional[dict] = None
    normalize: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@ENCODERS("sentence_transformer", config_class=SentenceTransformerEncoderConfig)
class SentenceTransformerEncoder(EncoderBase):
    def __init__(self, config: SentenceTransformerEncoderConfig) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.devices = config.device_id
        self.model = SentenceTransformer(
            model_name_or_path=config.model_path,
            device=f"cuda:{config.device_id[0]}" if config.device_id else "cpu",
            trust_remote_code=config.trust_remote_code,
            backend="torch",
            prompts=config.prompt_dict,
            model_kwargs=config.model_kwargs,
        )
        if len(config.device_id) > 1:
            self.pool = self.model.start_multi_process_pool(
                target_devices=[f"cuda:{i}" for i in config.device_id]
            )
        else:
            self.pool = None

        # set args
        self.prompt_name = config.prompt_name
        self.task = config.task
        self.prompt = config.prompt
        self.normalize = config.normalize
        return

    @TIME_METER("st_encode")
    def _encode(self, texts: list[str], **kwargs) -> np.ndarray:
        args = {
            "sentences": texts,
            "batch_size": len(texts),
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
        if (len(texts) >= len(self.devices) * 8) and (self.pool is not None):
            args["pool"] = self.pool
            args["batch_size"] = math.ceil(args["batch_size"] / len(self.devices))
            embeddings = self.model.encode_multi_process(**args)
        else:
            embeddings = self.model.encode(**args)
        return embeddings

    @property
    def embedding_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()
