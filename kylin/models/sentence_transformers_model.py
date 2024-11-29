from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from omegaconf import MISSING

from .model_base import ENCODERS, EncoderBase, EncoderBaseConfig


@dataclass
class SentenceTransformerEncoderConfig(EncoderBaseConfig):
    model_path: str = MISSING
    device_id: list[int] = field(default_factory=list)
    trust_remote_code: bool = False
    task: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt: Optional[str] = None
    prompt_dict: Optional[dict] = None


@ENCODERS("sentence_transformer", config_class=SentenceTransformerEncoderConfig)
class SentenceTransformerEncoder(EncoderBase):
    def __init__(self, config: SentenceTransformerEncoderConfig) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name_or_path=config.model_path,
            device=f"cuda:{config.device_id[0]}" if config.device_id else "cpu",
            trust_remote_code=config.trust_remote_code,
            backend="torch",
            prompts=config.prompt_dict,
        )

        # set args
        self.prompt_name = config.prompt_name
        self.task = config.task
        self.prompt = config.prompt
        return

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        args = {
            "sentences": texts,
            "batch_size": len(texts),
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        if kwargs.get("task", self.task) is not None:
            args["task"] = self.task
        if kwargs.get("prompt_name", self.prompt_name) is not None:
            args["prompt_name"] = self.prompt_name
        if kwargs.get("prompt", self.prompt) is not None:
            args["prompt"] = self.prompt
        embeddings = self.model.encode(**args)
        return embeddings

    @property
    def embedding_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()
