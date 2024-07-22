import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from omegaconf import MISSING
from torch.nn.parallel import DataParallel as DP
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig as HFGenerationConfig

from kylin.utils import Choices

from .model_base import (
    Encoders,
    EncoderBase,
    EncoderBaseConfig,
    Generators,
    GenerationConfig,
    GeneratorBase,
    GeneratorBaseConfig,
)
from .utils import get_prompt_func

logger = logging.getLogger(__name__)


@dataclass
class HFGeneratorConfig(GeneratorBaseConfig):
    model_path: str = MISSING
    tokenizer_path: Optional[str] = None
    pipeline_parallel: bool = False
    load_dtype: Choices(  # type: ignore
        [
            "bfloat16",
            "float32",
            "float16",
            "8bit",
            "4bit",
            "auto",
        ]
    ) = "auto"


@Generators("hf", config_class=HFGeneratorConfig)
class HFGenerator(GeneratorBase):
    def __init__(self, cfg: HFGeneratorConfig) -> None:
        # prepare gpu
        if cfg.pipeline_parallel:
            device_map = "auto"
        elif torch.cuda.is_available():
            device_map = 0
        else:
            device_map = None

        # prepare dtype
        match cfg.load_dtype:
            case "bfloat16":
                load_dtype = torch.bfloat16
            case "float32":
                load_dtype = torch.float32
            case "float16":
                load_dtype = torch.float16
            case "8bit":
                load_dtype = None
            case "4bit":
                load_dtype = None
            case "auto":
                load_dtype = "auto"

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map=device_map,
            torch_dtype=load_dtype,
            load_in_4bit=cfg.load_dtype == "4bit",
            load_in_8bit=cfg.load_dtype == "8bit",
            trust_remote_code=True,
        )

        # load tokenizer
        if cfg.tokenizer_path is not None:
            tokenizer_path = cfg.tokenizer_path
        else:
            tokenizer_path = cfg.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        # prepare prompt function
        self.prompt_func = get_prompt_func(model=self.model, tokenizer=self.tokenizer)
        return

    @torch.no_grad()
    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = None,
    ) -> list[list[str]]:
        bsz = len(prefixes)
        sample_num = generation_config.sample_num
        inputs = self.tokenizer(
            prefixes, return_tensors="pt", padding=True, truncation=True
        )
        inputs = inputs.to(self.model.device)

        # prepare generation config
        hf_gen_cfg = self._get_gen_cfg(generation_config)
        if generation_config.eos_token_id is not None:
            inputs["eos_token_id"] = generation_config.eos_token_id
        else:
            inputs["eos_token_id"] = self.tokenizer.eos_token_id

        # generate
        outputs = self.model.generate(
            **inputs,
            generation_config=hf_gen_cfg,
        )

        # truncate the input tokens
        outputs = outputs.view(bsz, sample_num, -1)
        input_lengths = inputs["attention_mask"].sum(dim=1)
        responses = []
        for i in range(bsz):
            samples = [sample[input_lengths[i] :] for sample in outputs[i]]
            samples = [
                self.tokenizer.decode(sample, skip_special_tokens=True)
                for sample in samples
            ]
            responses.append(samples)
        return responses

    def chat(
        self,
        prompts: list[list[dict[str, str]]],
        generation_config: GenerationConfig = None,
    ) -> list[list[str]]:
        prefixes = [self.prompt_func(prompt) for prompt in prompts]
        return self.generate(prefixes, generation_config)

    def _get_gen_cfg(self, generation_config: GenerationConfig) -> HFGenerationConfig:
        return HFGenerationConfig(
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            max_length=generation_config.max_new_tokens,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            num_return_sequences=generation_config.sample_num,
        )


# fmt: off
@dataclass
class HFEncoderConfig(EncoderBaseConfig):
    model_path: str = MISSING
    tokenizer_path: Optional[str] = None
    device_id: list[int] = field(default_factory=list)
    load_dtype: Choices(["bfloat16", "float32", "float16", "8bit", "4bit", "auto"]) = "auto" # type: ignore
    max_encode_length: int = 512
    encode_method: Choices(["cls", "mean"]) = "mean" # type: ignore
# fmt: on


@Encoders("hf", config_class=HFEncoderConfig)
class HFEncoder(EncoderBase):
    def __init__(self, cfg: HFEncoderConfig):
        # prepare gpu
        self.devices = cfg.device_id
        if len(cfg.device_id) < 1:
            device_map = None
        else:
            device_map = cfg.device_id[0]

        # prepare dtype
        match cfg.load_dtype:
            case "bfloat16":
                load_dtype = torch.bfloat16
            case "float32":
                load_dtype = torch.float32
            case "float16":
                load_dtype = torch.float16
            case "8bit":
                load_dtype = None
            case "4bit":
                load_dtype = None
            case "auto":
                load_dtype = "auto"

        # load model
        self.model = AutoModel.from_pretrained(
            cfg.model_path,
            device_map=device_map,
            torch_dtype=load_dtype,
            load_in_4bit=cfg.load_dtype == "4bit",
            load_in_8bit=cfg.load_dtype == "8bit",
            trust_remote_code=True,
        )
        if len(self.devices) > 1:
            self.dp_model = DP(self.model, device_ids=self.devices)
        else:
            self.dp_model = None

        # load tokenizer
        if cfg.tokenizer_path is not None:
            tokenizer_path = cfg.tokenizer_path
        else:
            tokenizer_path = cfg.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        # setup arguments
        self.max_encode_length = cfg.max_encode_length
        self.encode_method = cfg.encode_method
        return

    def get_embedding(
        self, hidden: torch.Tensor, attn_mask: torch.Tensor
    ) -> np.ndarray:
        if self.encode_method == "mean":
            attn_mask = attn_mask.to(hidden.device)
            embeddings = hidden.masked_fill(~attn_mask[..., None].bool(), 0.0)
            embeddings = embeddings.sum(dim=1) / attn_mask.sum(dim=1)[..., None]
            embeddings = embeddings.cpu().numpy()
        elif self.encode_method == "cls":
            embeddings = hidden[:, 0].cpu().numpy()
        else:
            raise ValueError(f"Unsupported encode method: {self.encode_method}")
        return embeddings

    def encode(self, texts: list[str]) -> np.ndarray:
        if (len(texts) >= len(self.devices) * 8) and (self.dp_model is not None):
            encoder = self.dp_model
        else:
            encoder = self.model
        return self._encode(texts, encoder)

    @torch.no_grad()
    def _encode(self, texts: list[str], model: torch.nn.Module | DP) -> np.ndarray:
        input_dict = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=self.max_encode_length,
            padding=True,
            truncation=True,
        )
        if not isinstance(model, DP):
            input_dict = input_dict.to(model.device)
        mask = input_dict["attention_mask"]
        output = model(**input_dict).last_hidden_state
        embeddings = self.get_embedding(output, mask)
        return embeddings

    @property
    def embedding_size(self) -> int:
        return self.model.config.hidden_size
