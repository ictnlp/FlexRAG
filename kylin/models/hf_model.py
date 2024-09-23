import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from omegaconf import MISSING
from torch.nn.parallel import DataParallel as DP
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers import GenerationConfig as HFGenerationConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from kylin.prompt import ChatPrompt, load_template
from kylin.utils import Choices, TimeMeter

from .model_base import (
    EncoderBase,
    EncoderBaseConfig,
    Encoders,
    GenerationConfig,
    GeneratorBase,
    GeneratorBaseConfig,
    Generators,
    RankerBase,
    RankerConfig,
    RankingResult,
)
from .utils import guess_model_name

logger = logging.getLogger(__name__)


def load_hf_model(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    model_type: Optional[str] = None,
    device_id: list[int] = [],
    load_dtype: str = "auto",
    trust_remote_code: bool = False,
    pipeline_parallel: bool = False,
    is_training: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    # prepare dtype
    load_in_4bit = False
    load_in_8bit = False
    match load_dtype:
        case "bfloat16":
            load_dtype = torch.bfloat16
        case "bf16":
            load_dtype = torch.bfloat16
        case "float32":
            load_dtype = torch.float32
        case "fp32":
            load_dtype = torch.float32
        case "float16":
            load_dtype = torch.float16
        case "fp16":
            load_dtype = torch.float16
        case "half":
            load_dtype = torch.float16
        case "8bit":
            load_dtype = None
            load_in_8bit = True
        case "4bit":
            load_dtype = None
            load_in_4bit = True
        case "auto":
            load_dtype = "auto"
        case _:
            raise ValueError(f"Unsupported load_dtype: {load_dtype}")

    # prepare device
    if pipeline_parallel:
        device_map = "auto"
    elif torch.cuda.is_available() and (len(device_id) > 0):
        device_map = device_id[0]
    else:
        device_map = None

    # load model
    match model_type:
        case "causal_lm":
            model_class = AutoModelForCausalLM
        case "seq2seq":
            model_class = AutoModelForSeq2SeqLM
        case "sequence_classification":
            model_class = AutoModelForSequenceClassification
        case _:
            model_class = AutoModel
    model = model_class.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=load_dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        trust_remote_code=trust_remote_code,
    )
    if not is_training:
        model.eval()

    # load tokenizer
    if tokenizer_path is not None:
        tokenizer_path = tokenizer_path
    else:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer


@dataclass
class HFModelConfig:
    model_path: str = MISSING
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    device_id: list[int] = field(default_factory=list)
    load_dtype: Choices(  # type: ignore
        [
            "bfloat16",
            "bf16",
            "float32",
            "fp32",
            "float16",
            "fp16",
            "half",
            "8bit",
            "4bit",
            "auto",
        ]
    ) = "auto"


@dataclass
class HFGeneratorConfig(GeneratorBaseConfig, HFModelConfig):
    pipeline_parallel: bool = False
    use_minference: bool = False


@Generators("hf", config_class=HFGeneratorConfig)
class HFGenerator(GeneratorBase):
    model: PreTrainedModel

    def __init__(self, cfg: HFGeneratorConfig) -> None:
        # load model
        self.model, self.tokenizer = load_hf_model(
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type="causal_lm",
            device_id=cfg.device_id,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
            pipeline_parallel=cfg.pipeline_parallel,
        )
        self._patch_model()

        # prepare prompt function
        model_name = guess_model_name(self.model.config)
        self.template = load_template(model_name=model_name, tokenizer=self.tokenizer)

        # load minference
        if cfg.use_minference:
            assert (
                not cfg.pipeline_parallel
            ), "Minference does not support pipeline parallel"
            from minference import MInference

            try:
                inf_patch = MInference("minference", model_name)
                self.model = inf_patch(self.model)
            except Exception as e:
                logger.warning(f"Unable to load minference: {e}")
        return

    @TimeMeter("hf_generate")
    @torch.no_grad()
    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        bsz = len(prefixes)
        sample_num = generation_config.sample_num
        inputs = self.tokenizer(
            prefixes, return_tensors="pt", padding=True, truncation=True
        )
        inputs = inputs.to(self.model.device)

        # prepare generation config
        hf_gen_cfg = self._get_options(generation_config)
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
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        assert self.template is not None, "Chat function is disabled."
        prefixes = [self.template.render_to_text(prompt) for prompt in prompts]
        return self.generate(prefixes, generation_config)

    def _get_options(self, generation_config: GenerationConfig) -> HFGenerationConfig:
        return HFGenerationConfig(
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            max_new_tokens=generation_config.max_new_tokens,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            num_return_sequences=generation_config.sample_num,
        )

    def _patch_model(self) -> None:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        return


@dataclass
class HFEncoderConfig(EncoderBaseConfig, HFModelConfig):
    max_encode_length: int = 512
    encode_method: Choices(["cls", "mean"]) = "mean"  # type: ignore


@Encoders("hf", config_class=HFEncoderConfig)
class HFEncoder(EncoderBase):
    def __init__(self, cfg: HFEncoderConfig):
        self.devices = cfg.device_id
        # load model
        self.model, self.tokenizer = load_hf_model(
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            load_dtype=cfg.load_dtype,
            device_id=cfg.device_id,
            trust_remote_code=cfg.trust_remote_code,
        )
        if len(self.devices) > 1:
            self.dp_model = DP(self.model, device_ids=self.devices)
        else:
            self.dp_model = None

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


@dataclass
class HFCrossEncoderRankerConfig(RankerConfig, HFModelConfig):
    max_encode_length: int = 512


class HFCrossEncoderRanker(RankerBase):
    def __init__(self, cfg: HFCrossEncoderRankerConfig):
        # load model
        self.model, self.tokenizer = load_hf_model(
            cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type="sequence_classification",
            device_id=cfg.device_id,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.max_encode_length = cfg.max_encode_length
        return

    @TimeMeter("hf_rank")
    @torch.no_grad()
    def rank(self, query: str, candidates: list[str]) -> RankingResult:
        # score the candidates
        input_texts = [(query, cand) for cand in candidates]
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=self.max_encode_length,
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(self.model.device)
        scores = self.model(**inputs).logits.squeeze().cpu().numpy()
        # rank the candidates
        rank_indices = np.argsort(-scores)
        return RankingResult(
            query=query,
            candidates=candidates,
            scores=list(scores),
            ranking=list(rank_indices),
        )


@dataclass
class HFSeq2SeqRankerConfig(RankerConfig, HFModelConfig):
    max_encode_length: int = 512
    input_template: str = "Query: {query} Document: {candidate} Relevant:"
    positive_token: str = "▁true"
    negative_token: str = "▁false"


class HFSeq2SeqRanker(RankerBase):
    def __init__(self, cfg: HFSeq2SeqRankerConfig):
        # load model
        self.model, self.tokenizer = load_hf_model(
            cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type="seq2seq",
            device_id=cfg.device_id,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.max_encode_length = cfg.max_encode_length
        self.input_template = cfg.input_template
        self.positive_token = self.tokenizer.convert_tokens_to_ids(cfg.positive_token)
        self.negative_token = self.tokenizer.convert_tokens_to_ids(cfg.negative_token)
        self.generation_config = HFGenerationConfig(
            max_new_tokens=1, output_logits=True
        )
        return

    @TimeMeter("hf_rank")
    @torch.no_grad()
    def rank(self, query: str, candidates: list[str]) -> RankingResult:
        # prepare prompts
        input_texts = [
            self.input_template.format(query=query, candidate=cand)
            for cand in candidates
        ]
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=self.max_encode_length,
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
        )
        logits = outputs.logits[0]
        positive_scores = logits[:, self.positive_token : self.positive_token + 1]
        negative_scores = logits[:, self.negative_token : self.negative_token + 1]
        scores = torch.softmax(
            torch.cat([positive_scores, negative_scores], dim=1), dim=1
        )[:, 0].cpu().numpy()  # fmt: skip
        # rank the candidates
        rank_indices = np.argsort(-scores)
        return RankingResult(
            query=query,
            candidates=candidates,
            scores=list(scores),
            ranking=list(rank_indices),
        )
