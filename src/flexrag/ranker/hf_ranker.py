import asyncio
import math
from dataclasses import dataclass

import torch
import numpy as np

from flexrag.models.hf_model import HFModelConfig, load_hf_model, HFGenerationConfig
from flexrag.utils import TIME_METER

from .ranker import RankerBase, RankerBaseConfig, RANKERS


@dataclass
class HFCrossEncoderRankerConfig(RankerBaseConfig, HFModelConfig):
    """The configuration for the HuggingFace Cross Encoder ranker.

    :param max_encode_length: the maximum length for the input encoding. Default is 512.
    :type max_encode_length: int
    """

    max_encode_length: int = 512


@RANKERS("hf_cross_encoder", config_class=HFCrossEncoderRankerConfig)
class HFCrossEncoderRanker(RankerBase):
    """HFCrossEncoderRanker: The ranker based on the HuggingFace Cross Encoder model."""

    def __init__(self, cfg: HFCrossEncoderRankerConfig):
        # load model
        super().__init__(cfg)
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

    @TIME_METER("hf_rank")
    @torch.no_grad()
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
        return None, scores

    async def _async_rank(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        return await asyncio.to_thread(self._rank, query, candidates)


@dataclass
class HFSeq2SeqRankerConfig(RankerBaseConfig, HFModelConfig):
    """The configuration for the HuggingFace Sequence-to-Sequence ranker.

    :param max_encode_length: the maximum length for the input encoding. Default is 512.
    :type max_encode_length: int
    :param input_template: the input template for the seq2seq model.
        Default is "Query: {query} Document: {candidate} Relevant:".
    :type input_template: str
    :param positive_token: the positive token for the seq2seq model. Default is "▁true".
    :type positive_token: str
    :param negative_token: the negative token for the seq2seq model. Default is "▁false".
    :type negative_token: str
    """

    max_encode_length: int = 512
    input_template: str = "Query: {query} Document: {candidate} Relevant:"
    positive_token: str = "▁true"
    negative_token: str = "▁false"


@RANKERS("hf_seq2seq", config_class=HFSeq2SeqRankerConfig)
class HFSeq2SeqRanker(RankerBase):
    """HFSeq2SeqRanker: The ranker based on the HuggingFace Sequence-to-Sequence model."""

    def __init__(self, cfg: HFSeq2SeqRankerConfig):
        # load model
        super().__init__(cfg)
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

    @TIME_METER("hf_rank")
    @torch.no_grad()
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
        return None, scores

    async def _async_rank(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        return await asyncio.to_thread(self._rank, query, candidates)


@dataclass
class HFColBertRankerConfig(RankerBaseConfig, HFModelConfig):
    """The configuration for the HuggingFace ColBERT ranker.

    :param base_model_type: the base model type for the ColBERT model. Default is "bert".
    :type base_model_type: str
    :param output_dim: the output dimension for the ColBERT model. Default is 128.
    :type output_dim: int
    :param max_encode_length: the maximum length for the input encoding. Default is 512.
    :type max_encode_length: int
    :param query_token: the query token for the ColBERT model. Default is "[unused0]".
    :type query_token: str
    :param document_token: the document token for the ColBERT model. Default is "[unused1]".
    :type document_token: str
    :param normalize_embeddings: whether to normalize the embeddings. Default is True.
    :type normalize_embeddings: bool
    """

    base_model_type: str = "bert"
    output_dim: int = 128
    max_encode_length: int = 512
    query_token: str = "[unused0]"
    document_token: str = "[unused1]"
    normalize_embeddings: bool = True


@RANKERS("hf_colbert", config_class=HFColBertRankerConfig)
class HFColBertRanker(RankerBase):
    """HFColBertRanker: The ranker based on the HuggingFace ColBERT model.
    Code adapted from https://github.com/hotchpotch/JQaRA/blob/main/evaluator/reranker/colbert_reranker.py
    """

    def __init__(self, cfg: HFColBertRankerConfig) -> None:
        super().__init__(cfg)
        self.model, self.tokenizer = load_hf_model(
            cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type="colbert",
            device_id=cfg.device_id,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
            colbert_base_model=cfg.base_model_type,
            colbert_dim=cfg.output_dim,
        )
        self.max_encode_length = cfg.max_encode_length
        self.query_token_id = self.tokenizer.convert_tokens_to_ids(cfg.query_token)
        self.document_token_id = self.tokenizer.convert_tokens_to_ids(
            cfg.document_token
        )
        self.normalize = cfg.normalize_embeddings
        return

    @TIME_METER("hf_rank")
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        # tokenize the query & candidates
        query_inputs = self._query_encode([query])
        cand_inputs = self._document_encode(candidates)
        # encode the query & candidates
        query_embeds = self._encode(query_inputs)
        cand_embeds = self._encode(cand_inputs)
        # compute the scores using maxsim(max-cosine)
        token_scores = torch.einsum("qin,pjn->qipj", query_embeds, cand_embeds)
        token_scores = token_scores.masked_fill(
            cand_inputs["attention_mask"].unsqueeze(0).unsqueeze(0) == 0, -1e4
        )
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1) / query_inputs["attention_mask"].sum(-1, keepdim=True)
        scores = scores.cpu().squeeze().float().numpy()
        return None, scores

    async def _async_rank(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        return await asyncio.to_thread(self._rank, query, candidates)

    @torch.no_grad()
    def _tokenize(self, texts: list[str], insert_token_id: int, is_query: bool = False):
        # tokenize the input
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=self.max_encode_length - 1,  # for insert token
            truncation=True,
        )
        inputs = self._insert_token(inputs, insert_token_id)  # type: ignore

        # padding for query
        if is_query:
            mask_token_id = self.tokenizer.mask_token_id

            new_encodings = {"input_ids": [], "attention_mask": []}

            for i, input_ids in enumerate(inputs["input_ids"]):
                original_length = (
                    (input_ids != self.tokenizer.pad_token_id).sum().item()
                )

                # Calculate QLEN dynamically for each query
                if original_length % 16 <= 8:
                    QLEN = original_length + 8
                else:
                    QLEN = math.ceil(original_length / 16) * 16

                if original_length < QLEN:
                    pad_length = QLEN - original_length
                    padded_input_ids = input_ids.tolist() + [mask_token_id] * pad_length
                    padded_attention_mask = (
                        inputs["attention_mask"][i].tolist() + [0] * pad_length
                    )
                else:
                    padded_input_ids = input_ids[:QLEN].tolist()
                    padded_attention_mask = inputs["attention_mask"][i][:QLEN].tolist()

                new_encodings["input_ids"].append(padded_input_ids)
                new_encodings["attention_mask"].append(padded_attention_mask)

            for key in new_encodings:
                new_encodings[key] = torch.tensor(
                    new_encodings[key], device=self.model.device
                )

            inputs = new_encodings

        return {key: value.to(self.model.device) for key, value in inputs.items()}

    @torch.no_grad()
    def _encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # encode
        embs = self.model(**inputs)
        if self.normalize:
            embs = embs / torch.clamp(embs.norm(dim=-1, keepdim=True), 1e-6)
        return embs

    def _insert_token(
        self,
        output: dict,
        insert_token_id: int,
        insert_position: int = 1,
        token_type_id: int = 0,
        attention_value: int = 1,
    ):
        updated_output = {}
        for key in output:
            updated_tensor_list = []
            for seqs in output[key]:
                if len(seqs.shape) == 1:
                    seqs = seqs.unsqueeze(0)
                for seq in seqs:
                    first_part = seq[:insert_position]
                    second_part = seq[insert_position:]
                    new_element = (
                        torch.tensor([insert_token_id])
                        if key == "input_ids"
                        else torch.tensor([token_type_id])
                    )
                    if key == "attention_mask":
                        new_element = torch.tensor([attention_value])
                    updated_seq = torch.cat(
                        (first_part, new_element, second_part), dim=0
                    )
                    updated_tensor_list.append(updated_seq)
            updated_output[key] = torch.stack(updated_tensor_list)
        return updated_output

    def _query_encode(self, query: list[str]):
        return self._tokenize(query, self.query_token_id, is_query=True)

    def _document_encode(self, documents: list[str]):
        return self._tokenize(documents, self.document_token_id)
