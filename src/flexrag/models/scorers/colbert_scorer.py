import math
from collections import defaultdict

import numpy as np
import torch

from flexrag.common import configure, trace

from ..hf_utils import HFModelConfig, load_hf_model
from .scorer_base import LocalPairScorerBase


@configure
class HFColBertScorerConfig(HFModelConfig):
    """The configuration for the HuggingFace ColBERT scorer.

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
    :param batch_size: Maximum direct-use batch size. This value is ignored
        when the scorer is created through Runtime or ResourceManager;
        configure the runtime or resource batch size instead.
    """

    base_model_type: str = "bert"
    output_dim: int = 128
    max_encode_length: int = 512
    query_token: str = "[unused0]"
    document_token: str = "[unused1]"
    normalize_embeddings: bool = True
    batch_size: int = 32


class HFColBertScorer(LocalPairScorerBase):
    """HFColBertScorer: The scorer based on the HuggingFace ColBERT model.
    Code adapted from https://github.com/hotchpotch/JQaRA/blob/main/evaluator/reranker/colbert_reranker.py
    """

    def __init__(self, cfg: HFColBertScorerConfig) -> None:
        super().__init__(batch_size=cfg.batch_size)

        self.model, self.tokenizer = load_hf_model(
            cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type="colbert",
            device_map=cfg.device_map,
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

    @trace("scorer.hf_colbert")
    def _score_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.float32)

        # ColBERT scores pairs, but repeated queries are common in reranking.
        # Deduplicating queries lets us encode each query only once per batch.
        query_to_idx: dict[str, int] = {}
        query_indices: list[int] = []
        unique_queries: list[str] = []
        for query, _ in pairs:
            idx = query_to_idx.get(query)
            if idx is None:
                idx = len(unique_queries)
                query_to_idx[query] = idx
                unique_queries.append(query)
            query_indices.append(idx)

        candidate_texts = [candidate for _, candidate in pairs]
        query_inputs = self._query_encode(unique_queries)
        cand_inputs = self._document_encode(candidate_texts)
        query_embeds = self._encode(query_inputs)
        cand_embeds = self._encode(cand_inputs)

        grouped_pair_indices: dict[int, list[int]] = defaultdict(list)
        for pair_idx, query_idx in enumerate(query_indices):
            grouped_pair_indices[query_idx].append(pair_idx)

        scores = np.empty(len(pairs), dtype=np.float32)
        for query_idx, pair_indices in grouped_pair_indices.items():
            # MaxSim still works most naturally per query, so we reuse the
            # shared query embedding against all candidates that belong to it.
            group_query_embeds = query_embeds[query_idx : query_idx + 1]
            group_query_mask = query_inputs["attention_mask"][query_idx : query_idx + 1]
            group_cand_embeds = cand_embeds[pair_indices]
            group_cand_mask = cand_inputs["attention_mask"][pair_indices]

            token_scores = torch.einsum(
                "qin,pjn->qipj",
                group_query_embeds,
                group_cand_embeds,
            )
            token_scores = token_scores.masked_fill(
                group_cand_mask.unsqueeze(0).unsqueeze(0) == 0,
                -1e4,
            )
            group_scores, _ = token_scores.max(-1)
            group_scores = group_scores.sum(1) / group_query_mask.sum(-1, keepdim=True)
            group_scores = np.atleast_1d(group_scores.squeeze(0).cpu().float().numpy())
            # Scatter grouped scores back to the original pair order expected
            # by the caller.
            scores[np.array(pair_indices)] = group_scores.astype(np.float32, copy=False)
        return scores

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
        inputs = self._insert_token(inputs, insert_token_id)

        # padding for query
        if is_query:
            mask_token_id = self.tokenizer.mask_token_id

            new_encodings = {"input_ids": [], "attention_mask": []}
            buffered_encodings: list[tuple[list[int], list[int]]] = []
            max_qlen = 0

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

                buffered_encodings.append((padded_input_ids, padded_attention_mask))
                max_qlen = max(max_qlen, len(padded_input_ids))

            for padded_input_ids, padded_attention_mask in buffered_encodings:
                pad_length = max_qlen - len(padded_input_ids)
                if pad_length > 0:
                    padded_input_ids = padded_input_ids + [mask_token_id] * pad_length
                    padded_attention_mask = padded_attention_mask + [0] * pad_length
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
