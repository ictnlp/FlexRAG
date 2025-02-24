import re
from copy import deepcopy
from dataclasses import dataclass
from string import Template
from typing import Optional

import numpy as np
from omegaconf import MISSING

from flexrag.common_dataclass import RetrievedContext
from flexrag.models import ENCODERS, GENERATORS, EncoderConfig, GeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.utils import TIME_METER

from .refiner import REFINERS, RefinerBase


@dataclass
class AbstractiveSummarizerConfig(GeneratorConfig):
    """The configuration for the ``AbstractiveSummarizer``.

    :param template: The template used to form the input text for the generator. Defaults to None.
        The template should be a Python string.Template object.
        The supported keys for the template are: [content, query].
    :type template: Optional[str]
    :param chat_prompt: The chat prompt for the generator. Defaults to None.
        Only used when the generator is a chat-based generator.
    :type chat_prompt: Optional[ChatPrompt]
    :param substitute: Whether to substitute the original text with the summary. Defaults to True.
        If False, the summary will be stored in a new field named as refined_field + "_summary".
    :type substitute: bool
    :param concatenate_contexts: Whether to concatenate the contexts into one text. Defaults to False.
    :type concatenate_contexts: bool
    :param refined_field: The field to refine. Required.
    :type refined_field: str

    The ``AbstractiveSummarizer`` supports multiple styles of summarizers, including T5, RECOMP, and LLM.
    For example, to summarize the contexts using a `T5 style summarizer <https://arxiv.org/abs/1910.10683)>`_,
    you can run the following code:

    .. code-block:: python

        cfg = AbstractiveSummarizerConfig(
            template="summarize: ${content}",
            generator_type="hf",
            refined_field="text",
            hf_config=HFGeneratorConfig(
                model_path="google-t5/t5-small",
                model_type="seq2seq",
            )
        )
        summarizer = AbstractiveSummarizer(cfg)

    To summarize the contexts using a `RECOMP style summarizer <https://arxiv.org/abs/2010.04348>`_,
    you can run the following code:

    .. code-block:: python

        cfg = AbstractiveSummarizerConfig(
            template="Question: ${query}\\n Document: ${content}\\n Summary: ",
            generator_type="hf",
            refined_field="text",
            hf_config=HFGeneratorConfig(
                model_path="fangyuan/hotpotqa_abstractive_compressor",
                model_type="seq2seq",
            )
        )
        summarizer = AbstractiveSummarizer(cfg)

    To summarize the contexts using a `LLM style summarizer <https://arxiv.org/abs/2203.02155>`_,
    you can run the following code:

    .. code-block:: python

        cfg = AbstractiveSummarizerConfig(
            refined_field="text",
            template="Query: ${query}\\nText: ${content}",
            chat_prompt=ChatPrompt(
                system="You are a skillful summarizer. Please summarize the following text based on given query.",
            ),
            generator_type="openai",
            openai_config=OpenAIGeneratorConfig(api_key=api_key, model_name="gpt-3.5-turbo")
        )
        summarizer = AbstractiveSummarizer(cfg)
    """

    template: Optional[str] = None
    chat_prompt: Optional[ChatPrompt] = None
    substitute: bool = True
    concatenate_contexts: bool = False
    refined_field: str = MISSING


@REFINERS("abstractive_summarizer", config_class=AbstractiveSummarizerConfig)
class AbstractiveSummarizer(RefinerBase):
    """The ``AbstractiveSummarizer`` summarizes the contexts using a generator."""

    def __init__(self, cfg: AbstractiveSummarizerConfig):
        super().__init__(cfg)
        self.model = GENERATORS.load(cfg)
        if cfg.template is not None:
            self.template = Template(cfg.template)
        else:
            self.template = None
        self.chat_prompt = cfg.chat_prompt
        self.substitute = cfg.substitute
        self.concatenate = cfg.concatenate_contexts
        self.refined_field = cfg.refined_field
        return

    @TIME_METER("abstractive_summarize")
    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        # prepare input texts
        if self.concatenate:
            assert all(
                contexts[0].query == context.query for context in contexts
            ), "All queries should be the same."
            args = [
                {
                    "content": " ".join(
                        [context.data[self.refined_field] for context in contexts]
                    ),
                    "query": contexts[0].query,
                }
            ]
        else:
            args = [
                {
                    "content": context.data[self.refined_field],
                    "query": context.query,
                }
                for context in contexts
            ]
        if self.template is not None:
            input_texts = [self.template.safe_substitute(**arg) for arg in args]
        else:
            input_texts = [arg["content"] for arg in args]

        # generate summaries
        if self.chat_prompt is not None:
            input_prompts = []
            for text in input_texts:
                prompt = deepcopy(self.chat_prompt)
                prompt.update(ChatTurn(role="user", content=text))
                input_prompts.append(prompt)
            summaries = [i[0] for i in self.model.chat(input_prompts)]
        else:
            summaries = [i[0] for i in self.model.generate(input_texts)]

        # update contexts
        new_contexts = []
        for context, summary in zip(contexts, summaries):
            context = deepcopy(context)
            if self.substitute:
                context.data[self.refined_field] = summary
            else:
                if self.concatenate:
                    context.data[self.refined_field] = args[0]["content"]
                context.data[self.refined_field + "_summary"] = summary
            new_contexts.append(context)
        return new_contexts


@dataclass
class RecompExtractiveSummarizerConfig(EncoderConfig):
    """The configuration for the ``RecompExtractiveSummarizer``.

    :param preserved_sents: The number of sentences to preserve. Defaults to 5.
    :type preserved_sents: int
    :param concatenate_contexts: Whether to concatenate the contexts into one text. Defaults to False.
    :type concatenate_contexts: bool
    :param substitute: Whether to substitute the original text with the summary. Defaults to False.
    :type substitute: bool
    :param refined_field: The field to refine. Required.
    :type refined_field: str

    The ``RecompExtractiveSummarizer`` is motivated by the RECOMP (https://arxiv.org/abs/2310.04408).
    For example, to load a summarizer trained on hotpotqa dataset, you can run the following code:

    .. code-block:: python

        cfg = RecompExtractiveSummarizerConfig(
            encoder_type="hf",
            hf_config=HFEncoderConfig(
                model_path="fangyuan/hotpotqa_extractive_compressor",
            ),
            preserved_sents=5,
            refined_field="text",
        )
        summarizer = RecompExtractiveSummarizer(cfg)
    """

    preserved_sents: int = 5
    concatenate_contexts: bool = False
    substitute: bool = False
    refined_field: str = MISSING


@REFINERS("extractive_summarizer", config_class=RecompExtractiveSummarizerConfig)
class RecompExtractiveSummarizer(RefinerBase):
    """The ``ExtractiveSummarizer`` summarizes the contexts using an encoder."""

    def __init__(self, cfg: RecompExtractiveSummarizerConfig) -> None:
        self.model = ENCODERS.load(cfg)
        assert self.model is not None, "The encoder model is not provided."
        self.concatenate = cfg.concatenate_contexts
        self.top_k = cfg.preserved_sents
        self.substitute = cfg.substitute
        self.refined_field = cfg.refined_field
        return

    @TIME_METER("extractive_summarize")
    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        if self.concatenate:
            assert all(
                contexts[0].query == context.query for context in contexts
            ), "All queries should be the same."
            contents = [
                " ".join([context.data[self.refined_field] for context in contexts])
            ]
            queries = [contexts[0].query]
        else:
            contents = [context.data[self.refined_field] for context in contexts]
            queries = [context.query for context in contexts]

        # cut the texts into sentences
        sent_lists = [
            [i.strip() for i in re.split(r"(?<=[.!?])\s+", t) if len(i.strip()) > 5]
            for t in contents
        ]

        # encode the sentences & query
        flat_sents = sum(sent_lists, [])
        query_emb = self.model.encode(queries)
        sents_emb = self.model.encode(flat_sents)
        all_scores = query_emb @ sents_emb.T

        # select topk sents
        new_ctxs = []
        for n, (sents, scores) in enumerate(zip(sent_lists, all_scores)):
            scores = scores[
                sum([len(i) for i in sent_lists[:n]]) : sum(
                    [len(i) for i in sent_lists[:n]]
                )
                + len(sent_lists[n])
            ]
            if len(scores) < self.top_k:
                indices = np.arange(len(scores))
            else:
                indices = np.argpartition(-scores, self.top_k)[: self.top_k]
            new_ctx = deepcopy(contexts[n])
            if self.substitute:
                new_ctx.data[self.refined_field] = " ".join(
                    [sents[i] for i in sorted(indices)]
                )
            else:
                new_ctx.data[self.refined_field + "_summary"] = " ".join(
                    [sents[i] for i in sorted(indices)]
                )
                if self.concatenate:
                    new_ctx.data[self.refined_field] = contents[0]
            new_ctxs.append(new_ctx)
        return new_ctxs
