import re
from copy import deepcopy
from string import Template
from typing import Any, Optional

import numpy as np

from flexrag.common import configure, trace
from flexrag.common.dataclasses import ChatMessages, ChatTurn, RetrievedContext
from flexrag.models.encoders import EncoderProtocol
from flexrag.models.generators import GeneratorProtocol

from .refiner_base import REFINERS


@configure
class AbstractiveSummarizerConfig:
    """The configuration for the ``AbstractiveSummarizer``.

    :param template: The template used to form the input text for the generator. Defaults to None.
        The template should be a Python string.Template object.
        The supported keys for the template are: [content, query].
    :param chat_prompt: The chat prompt for the generator. Defaults to None.
        Only used when the generator is a chat-based generator.
    :param substitute: Whether to substitute the original text with the summary. Defaults to True.
        If False, the summary will be stored in a new field named as refined_field + "_summary".
    :param concatenate_contexts: Whether to concatenate the contexts into one text. Defaults to False.
    :param refined_field: The field to refine. Required.

    The ``AbstractiveSummarizer`` supports multiple styles of summarizers, including T5, RECOMP, and LLM.
    For example, to summarize the contexts using a `T5 style summarizer <https://arxiv.org/abs/1910.10683)>`_,
    you can run the following code:

    .. code-block:: python

        cfg = AbstractiveSummarizerConfig(
            template="summarize: ${content}",
            refined_field="text",
        )
        summarizer = AbstractiveSummarizer(cfg, generator=generator)

    To summarize the contexts using a `RECOMP style summarizer <https://arxiv.org/abs/2010.04348>`_,
    you can run the following code:

    .. code-block:: python

        cfg = AbstractiveSummarizerConfig(
            template="Question: ${query}\\n Document: ${content}\\n Summary: ",
            refined_field="text",
        )
        summarizer = AbstractiveSummarizer(cfg, generator=generator)

    To summarize the contexts using a `LLM style summarizer <https://arxiv.org/abs/2203.02155>`_,
    you can run the following code:

    .. code-block:: python

        cfg = AbstractiveSummarizerConfig(
            refined_field="text",
            template="Query: ${query}\\nText: ${content}",
            chat_prompt=ChatMessages(
                system="You are a skillful summarizer. Please summarize the following text based on given query.",
            ),
        )
        summarizer = AbstractiveSummarizer(cfg, generator=generator)
    """

    template: Optional[str] = None
    chat_prompt: Optional[ChatMessages] = None
    substitute: bool = True
    concatenate_contexts: bool = False
    refined_field: Optional[str] = None


@REFINERS("abstractive_summarizer", config_class=AbstractiveSummarizerConfig)
class AbstractiveSummarizer:
    """The ``AbstractiveSummarizer`` summarizes the contexts using a generator."""

    def __init__(self, cfg: AbstractiveSummarizerConfig, generator: GeneratorProtocol):
        self.model = generator
        if cfg.template is not None:
            self.template = Template(cfg.template)
        else:
            self.template = None
        self.chat_prompt = cfg.chat_prompt
        self.substitute = cfg.substitute
        self.concatenate = cfg.concatenate_contexts
        assert cfg.refined_field is not None, "The refined_field must be provided."
        self.refined_field = cfg.refined_field
        return

    @trace("refiner.abstractive_summarize")
    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Summarize retrieved contexts with the injected generator.

        :param contexts: Retrieved contexts to summarize.
        :return: Summarized contexts.
        """
        args, input_texts = self._prepare_inputs(contexts)
        if self.chat_prompt is not None:
            summaries = [
                result[0] for result in self.model.chat(self._make_prompts(input_texts))
            ]
        else:
            summaries = [result[0] for result in self.model.generate(input_texts)]
        return self._update_contexts(contexts, args, summaries)

    @trace("refiner.abstractive_summarize")
    async def async_refine(
        self, contexts: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        """Summarize retrieved contexts asynchronously.

        :param contexts: Retrieved contexts to summarize.
        :return: Summarized contexts.
        """
        args, input_texts = self._prepare_inputs(contexts)
        if self.chat_prompt is not None:
            summaries = [
                result[0]
                for result in await self.model.async_chat(
                    self._make_prompts(input_texts)
                )
            ]
        else:
            summaries = [
                result[0] for result in await self.model.async_generate(input_texts)
            ]
        return self._update_contexts(contexts, args, summaries)

    def _prepare_inputs(
        self, contexts: list[RetrievedContext]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Prepare template arguments and generator input texts."""
        if self.concatenate:
            assert all(contexts[0].query == context.query for context in contexts), (
                "All queries should be the same."
            )
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
        return args, input_texts

    def _make_prompts(self, input_texts: list[str]) -> list[ChatMessages]:
        """Build independent chat prompts for prepared input texts."""
        assert self.chat_prompt is not None
        input_prompts = []
        for text in input_texts:
            prompt = deepcopy(self.chat_prompt)
            prompt.append(ChatTurn(role="user", content=text))
            input_prompts.append(prompt)
        return input_prompts

    def _update_contexts(
        self,
        contexts: list[RetrievedContext],
        args: list[dict[str, Any]],
        summaries: list[Any],
    ) -> list[RetrievedContext]:
        """Copy contexts and write generated summaries to their data."""
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


@configure
class RecompExtractiveSummarizerConfig:
    """The configuration for the ``RecompExtractiveSummarizer``.

    :param preserved_sents: The number of sentences to preserve. Defaults to 5.
    :param concatenate_contexts: Whether to concatenate the contexts into one text. Defaults to False.
    :param substitute: Whether to substitute the original text with the summary. Defaults to False.
    :param refined_field: The field to refine. Required.

    The ``RecompExtractiveSummarizer`` is motivated by the RECOMP (https://arxiv.org/abs/2310.04408).
    For example, to load a summarizer trained on hotpotqa dataset, you can run the following code:

    .. code-block:: python

        cfg = RecompExtractiveSummarizerConfig(
            preserved_sents=5,
            refined_field="text",
        )
        summarizer = RecompExtractiveSummarizer(cfg, encoder=encoder)
    """

    preserved_sents: int = 5
    concatenate_contexts: bool = False
    substitute: bool = False
    refined_field: Optional[str] = None


@REFINERS("extractive_summarizer", config_class=RecompExtractiveSummarizerConfig)
class RecompExtractiveSummarizer:
    """The ``ExtractiveSummarizer`` summarizes the contexts using an encoder."""

    def __init__(
        self, cfg: RecompExtractiveSummarizerConfig, encoder: EncoderProtocol
    ) -> None:
        self.model = encoder
        self.concatenate = cfg.concatenate_contexts
        self.top_k = cfg.preserved_sents
        self.substitute = cfg.substitute
        assert cfg.refined_field is not None, "The refined_field must be provided."
        self.refined_field = cfg.refined_field
        return

    @trace("refiner.extractive_summarize")
    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Extract the most relevant sentences from retrieved contexts.

        :param contexts: Retrieved contexts to summarize.
        :return: Extractively summarized contexts.
        """
        contents, queries, sent_lists, flat_sents = self._prepare_inputs(contexts)
        query_emb = self.model.encode(queries)
        sents_emb = self.model.encode(flat_sents)
        return self._update_contexts(
            contexts, contents, sent_lists, query_emb @ sents_emb.T
        )

    @trace("refiner.extractive_summarize")
    async def async_refine(
        self, contexts: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        """Extract the most relevant sentences asynchronously.

        :param contexts: Retrieved contexts to summarize.
        :return: Extractively summarized contexts.
        """
        contents, queries, sent_lists, flat_sents = self._prepare_inputs(contexts)
        query_emb = await self.model.async_encode(queries)
        sents_emb = await self.model.async_encode(flat_sents)
        return self._update_contexts(
            contexts, contents, sent_lists, query_emb @ sents_emb.T
        )

    def _prepare_inputs(
        self, contexts: list[RetrievedContext]
    ) -> tuple[list[str], list[str | None], list[list[str]], list[str]]:
        """Prepare context texts, queries, and sentence batches for encoding."""
        if self.concatenate:
            assert all(contexts[0].query == context.query for context in contexts), (
                "All queries should be the same."
            )
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
        flat_sents = sum(sent_lists, [])
        return contents, queries, sent_lists, flat_sents

    def _update_contexts(
        self,
        contexts: list[RetrievedContext],
        contents: list[str],
        sent_lists: list[list[str]],
        all_scores: np.ndarray,
    ) -> list[RetrievedContext]:
        """Copy contexts and write the selected sentences to their data."""
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
