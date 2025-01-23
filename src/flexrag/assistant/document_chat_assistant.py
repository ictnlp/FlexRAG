from dataclasses import dataclass
from typing import Any

from flexrag.common_dataclass import RetrievedContext
from flexrag.chunking import CHUNKERS, ChunkerConfig
from flexrag.document_parser import DOCUMENTPARSERS, DocumentParserConfig
from flexrag.models import GENERATORS, GenerationConfig, GeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.ranker import RANKERS, RankerConfig
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig
from flexrag.utils import LOGGER_MANAGER

from .assistant import ASSISTANTS, AssistantBase

logger = LOGGER_MANAGER.get_logger("flexrag.assistant.modular")


@dataclass
class DocumentChatAssistantConfig(
    GeneratorConfig,
    GenerationConfig,
    DenseRetrieverConfig,
    RankerConfig,
    DocumentParserConfig,
    ChunkerConfig,
): ...


@ASSISTANTS("document_chat", config_class=DocumentChatAssistantConfig)
class DocumentChatAssistant(AssistantBase):
    def __init__(self, cfg: DocumentChatAssistantConfig):
        # set basic args
        self.gen_cfg = cfg
        if self.gen_cfg.sample_num > 1:
            logger.warning("Sample num > 1 is not supported for Assistant")
            self.gen_cfg.sample_num = 1

        # load generator
        self.generator = GENERATORS.load(cfg)

        # load retriever
        self.retriever = DenseRetriever(cfg)
        assert len(self.retriever) == 0, "Retriever is not empty."

        # load ranker
        self.reranker = RANKERS.load(cfg)

        # load parser
        self.parser = DOCUMENTPARSERS.load(cfg)

        # load chunker
        self.chunker = CHUNKERS.load(cfg)
        return

    def attach_document(self, document_path: str = None) -> None:
        if document_path is None:
            self.retriever.clean()
            return
        # parse document
        self.retriever.clean()
        document = self.parser.parse(document_path)
        if self.chunker is not None:
            chunks = self.chunker.chunk(document.text)
        else:
            chunks = [document.text]

        # build index
        self.retriever.add_passages(chunks)
        return

    def answer(
        self, question: str
    ) -> tuple[str, list[RetrievedContext], dict[str, Any]]:
        # answer without contexts
        if len(self.retriever) == 0:
            prompt = ChatPrompt()
            prompt.update(ChatTurn(role="user", content=question))
            response = self.generator.chat([prompt], generation_config=self.gen_cfg)
            return response[0][0], [], {"prompt": prompt}

        # retrieve
        retrieved_contexts = self.retriever.search(question)[0]

        # rerank
        if self.reranker is not None:
            contexts = self.reranker.rank(question, retrieved_contexts).candidates
        else:
            contexts = retrieved_contexts

        # prepare prompt
        prompt = ChatPrompt(
            system="Answer the user question based on the given contexts."
        )
        usr_prompt = ""
        for n, context in enumerate(contexts):
            ctx = ""
            for field_name, field_value in context.data.items():
                ctx += f"{field_name}: {field_value}\n"
            usr_prompt += f"Context {n + 1}: {ctx}\n\n"
        usr_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=usr_prompt))

        # generate response
        response = self.generator.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return response, contexts, {"prompt": prompt}
