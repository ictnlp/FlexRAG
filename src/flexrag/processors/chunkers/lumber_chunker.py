import re
from dataclasses import field

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.models import GENERATORS, GeneratorConfig

from .basic_chunkers import RecursiveChunker, RecursiveChunkerConfig
from .chunker_base import CHUNKERS, Chunk, ChunkerBase

logger = LOGGER_MANAGER.get_logger("flexrag.processors.chunkers.lumber_chunker")

DEFAULT_SYSTEM_PROMPT = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""


DEFAULT_PRE_CHUNKER_CONFIG = RecursiveChunkerConfig(chunk_size=120)


@configure
class LumberChunkerConfig(GeneratorConfig):
    """Configuration for LumberChunker.

    :param system_prompt: The system prompt for the LLM.
    :type system_prompt: str
    :param max_tokens: The maximum number of tokens in each group of
        paragraphs sent to the LLM. Default is 550.
    :type max_tokens: int
    :param min_paragraphs: The minimum number of paragraphs to keep
        at the end of the document. Default is 5.
    :type min_paragraphs: int
    :param pre_chunk_config: The configuration for the pre-chunker
        used to split the text into paragraphs.
    :type pre_chunk_config: RecursiveChunkerConfig
    """

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 550
    min_paragraphs: int = 5
    pre_chunk_config: RecursiveChunkerConfig = field(
        default_factory=lambda: DEFAULT_PRE_CHUNKER_CONFIG
    )


@CHUNKERS("lumber", config_class=LumberChunkerConfig)
class LumberChunker(ChunkerBase):
    """`LumberChunker <https://arxiv.org/abs/2406.17526>`_ is a chunker
    that uses an LLM to identify content shifts between paragraphs.
    It sends groups of paragraphs to the LLM and asks it to find the
    first paragraph where the content clearly changes compared to the
    previous paragraphs.
    """

    def __init__(self, cfg: LumberChunkerConfig) -> None:
        # load generator
        self.generator = GENERATORS.load(cfg)
        # load pre-chunker
        self.pre_chunker = RecursiveChunker(cfg.pre_chunk_config)
        assert (
            self.pre_chunker.chunk_size < cfg.max_tokens
        ), "Pre-chunker chunk size must be less than max_tokens"
        # other configs
        self.system_prompt = cfg.system_prompt
        self.max_tokens = cfg.max_tokens
        self.min_paragraphs = cfg.min_paragraphs
        return

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk] | list[str]:
        if self.pre_chunker.tokenizer.vocab_size > 0:
            encode_fn = self.pre_chunker.tokenizer.encode
        else:
            encode_fn = self.pre_chunker.tokenizer.tokenize
        # 1. Split text into paragraphs using RecursiveChunker
        paragraphs = self.pre_chunker.chunk(text, return_str=True)
        if not paragraphs:
            return []

        # 2. Add IDs to paragraphs
        id_paragraphs = [f"ID {i}: {p}" for i, p in enumerate(paragraphs)]

        new_id_list = []
        chunk_number = 0
        total_paragraphs = len(id_paragraphs)

        while chunk_number < total_paragraphs - self.min_paragraphs:
            token_count = 0
            i = 0
            while (
                token_count < self.max_tokens
                and i + chunk_number < total_paragraphs - 1
            ):
                i += 1
                final_document = "\n".join(
                    id_paragraphs[chunk_number : i + chunk_number]
                )
                token_count = len(encode_fn(final_document))

            if i == 1:
                final_document = id_paragraphs[chunk_number]
            else:
                final_document = "\n".join(
                    id_paragraphs[chunk_number : i - 1 + chunk_number]
                )

            prompt = f"{self.system_prompt}\n\nDocument:\n{final_document}"

            # Default move forward
            chunk_number = chunk_number + i - 1

            try:
                response = self.generator.generate(prompt)[0][0]
                match = re.search(r"Answer: ID (\d+)", response)
                if match:
                    llm_id = int(match.group(1))
                    # Ensure llm_id is valid and progressing
                    if llm_id < total_paragraphs:
                        chunk_number = llm_id
                        new_id_list.append(chunk_number)
                        chunk_number += 1
                else:
                    logger.warning(f"No ID found in LLM response: {response}")
            except Exception as e:
                logger.error(f"Error during LLM generation: {e}")
                chunk_number += 1

        # Add the last chunk index
        new_id_list.append(total_paragraphs)

        # Create chunks
        chunks = []
        start_idx = 0
        for end_idx in new_id_list:
            if end_idx > start_idx:
                chunk_text = "\n\n".join(paragraphs[start_idx:end_idx])
                chunks.append(Chunk(text=chunk_text))
                start_idx = end_idx

        if start_idx < total_paragraphs:
            chunk_text = "\n\n".join(paragraphs[start_idx:])
            chunks.append(Chunk(text=chunk_text))

        if return_str:
            return [c.text for c in chunks]
        return chunks
