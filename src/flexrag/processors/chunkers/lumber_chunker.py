import re
from dataclasses import field

from flexrag.common import LOGGER_MANAGER, ChatMessages, ChatTurn, configure
from flexrag.models.generators import GenerationConfig, GeneratorProtocol

from .basic_chunkers import RecursiveChunker, RecursiveChunkerConfig
from .chunker_base import CHUNKERS, Chunk, ChunkerBase

logger = LOGGER_MANAGER.get_logger("flexrag.processors.chunkers.lumber_chunker")

DEFAULT_SYSTEM_PROMPT = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""


@configure
class LumberChunkerConfig:
    """Configuration for LumberChunker.

    :param system_prompt: The system prompt for the LLM.
    :param window_size: The maximum number of tokens in each group of
        paragraphs sent to the LLM. Default is 550.
    :param min_tail_chunks: The minimum number of paragraphs to keep
        at the end of the document. Default is 5.
    :param pre_chunk_config: The configuration for the pre-chunker
        used to split the text into paragraphs.
    :param use_chat: Whether to use chat-based generation. Default is False.
        This is useful if using chat-based LLMs.
    """

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_chat: bool = False
    window_size: int = 550
    min_tail_chunks: int = 5
    pre_chunk_config: RecursiveChunkerConfig = field(
        default_factory=lambda: RecursiveChunkerConfig(max_tokens=120)
    )


@CHUNKERS("lumber", config_class=LumberChunkerConfig)
class LumberChunker(ChunkerBase):
    """`LumberChunker <https://arxiv.org/abs/2406.17526>`_ is a chunker
    that uses an LLM to identify content shifts between paragraphs.
    It sends groups of paragraphs to the LLM and asks it to find the
    first paragraph where the content clearly changes compared to the
    previous paragraphs.
    """

    def __init__(
        self, cfg: LumberChunkerConfig, generator: GeneratorProtocol
    ) -> None:
        self.generator = generator
        # load pre-chunker
        self.pre_chunker = RecursiveChunker(cfg.pre_chunk_config)
        assert self.pre_chunker.chunk_size < (cfg.window_size // 2), (
            "Pre-chunker chunk size must be less than window_size // 2"
        )
        # other configs
        self.use_chat = cfg.use_chat
        self.system_prompt = cfg.system_prompt
        self.window_size = cfg.window_size
        self.min_tail_chunks = cfg.min_tail_chunks
        self.gen_cfg = GenerationConfig(do_sample=False)
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

        # 3. Slide window to find content shifts
        total_paragraphs = len(id_paragraphs)
        left_idx = 0
        split_poses = []
        while left_idx < total_paragraphs - self.min_tail_chunks:
            # Preare window
            right_idx = left_idx
            while right_idx < total_paragraphs:
                current_window = "\n".join(id_paragraphs[left_idx : right_idx + 1])
                token_count = len(encode_fn(current_window))
                if token_count > self.window_size:
                    break
                right_idx += 1
            if right_idx - left_idx <= 1:
                logger.warning("No sufficient paragraphs to analyze for content shift.")
                left_idx = max(right_idx, left_idx + 1)
                continue

            # Find split position using LLM
            current_window = "\n".join(id_paragraphs[left_idx:right_idx])
            try:
                if self.use_chat:
                    usr_prompt = f"Document:\n{current_window}"
                    prompt = ChatMessages.from_list(
                        [
                            ChatTurn(role="system", content=self.system_prompt),
                            ChatTurn(role="user", content=usr_prompt),
                        ]
                    )
                    response = self.generator.chat([prompt], self.gen_cfg)
                    response = response[0][0].text_content
                else:
                    prompt = f"{self.system_prompt}\n\nDocument:\n{current_window}"
                    response = self.generator.generate([prompt], self.gen_cfg)[0][0]
                match = re.search(r"Answer: ID (\d+)", response)
                assert match is not None
                split_id = int(match.group(1))
                assert left_idx < split_id < right_idx
                split_poses.append(split_id)
                left_idx = split_id
            except Exception as e:
                logger.warning(f"Error during LLM generation: {e}")
                left_idx = max(right_idx, left_idx + 1)
        # Add the last chunk index
        split_poses.append(total_paragraphs)

        # 4. Create chunks
        chunks = []
        start_idx = 0
        for end_idx in split_poses:
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
