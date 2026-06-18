import json
from dataclasses import field

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.models.generators import GenerationConfig, GeneratorProtocol

from .basic_chunkers import RecursiveChunker, RecursiveChunkerConfig
from .chunker_base import CHUNKERS, Chunk, ChunkerBase

logger = LOGGER_MANAGER.get_logger("flexrag.processors.chunkers.densex_chunker")


@configure
class DenseXChunkerConfig:
    """Configuration for DenseXChunker.

    :param pre_chunk_config: The configuration for the pre-chunker
        used to split the text into paragraphs.
    """

    pre_chunk_config: RecursiveChunkerConfig = field(
        default_factory=lambda: RecursiveChunkerConfig(max_tokens=120)
    )


@CHUNKERS("densex", config_class=DenseXChunkerConfig)
class DenseXChunker(ChunkerBase):
    """`DenseXChunker <https://arxiv.org/abs/2312.06648>`_ uses a propositionizer
    model to split text into propositions.

    The generator must be compatible with the DenseX propositionizer prompt and
    JSON-list output format. The recommended model is the Hugging Face seq2seq
    model ``chentong00/propositionizer-wiki-flan-t5-large`` deployed through a
    raw ``HFGenerator`` or a generator runtime/resource. Arbitrary generators
    are not guaranteed to produce parseable DenseX propositions.

    Example:

    .. code-block:: python

        from flexrag.models import HFGenerator, HFGeneratorConfig
        from flexrag.processors.chunkers import DenseXChunker, DenseXChunkerConfig

        generator = HFGenerator(
            HFGeneratorConfig(
                model_path="chentong00/propositionizer-wiki-flan-t5-large",
                model_type="seq2seq",
                device_id=[0],
            )
        )
        chunker = DenseXChunker(DenseXChunkerConfig(), generator=generator)
        propositions = chunker.chunk("DenseX turns paragraphs into propositions.")
    """

    def __init__(
        self,
        cfg: DenseXChunkerConfig,
        generator: GeneratorProtocol,
    ) -> None:
        self.generator = generator
        self.gen_cfg = GenerationConfig(max_new_tokens=512, do_sample=False)
        # load pre-chunker
        self.pre_chunker = RecursiveChunker(cfg.pre_chunk_config)
        return

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk] | list[str]:
        """Chunk the given text into propositions.

        :param text: The text to chunk.
        :param return_str: If True, return the chunks as strings instead of Chunk objects.
            Default is False.
        :return: The propositions of the text.
        """
        paragraphs = self.pre_chunker.chunk(text, return_str=True)
        if not paragraphs:
            return []

        prop_list = []
        input_texts = [f"Title: . Section: . Content: {para}" for para in paragraphs]
        outputs = self.generator.generate(input_texts, generation_config=self.gen_cfg)
        for output in outputs:
            output_text = output[0]
            try:
                props = json.loads(output_text)
                if not isinstance(props, list):
                    logger.warning(f"Output text is not a list: {output_text}")
                    props = [output_text]
                prop_list.extend(props)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse output text as JSON: {output_text}")
                continue

        if return_str:
            return prop_list
        return [Chunk(text=p) for p in prop_list]
