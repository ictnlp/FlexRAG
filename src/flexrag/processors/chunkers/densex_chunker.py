import json
from dataclasses import field

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.models.generators import GenerationConfig, HFGenerator, HFGeneratorConfig

from .basic_chunkers import RecursiveChunker, RecursiveChunkerConfig
from .chunker_base import CHUNKERS, Chunk, ChunkerBase

logger = LOGGER_MANAGER.get_logger("flexrag.processors.chunkers.densex_chunker")


@configure
class DenseXChunkerConfig:
    """Configuration for DenseXChunker.

    :param model_path: The path to the propositionizer model.
        Default is "chentong00/propositionizer-wiki-flan-t5-large".
    :type model_path: str
    :param device_id: The list of device IDs to use for the model.
        Default is an empty list, which means using CPU.
    :type device_id: list[int]
    :param pre_chunk_config: The configuration for the pre-chunker
        used to split the text into paragraphs.
    :type pre_chunk_config: RecursiveChunkerConfig
    """

    model_path: str = "chentong00/propositionizer-wiki-flan-t5-large"
    device_id: list[int] = field(default_factory=list)
    pre_chunk_config: RecursiveChunkerConfig = field(
        default_factory=RecursiveChunkerConfig
    )


@CHUNKERS("densex", config_class=DenseXChunkerConfig)
class DenseXChunker(ChunkerBase):
    """`DenseXChunker <https://arxiv.org/abs/2312.06648>`_ uses a propositionizer
    model to split text into propositions.
    """

    def __init__(self, cfg: DenseXChunkerConfig) -> None:
        # load generator
        self.generator = HFGenerator(
            HFGeneratorConfig(
                model_path=cfg.model_path,
                device_id=cfg.device_id,
                model_type="seq2seq",
            )
        )
        self.gen_cfg = GenerationConfig(max_new_tokens=512, do_sample=False)
        # load pre-chunker
        self.pre_chunker = RecursiveChunker(cfg.pre_chunk_config)
        return

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk] | list[str]:
        """Chunk the given text into propositions.

        :param text: The text to chunk.
        :type text: str
        :param return_str: If True, return the chunks as strings instead of Chunk objects.
            Default is False.
        :type return_str: bool
        :return: The propositions of the text.
        :rtype: list[Chunk] | list[str]
        """
        paragraphs = self.pre_chunker.chunk(text, return_str=True)
        if not paragraphs:
            return []

        prop_list = []
        for para in paragraphs:
            input_text = f"Title: . Section: . Content: {para}"
            output_text = self.generator.generate(
                input_text, generation_config=self.gen_cfg
            )
            try:
                props = json.loads(output_text)
                if not isinstance(props, list):
                    logger.warning(f"Output text is not a list: {output_text}")
                    props = [output_text]
                prop_list.extend(props)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse output text as JSON: {output_text}")
                prop_list = []

        if return_str:
            return prop_list
        return [Chunk(text=p) for p in prop_list]
