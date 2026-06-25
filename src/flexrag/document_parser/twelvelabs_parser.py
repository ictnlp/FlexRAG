import os
from typing import Optional

from flexrag.utils import configure

from .document_parser_base import DOCUMENTPARSERS, Document, DocumentParserBase


@configure
class TwelveLabsVideoParserConfig:
    """Configuration for TwelveLabsVideoParser.

    TwelveLabs `Pegasus <https://docs.twelvelabs.io/>`_ is a video-understanding
    model. This parser turns a video (a local file or a publicly reachable URL)
    into a textual :class:`Document` by prompting Pegasus, so that videos can be
    indexed and retrieved alongside ordinary documents in FlexRAG.

    :param model: The Pegasus model to use. Default is "pegasus1.5".
    :type model: str
    :param prompt: The instruction passed to Pegasus to describe the video.
        Defaults to a generic summarization prompt.
    :type prompt: str
    :param max_tokens: The maximum number of tokens Pegasus may generate.
        Defaults to 2048.
    :type max_tokens: int
    :param api_key: The API key for the TwelveLabs API.
        If not provided, it will use the environment variable `TWELVELABS_API_KEY`.
        Defaults to None.
    :type api_key: Optional[str]
    """

    model: str = "pegasus1.5"
    prompt: str = (
        "Describe this video in detail, including the main events, "
        "people, objects, and any spoken or on-screen text."
    )
    max_tokens: int = 2048
    api_key: Optional[str] = None


@DOCUMENTPARSERS("twelvelabs_video", config_class=TwelveLabsVideoParserConfig)
class TwelveLabsVideoParser(DocumentParserBase):
    """Parse a video into a textual ``Document`` using TwelveLabs Pegasus."""

    def __init__(self, config: TwelveLabsVideoParserConfig):
        try:
            from twelvelabs import TwelveLabs
        except ImportError:
            raise ImportError(
                "TwelveLabs is not installed. Please install it via `pip install twelvelabs`."
            )

        api_key = config.api_key or os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for TwelveLabs is not provided. "
                "Please set it in the configuration or as an environment variable 'TWELVELABS_API_KEY'."
            )
        self.client = TwelveLabs(api_key=api_key)
        self.model = config.model
        self.prompt = config.prompt
        self.max_tokens = config.max_tokens
        return

    def parse(self, input_file_path: str) -> Document:
        # A publicly reachable URL is analysed server-side; a local file is
        # uploaded as an asset first and analysed by its asset id.
        if input_file_path.startswith(("http://", "https://")):
            video = {"type": "url", "url": input_file_path}
        else:
            assert os.path.exists(
                input_file_path
            ), f"Video file not found: {input_file_path}"
            with open(input_file_path, "rb") as f:
                asset = self.client.assets.create(method="direct", file=f)
            video = {"type": "asset_id", "asset_id": asset.id}

        response = self.client.analyze(
            model_name=self.model,
            video=video,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
        )
        return Document(
            source_file_path=input_file_path,
            text=response.data,
            title=os.path.basename(input_file_path),
        )
