import csv
import json
from dataclasses import dataclass, field
from glob import glob

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from flexrag.chunking import CHUNKERS, ChunkerConfig
from flexrag.common_dataclass import Context
from flexrag.document_parser import DOCUMENTPARSERS, DocumentParserConfig
from flexrag.text_process import TextProcessPipeline, TextProcessPipelineConfig
from flexrag.utils import LOGGER_MANAGER, SimpleProgressLogger

logger = LOGGER_MANAGER.get_logger("flexrag.entrypoints.prepare_corpus")


@dataclass
class Config(DocumentParserConfig, ChunkerConfig, TextProcessPipelineConfig):
    """The configuration for prepare corpus.
    The documents will be parsed by the DocumentParser specified in the config and then chunked by the Chunker.

    :param document_paths: The paths to the documents, allow glob patterns.
    :type document_paths: list[str]
    :param output_path: The path to save the prepared corpus. Required.
    :type output_path: str
    """

    document_paths: list[str] = field(default_factory=list)
    output_path: str = MISSING


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


class ContextWriter:
    def __init__(self, file_path: str):
        self.f = open(file_path, "w")
        self.save_format = file_path.split(".")[-1]
        self.writer = None
        return

    def write(self, ctx: Context):
        ctx = ctx.to_dict()
        ctx.update(ctx.pop("meta_data"))
        ctx.update(ctx.pop("data"))
        match self.save_format:
            case "jsonl":
                self.f.write(json.dumps(ctx))
                self.f.write("\n")
            case "tsv":
                if self.writer is None:
                    self.writer = csv.DictWriter(self.f, ctx.keys(), delimiter="\t")
                    self.writer.writeheader()
                self.writer.writerow(ctx)
            case "csv":
                if self.writer is None:
                    self.writer = csv.DictWriter(self.f, ctx.keys())
                    self.writer.writeheader()
                self.writer.writerow(ctx)
            case _:
                raise ValueError(f"Unsupported save format: {self.save_format}")
        return

    def __enter__(self) -> "ContextWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()
        return


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
    # parse paths
    if isinstance(cfg.document_paths, str):
        document_paths = [cfg.document_paths]
    else:
        document_paths = cfg.document_paths
    document_paths = [glob(p) for p in document_paths]
    document_paths = [p for doc_path in document_paths for p in doc_path]
    logger.info(f"Found {len(document_paths)} documents.")

    # load document parser and chunker
    parser = DOCUMENTPARSERS.load(cfg)
    chunker = CHUNKERS.load(cfg)
    processor = TextProcessPipeline(cfg)

    # parse the documents
    global_id = 0
    p_logger = SimpleProgressLogger(logger, total=len(document_paths))
    with ContextWriter(cfg.output_path) as writer:
        for path in document_paths:
            in_doc_id = 0
            document = parser.parse(path)
            if chunker is not None:
                chunks = chunker.chunk(document.text)
                for chunk in chunks:
                    text = processor(chunk.text)
                    if text is None:
                        continue
                    writer.write(
                        Context(
                            context_id=f"{global_id}-{in_doc_id}",
                            data={"text": text, "title": document.title},
                            source=path,
                            meta_data={"source_file_path": document.source_file_path},
                        )
                    )
                    in_doc_id += 1
            else:
                text = processor(document.text)
                if text is None:
                    continue
                writer.write(
                    Context(
                        context_id=f"{global_id}-{in_doc_id}",
                        data={"text": text, "title": document.title},
                        source=path,
                        meta_data={"source_file_path": document.source_file_path},
                    )
                )
                in_doc_id += 1
            global_id += 1
            p_logger.update(desc="Parsing documents")
    return


if __name__ == "__main__":
    main()
