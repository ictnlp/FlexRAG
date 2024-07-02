import pathlib
import sys
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from utils import read_data

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


from kylin.retriever import BM25Retriever, BM25RetrieverConfig


@dataclass
class Config:
    retriever_config: BM25RetrieverConfig = field(default_factory=BM25RetrieverConfig)
    corpus_path: list[str] = MISSING
    reinit: bool = False


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.1", config_path=None, config_name="default")
def main(cfg: Config):
    default_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(default_cfg, cfg)
    retriever = BM25Retriever(cfg.retriever_config)
    retriever.add_passages(passages=read_data(cfg.corpus_path), reinit=cfg.reinit)
    retriever.close()


if __name__ == "__main__":
    main()
