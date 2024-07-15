import pathlib
import sys
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


from kylin.retriever import DenseRetriever, DenseRetrieverConfig
from kylin.utils import read_data


@dataclass
class Config:
    retriever_config: DenseRetrieverConfig = field(default_factory=DenseRetrieverConfig)
    corpus_path: list[str] = MISSING
    reinit: bool = False


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.1", config_path=None, config_name="default")
def main(cfg: Config):
    default_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(default_cfg, cfg)
    retriever = DenseRetriever(cfg.retriever_config)
    retriever.add_passages(passages=read_data(cfg.corpus_path), reinit=cfg.reinit)
    retriever.close()


if __name__ == "__main__":
    main()
