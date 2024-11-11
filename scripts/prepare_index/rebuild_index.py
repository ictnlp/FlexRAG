import logging
import pathlib
import sys

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


from kylin.retriever import DenseRetriever, DenseRetrieverConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="default", node=DenseRetrieverConfig)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: DenseRetrieverConfig):
    default_cfg = OmegaConf.structured(DenseRetrieverConfig)
    cfg = OmegaConf.merge(default_cfg, cfg)

    # rebuild index
    retriever = DenseRetriever(cfg)
    retriever.rebuild_index()
    retriever.close()
    return


if __name__ == "__main__":
    main()
