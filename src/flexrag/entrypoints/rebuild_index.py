import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from flexrag.retriever import DenseRetriever, DenseRetrieverConfig


cs = ConfigStore.instance()
cs.store(name="default", node=DenseRetrieverConfig)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: DenseRetrieverConfig):
    default_cfg = OmegaConf.structured(DenseRetrieverConfig)
    cfg = OmegaConf.merge(default_cfg, cfg)

    # rebuild index
    retriever = DenseRetriever(cfg, no_check=True)
    retriever.build_index(rebuild=True)
    return


if __name__ == "__main__":
    main()
