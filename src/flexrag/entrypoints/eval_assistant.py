import sys
from dataclasses import field
from typing import Annotated

import hydra
from hydra.core.config_store import ConfigStore

from flexrag.assistant import ASSISTANTS
from flexrag.datasets import QA_DATASETS
from flexrag.tasks import QATask, QATaskConfig
from flexrag.utils import (
    LOGGER_MANAGER,
    Choices,
    configure,
    extract_config,
    load_user_module,
)

# load user modules before loading config
for arg in sys.argv:
    if arg.startswith("user_module="):
        load_user_module(arg.split("=")[1])
        sys.argv.remove(arg)


AssistantConfig = ASSISTANTS.make_config(config_name="AssistantConfig")


@configure
class Config(AssistantConfig):
    task_data_split: Annotated[
        str,
        Choices(
            "qa_nq_test",
            # ...
        ),
    ]


cs = ConfigStore.instance()
cs.store(name="default", node=Config)
logger = LOGGER_MANAGER.get_logger("eval_assistant")


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    config = extract_config(config, Config)

    match config.task:
        case "qa":
            # load dataset
            testset = QA_DATASETS.load(config.qa_data_config)
            # load assistant
            assistant = ASSISTANTS.load(config)
            task = QATask(config.qa_task_config)
            task.setup(assistant=assistant, dataset=testset)
            task.run()
        case _:
            raise ValueError(f"Unknown task: {config.task}")
    return


if __name__ == "__main__":
    main()
