import sys

import hydra
from hydra.core.config_store import ConfigStore

from flexrag.tasks import TASKS, TaskConfig
from flexrag.utils import LOGGER_MANAGER, extract_config, load_user_module

# load user modules before loading config
for arg in sys.argv:
    if arg.startswith("user_module="):
        load_user_module(arg.split("=")[1])
        sys.argv.remove(arg)


cs = ConfigStore.instance()
cs.store(name="default", node=TaskConfig)
logger = LOGGER_MANAGER.get_logger("eval_assistant")


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: TaskConfig):  # type: ignore
    config = extract_config(config, TaskConfig)
    task = TASKS.load(config)
    task.run()
    return


if __name__ == "__main__":
    main()
