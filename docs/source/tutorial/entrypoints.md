# FlexRAG Entrypoints
FlexRAG entrypoints refer to a series of command-line executable programs provided by FlexRAG. These programs can help you build indexes, evaluate RAG assistants, manage retrieval caches, launch GUI applications, and more. Each entry point allows parameters to be passed either via the command line or through a configuration file. This tutorial will guide you on how to use these entrypoints and customize your workflow with parameters.

## Provided Entrypoints
In this section, we will introduce all FlexRAG entrypoints and their corresponding configuration structures.

### Preparing the Retriever Index
This entrypoint is used to prepare the retriever index. You can use this entrypoint by running `python -m flexrag.entrypoints.prepare_index`.
The defination of the configuration structure for the `prepare_index` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.prepare_index::Config
    :members:
    :noindex:
    :show-inheritance:
```

### Rebuilding the Retriever Index
This entrypoint is used to rebuild the index for the `DenseRetriever`. You can use this entrypoint by running `python -m flexrag.entrypoints.rebuid_index`.
The defination of the configuration structure for the `rebuid_index` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.rebuild_index::DenseRetrieverConfig
    :members:
    :noindex:
    :show-inheritance:
```

### Evaluating the Assistant
This entrypoint is used to evaluate the assistant on a given dataset. You can use this entrypoint by running `python -m flexrag.entrypoints.run_assistant`.
The defination of the configuration structure for the `run_assistant` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.run_assistant::Config
    :members:
    :noindex:
    :show-inheritance:
```

### Running GUI Application
This entrypoint is used to run the assistant using the built-in Gradio GUI interface. You can use this entrypoint by running `python -m flexrag.entrypoints.run_interactive`.
The defination of the configuration structure for the `run_interactive` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.run_interactive::Config
    :members:
    :noindex:
    :show-inheritance:
```

### Cache Management
This entrypoint is used to manage the cache for the retrievers. You can use this entrypoint by running `python -m flexrag.entrypoints.cache`.
The defination of the configuration structure for the `cache` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.cache::Config
    :members:
    :noindex:
    :show-inheritance:
```

```{tip}
If you wish to disable the Cache during retrieval, you can set the environment variable by `export DISABLE_CACHE=True`.
```

### Evaluating the Generated Responses
This entrypoint is used to evaluate the generated responses. You can use this entrypoint by running `python -m flexrag.entrypoints.evaluate`.
The defination of the configuration structure for the `evaluate` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.evaluate::Config
    :members:
    :noindex:
    :show-inheritance:
```


## Configuration Management
FlexRAG employs `dataclass` and [hydra-core](https://github.com/facebookresearch/hydra) for configuration management, which brings remarkable clarity to the complex configurations within the RAG pipeline. Moreover, you can pass parameters to the FlexRAG's entrypoints either via the command line or through configuration files. This section will illustrate how to utilize both methods to convey parameters to the FlexRAG entry point.

### Passing Configuration via Command Line
Configurations can be passed via the command line using the `<config_key>=<config_value>` format. For example, you can run the following command to set the configuration for a *modular assistant* with a *dense retriever* and an *OpenAI generator*:
```bash
python -m flexrag.entrypoints.run_interactive \
    assistant_type=modular \
    modular_config.used_fields=[title,text] \
    modular_config.retriever_type=dense \
    modular_config.dense_config.top_k=5 \
    modular_config.dense_config.database_path=${DB_PATH} \
    modular_config.dense_config.query_encoder_config.encoder_type=hf \
    modular_config.dense_config.query_encoder_config.hf_config.model_path='facebook/contriever-msmarco' \
    modular_config.dense_config.query_encoder_config.hf_config.device_id=[0] \
    modular_config.response_type=short \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name='gpt-4o-mini' \
    modular_config.openai_config.api_key=$OPENAI_KEY \
    modular_config.do_sample=False
```

### Passing Configuration via Configuration File
Configurations can also be passed via a `YAML` file. For example, you can create a `config.yaml` file with the following content:
```yaml
assistant_type: modular
modular_config:
  used_fields: [title, text]
  retriever_type: dense
  dense_config:
    top_k: 5
    database_path: ${DB_PATH}
    query_encoder_config:
      encoder_type: hf
      hf_config:
        model_path: facebook/contriever-msmarco
        device_id: [0]
    response_type: short
```

Then, you can run the following command to use the configuration file:
```bash
python -m flexrag.entrypoints.run_assistant \
    --config-file config.yaml
```

```{tip}
For more detailed usage, we recommend you to go through the [Hydra documentation](https://hydra.cc/docs/intro/) to get a better understanding of the concepts and features.
```



## Defining Your Own Configuration
You can define your own configuration structure by creating a new `dataclass`. For example, you can define a new configuration structure for a custom assistant as follows:

```python
from dataclass import dataclass
from omegaconf import MISSING

from flexrag.retriever import DenseRetrieverConfig
from flexrag.models import OpenAIGeneratorConfig


@dataclass
class CustomAssistantConfig(DenseRetrieverConfig, OpenAIGeneratorConfig):
    prompt_path: str = MISSING

```
