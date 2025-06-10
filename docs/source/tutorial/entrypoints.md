# FlexRAG Entrypoints
FlexRAG entrypoints refer to a series of command-line executable programs provided by FlexRAG. These programs can help you build indexes, evaluate RAG assistants, manage retrieval caches, launch GUI applications, and more. Each entry point allows parameters to be passed either via the command line or through a configuration file. This tutorial will guide you on how to use these entrypoints and customize your workflow with parameters.

## Provided Entrypoints
In this section, we will introduce all FlexRAG entrypoints and their corresponding configuration structures.

### Adding Passages to the Retriever
This entrypoint is used to add the passage to the retriever. You can use this entrypoint by running `python -m flexrag.entrypoints.prepare_retriever`.
The defination of the configuration structure for the `prepare_retriever` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.prepare_retriever::Config
    :members:
    :noindex:
    :show-inheritance:
    :exclude-members: dump,dumps,load,loads
```

### Adding the Index for FlexRetriever
This entrypoint is used to add the index for the `FlexRetriever`. You can use this entrypoint by running `python -m flexrag.entrypoints.add_index`.
The defination of the configuration structure for the `add_index` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.add_index::Config
    :members:
    :noindex:
    :show-inheritance:
    :exclude-members: dump,dumps,load,loads
```

### Evaluating the Assistant
This entrypoint is used to evaluate the assistant on a given dataset. You can use this entrypoint by running `python -m flexrag.entrypoints.eval_assistant`.
The defination of the configuration structure for the `eval_assistant` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.eval_assistant::Config
    :members:
    :noindex:
    :show-inheritance:
    :exclude-members: dump,dumps,load,loads
```

### Running GUI Application
This entrypoint is used to run the assistant using the built-in Gradio GUI interface. You can use this entrypoint by running `python -m flexrag.entrypoints.run_interactive`.
The defination of the configuration structure for the `run_interactive` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.run_interactive::Config
    :members:
    :noindex:
    :show-inheritance:
    :exclude-members: dump,dumps,load,loads
```

### Cache Management
This entrypoint is used to manage the cache for the retrievers. You can use this entrypoint by running `python -m flexrag.entrypoints.cache`.
The defination of the configuration structure for the `cache` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.cache::Config
    :members:
    :noindex:
    :show-inheritance:
    :exclude-members: dump,dumps,load,loads
```

```{tip}
If you wish to disable the Cache during retrieval, you can set the environment variable by `export DISABLE_CACHE=True`.
```

### Deploying the Retriever
FlexRAG also provides an entrypoint to deploy the retriever as a service. This is helpful when you want to use the retriever to fine-tune your own RAG assistant or when you want to use the retriever in a production demonstration.
You can use this entrypoint by running `python -m flexrag.entrypoints.serve_retriever`.
The defination of the configuration structure for the `deploy` entrypoint is as follows:

```{eval-rst}
.. autoclass:: flexrag.entrypoints.serve_retriever::Config
    :members:
    :noindex:
    :show-inheritance:
    :exclude-members: dump,dumps,load,loads
```


## Configuration Management
FlexRAG employs `dataclass` and [hydra-core](https://github.com/facebookresearch/hydra) for configuration management, which brings remarkable clarity to the complex configurations within the RAG pipeline. Moreover, you can pass parameters to the FlexRAG's entrypoints either via the command line or through configuration files. This section will illustrate how to utilize both methods to convey parameters to the FlexRAG entry point.

### Passing Configuration via Command Line
Configurations can be passed via the command line using the `<config_key>=<config_value>` format. For example, you can run the following command to set the configuration for a *ModularAssistant* with a *FlexRetriever* and an *OpenAIGenerator*:
```bash
RETRIEVER_PATH=<path_to_retriever>

python -m flexrag.entrypoints.run_interactive \
    assistant_type=modular \
    modular_config.used_fields=[title,text] \
    modular_config.retriever_type=flex \
    modular_config.flex_config.top_k=5 \
    modular_config.flex_config.retriever_path=${RETRIEVER_PATH} \
    modular_config.flex_config.used_indexes=[bm25] \
    modular_config.response_type=original \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name='gpt-4o-mini' \
    modular_config.openai_config.api_key=${OPENAI_KEY} \
    modular_config.do_sample=False
```

### Passing Configuration via Configuration File
Configurations can also be passed via a `YAML` file. For example, you can create a `config.yaml` file with the following content:
```yaml
# The `defaults` option specifies the default configuration to be used.
# This three lines cannot be omitted.
defaults:
    - default
    - _self_

# The configuration passed to the entrypoint.
assistant_type: modular
modular_config:
    used_fields: [title, text]
    retriever_type: flex
    flex_config:
        top_k: 5
        retriever_path: <path_to_retriever>
        used_indexes: [bm25]
    response_type: original
    generator_type: openai
    openai_config:
        model_name: "gpt-4o-mini"
        api_key: <your_openai_key>
    do_sample: False
```

Then, you can run the following command to use the configuration file:
```bash
python -m flexrag.entrypoints.eval_assistant \
    --config-path '<parent_path_of_your_config_file>' \
    --config-name config
```

```{tip}
For more detailed usage, we recommend you to go through the [Hydra documentation](https://hydra.cc/docs/intro/) to get a better understanding of the concepts and features.
```
