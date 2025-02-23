# Building your own RAG Assistant
FlexRAG provides a flexible and modularized framework for building RAG assistants. You can build your own RAG assistant by defining your own `Assistant` class and registering it with the `ASSISTANTS` decorator.

## Define the Assistant Class
To build your own RAG assistant, you can create a new Python file and import the necessary FlexRAG modules. Here is an example of how to build a RAG assistant:

```python
from dataclasses import dataclass

from flexrag.assistant import ASSISTANTS, AssistantBase
from flexrag.models import OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig


@dataclass
class SimpleAssistantConfig(DenseRetrieverConfig, OpenAIGeneratorConfig): ...


@ASSISTANTS("simple", config_class=SimpleAssistantConfig)
class SimpleAssistant(AssistantBase):
    def __init__(self, config: SimpleAssistantConfig):
        self.retriever = DenseRetriever(config)
        self.generator = OpenAIGenerator(config)
        return

    def answer(self, question: str) -> str:
        prompt = ChatPrompt()
        context = self.retriever.search(question)[0]
        prompt_str = ""
        for ctx in context:
            prompt_str += f"Question: {question}\nContext: {ctx.data['text']}"
        prompt.update(ChatTurn(role="user", content=prompt_str))
        response = self.generator.chat([prompt])[0][0]
        prompt.update(ChatTurn(role="assistant", content=response))
        return response
```


### Running your own RAG Application
After defining the `SimpleAssistant` class and registering it with the `ASSISTANTS` decorator, you can your assistant with any FlexRAG's entrypoints by adding the `user_module=<your_module_path>` argument to the command. 

For example, you can evaluate your assistant on the Natural Questions dataset using the following command:

```bash
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>
MODULE_PATH=<path_to_simple_assistant_module>

python -m flexrag.entrypoints.run_assistant \
    user_module=${MODULE_PATH} \
    name=nq \
    split=test \
    assistant_type=simple \
    simple_config.model_name='gpt-4o-mini' \
    simple_config.api_key=${OPENAI_KEY} \
    simple_config.database_path=${DB_PATH} \
    simple_config.index_type=faiss \
    simple_config.query_encoder_config.encoder_type=hf \
    simple_config.query_encoder_config.hf_config.model_path='facebook/contriever-msmarco' \
    simple_config.query_encoder_config.hf_config.device_id=[0] \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
```

In [flexrag_examples](https://github.com/ictnlp/flexrag_examples) repository, we provide several detailed examples of how to build a RAG assistant.
