# Quickstart

## Step 1. Preparing the Retriever

### Downloading the Corpus
Before starting you RAG application, you need to download the corpus. In this example, we will use the wikipedia corpus provided by [DPR project](https://github.com/facebookresearch/DPR) as the corpus. You can download the corpus by running the following command:
```bash
# Download the corpus
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
# Unzip the corpus
gzip -d psgs_w100.tsv.gz
```

### Preparing the Index
After downloading the corpus, you need to build the index for the retriever. If you want to employ the dense retriever, you can simply run the following command to build the index:
```bash
CORPUS_PATH='psgs_w100.tsv'
CORPUS_FIELDS='[title,text]'
DB_PATH=<path_to_database>

python -m flexrag.entrypoints.prepare_index \
    file_paths=[$CORPUS_PATH] \
    saving_fields=$CORPUS_FIELDS \
    id_field='id' \
    retriever_type=dense \
    dense_config.database_path=$DB_PATH \
    dense_config.encode_fields='[text]' \
    dense_config.passage_encoder_config.encoder_type=hf \
    dense_config.passage_encoder_config.hf_config.model_path='facebook/contriever' \
    dense_config.passage_encoder_config.hf_config.device_id=[0,1,2,3] \
    dense_config.index_type=faiss \
    dense_config.faiss_config.batch_size=4096 \
    dense_config.faiss_config.log_interval=100000 \
    dense_config.batch_size=2048 \
    dense_config.log_interval=100000 \
    reinit=True
```

If you want to employ the sparse retriever, you can run the following command to build the index:
```bash
CORPUS_PATH='psgs_w100.tsv'
CORPUS_FIELDS='[title,text]'
DB_PATH=<path_to_database>

python -m flexrag.entrypoints.prepare_index \
    file_paths=[$CORPUS_PATH] \
    saving_fields=$CORPUS_FIELDS \
    id_field='id' \
    retriever_type=bm25s \
    bm25s_config.database_path=$DB_PATH \
    bm25s_config.indexed_fields='[title,text]' \
    bm25s_config.method=lucene \
    bm25s_config.batch_size=512 \
    bm25s_config.log_interval=100000 \
    reinit=True
```

## Step 2. Running Modular Assistant
When the index is ready, you can run RAG `Assistant` provided by FlexRAG. Here is an example of how to run a `Modular Assistant`.

### Running the FlexRAG Modular Assistant
```bash
python -m flexrag.entrypoints.run_interactive \
    assistant_type=modular \
    modular_config.used_fields=[title,text] \
    modular_config.retriever_type=dense \
    modular_config.dense_config.top_k=5 \
    modular_config.dense_config.database_path=${DB_PATH} \
    modular_config.dense_config.query_encoder_config.encoder_type=hf \
    modular_config.dense_config.query_encoder_config.hf_config.model_path='facebook/contriever' \
    modular_config.dense_config.query_encoder_config.hf_config.device_id=[0] \
    modular_config.response_type=short \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name='gpt-4o-mini' \
    modular_config.openai_config.api_key=$OPENAI_KEY \
    modular_config.do_sample=False
```

### Evaluating the FlexRAG Modular Assistants
You can evaluate your RAG assistant on several knowledge intensive datasets with great ease. FlexRAG support Knowledge intensive datasets provided by [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG), which can be access from [huggingface](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets). The following command let you evaluate the `Modular Assistant` with dense retriever on the Natural Questions (NQ) dataset:
```bash
OUTPUT_PATH=<path_to_output>
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>

python -m flexrag.entrypoints.run_assistant \
    name=nq \
    split=test \
    output_path=${OUTPUT_PATH} \
    assistant_type=modular \
    modular_config.used_fields=[title,text] \
    modular_config.retriever_type=dense \
    modular_config.dense_config.top_k=10 \
    modular_config.dense_config.database_path=${DB_PATH} \
    modular_config.dense_config.query_encoder_config.encoder_type=hf \
    modular_config.dense_config.query_encoder_config.hf_config.model_path='facebook/contriever' \
    modular_config.dense_config.query_encoder_config.hf_config.device_id=[0] \
    modular_config.response_type=short \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name='gpt-4o-mini' \
    modular_config.openai_config.api_key=$OPENAI_KEY \
    modular_config.do_sample=False \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.context_preprocess.processor_type=[simplify_answer] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
```

Similarly, you can evaluate the `Modular Assistant` with sparse retriever on the Natural Questions dataset:
```bash
OUTPUT_PATH=<path_to_output>
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>

python -m flexrag.entrypoints.run_assistant \
    name=nq \
    split=test \
    output_path=${OUTPUT_PATH} \
    assistant_type=modular \
    modular_config.used_fields=[title,text] \
    modular_config.retriever_type=bm25s \
    modular_config.bm25s_config.top_k=10 \
    modular_config.bm25s_config.database_path=${DB_PATH} \
    modular_config.response_type=short \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name='gpt-4o-mini' \
    modular_config.openai_config.api_key=$OPENAI_KEY \
    modular_config.do_sample=False \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.context_preprocess.processor_type=[simplify_answer] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
```

You can also evaluate your own assistant by adding the `user_module=<your_module_path>` argument to the command.

## Step 3. Developing Your Own RAG Assistant
FlexRAG provides a flexible and modularized framework for building RAG assistants. You can build your own RAG assistant by defining your own `Assistant` class and registering it with the `ASSISTANTS` decorator.

### Building your own RAG Assistant
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
After defining the `SimpleAssistant` class and registering it with the `ASSISTANTS` decorator, you can run the assistant with the following command:
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
    simple_config.query_encoder_config.hf_config.model_path='facebook/contriever' \
    simple_config.query_encoder_config.hf_config.device_id=[0] \
    eval_config.metrics_type=[retrieval_success_rate,generation_f1,generation_em] \
    eval_config.retrieval_success_rate_config.eval_field=text \
    eval_config.response_preprocess.processor_type=[simplify_answer] \
    log_interval=10
```
In [flexrag_examples](https://github.com/ictnlp/flexrag_examples) repository, we provide several detailed examples of how to build a RAG assistant.
