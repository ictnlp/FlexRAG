# Preparing the Retriever
Retriever is one of the most important component in the RAG pipeline. It retrieves the top-k relevant contexts from the knowledge base for a given query. In FlexRAG, there are three types of retrievers: `WebRetriever`, `EditableRetriever`, and `LocalRetriever`. The relationship between these retrievers is shown in the following figure:

```{eval-rst}
.. image:: ../../../assets/Retrievers.png
   :alt: Retrievers
   :align: center
   :width: 50%
```

The difference between these retrievers is as follows:
- `WebRetriever`: A retriever that helps fetching contexts from the web, making it ideal for building personal RAG applications with good timeliness.
- `EditableRetriever`: This retriever retrieves information from a knowledge base and allows easy customization through the `add_passages` method, offering great flexibility in building a tailored knowledge repository.
- `LocalRetriever`: A variant of the `EditableRetriever`, the `LocalRetriever` stores its knowledge base locally, making it easy to load from local storage or the Hugging Face Hub. It offers the best reproducibility.

In this tutorial, we will show you how to load the retriever from the HuggingFace Hub or prepare your own retriever.

## Loading the predefined `LocalRetriever` from HuggingFace Hub
FlexRAG implement two built-in `LocalRetriever`s, including the `DenseRetriever` which employs the semantic similarity between the query and the context to retrieve the top-k relevant contexts, and the `BM25SRetriever` which uses the BM25 algorithm to retrieve the top-k relevant contexts. In this tutorial, we will show you how to load any predefined retriever from the HuggingFace Hub.

```{eval-rst}
.. note::
    In [quickstart](../getting_started/quickstart.md), we provide several examples that employ the predefined `LocalRetriever`. FlexRAG provides several predefined retrievers, which can be accessed from the [HuggingFace Hub](https://huggingface.co/collections/ICTNLP/flexrag-retrievers-67b5373b70123669108a2e59).
```

### Loading the `LocalRetriever` using FlexRAG's entrypoints
The simplest way to load a predefined retriever in a RAG application is by using FlexRAG's entry points. To load the `BM25SRetriever` built on the *wiki2021_atlas* dataset in the GUI application, simply run the following command:

```bash
python -m flexrag.entrypoints.run_interactive \
    assistant_type=modular \
    modular_config.retriever_type='FlexRAG/wiki2021_atlas_bm25s' \
    modular_config.response_type=original \
    modular_config.generator_type=openai \
    modular_config.openai_config.model_name='gpt-4o-mini' \
    modular_config.openai_config.api_key=$OPENAI_KEY \
    modular_config.do_sample=False
```

In the command above, we specify the retriever to be loaded by setting `modular_config.retriever_type='FlexRAG/wiki2021_atlas_bm25s'`. FlexRAG will automatically download this retriever from the HuggingFace Hub and utilize it within the current entrypoint program.

### Loading the `LocalRetriever` in your own code
Another way to load a predefined retriever is by importing FlexRAG as a library. For example, to load the `DenseRetriever` built on the *wiki2021_atlas* dataset in your own code, you can run the following code:

```python
from flexrag.retriever import LocalRetriever

retriever = LocalRetriever.load_from_hub(repo_id='FlexRAG/wiki2021_atlas_bm25s')
passages = retriever.search('What is the capital of France?')
```

In this code snippet, we utilize the `LocalRetriever.load_from_hub` function to download and load the retriever from the HuggingFace Hub.

## Preparing Your Own `EditableRetriever`
FlexRAG provides several `EditableRetriever` retrievers, including `DenseRetriever`, `BM25SRetriever`, `ElasticRetriever` and `TypesenseRetriever`. In this section, we will show you how to build your own retriever for the RAG application.

### Preparing the knowledge base
Before preparing your retriever, you need to prepare the knowledge base. In this example, we will use the Wikipedia knowledge base provided by the [DPR project](https://github.com/facebookresearch/DPR). You can download the knowledge base by running the following command:

```bash
# Download the corpus
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
# Unzip the corpus
gzip -d psgs_w100.tsv.gz
```

You may also utilize your own knowledge base. FlexRAG supports knowledge bases saved in *line-delimited file formats* (such as *.csv, *.jsonl, or *.tsv), where each line represents a piece of knowledge, and each piece can contain multiple fields (such as id, text, etc.). You can store your knowledge base across multiple files or within a single file. In this case, the Wikipedia knowledge base provides three fields: `id`, `title`, and `text`, where the `text` field contains a text chunk of the Wikipedia page, `title` contains the title of the corresponding Wikipedia page, and `id` contains the unique identifier of the knowledge piece. You can check the first line of the knowledge base by running the following command:

```bash
head -n 5 psgs_w100.tsv
```

The output should be like this:

```
id      text    title
<id-1>  <text-1>        <title-1>
<id-2>  <text-2>        <title-2>
...
```

### Preparing the Sparse Retriever
After preparing the knowledge base, you can proceed to build the index for it. This section will demonstrate how to construct a sparse index using the BM25 algorithm. In FlexRAG, the `BM25SRetriever` is a sparse retriever based on the BM25 algorithm. You can execute the following command to build the sparse index for it:

```bash
CORPUS_PATH='[psgs_w100.tsv]'
CORPUS_FIELDS='[title,text]'
DB_PATH=<path_to_database>

python -m flexrag.entrypoints.prepare_index \
    file_paths=$CORPUS_PATH \
    saving_fields=$CORPUS_FIELDS \
    id_field='id' \
    retriever_type=bm25s \
    bm25s_config.database_path=$DB_PATH \
    bm25s_config.indexed_fields=$CORPUS_FIELDS \
    reinit=True
```

In this script, we specify the use of the `BM25SRetriever` with the command-line parameter `retriever_type=bm25s`, and set the input file as *psgs_w100.tsv* using the parameter `file_paths=[psgs_w100.tsv]`. Next, the `saving_fields=$CORPUS_FIELDS` parameter is specified to read the `title` and `text` fields from the knowledge base, and the `bm25s_config.indexed_fields=$CORPUS_FIELDS` parameter is used to build an index based on the `title` and `text` fields. Meanwhile, the parameter `id_field='id'` specifies that the `id` field in the knowledge base serves as the unique identifier for each piece of knowledge (this parameter is optional). Finally, the parameter `bm25s_config.database_path=$DB_PATH` indicates that the prepared retriever will be stored at the `<path_to_database>` location.

### Preparing the Dense Retriever
You can also build a dense retriever by specifying `retriever_type=dense` when constructing the retriever. A dense retriever finds the most relevant documents by computing the semantic similarity between a query and the documents being searched. The query and documents are encoded by a query encoder and a document encoder, respectively, to obtain their corresponding dense vectors.  

To further improve retrieval efficiency, a vector index needs to be built for the dense vectors. Therefore, when constructing the retriever, you need to specify the appropriate document encoder and the relevant parameters for building the vector index.  

You can run the following command to build a dense retriever using Wikipedia as the knowledge base:

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
    dense_config.passage_encoder_config.hf_config.model_path='facebook/contriever-msmarco' \
    dense_config.passage_encoder_config.hf_config.device_id=[0,1,2,3] \  # optional
    dense_config.index_type=faiss \
    dense_config.batch_size=2048 \
    reinit=True
```

In this command, we specify the use of a dense retriever with the parameter retriever_type=dense and designate Wikipedia as the knowledge base using `file_paths=[$CORPUS_PATH]`. Similar to before, we specify saving the `title` and `text` fields while using the `id` field as the unique identifier for each piece of knowledge.

The key difference here is that we explicitly define multiple parameters under `dense_config` in the command-line arguments. These parameters instruct FlexRAG on how to configure the dense retriever. Specifically:
- `dense_config.database_path=$DB_PATH` sets the path where the retriever will be stored.
- `dense_config.encode_fields='[text]'` specifies that the text field will be encoded into semantic vectors and indexed.
- `dense_config.passage_encoder_config.encoder_type=hf` indicates that we are using an encoder from Hugging Face.
- `dense_config.passage_encoder_config.hf_config.model_path='facebook/contriever-msmarco'` explicitly defines `facebook/contriever-msmarco` as the encoder to be used.
- Finally, `dense_config.index_type=faiss` specifies that Faiss will be used to build the vector index.

```{note}
In the above script, we specify the `device_id` as `[0,1,2,3]` to use 4 GPUs for encoding the text field. This configuration will speed up the encoding process. If you do not have multiple GPUs, you can simply set `device_id=[0]` to use a single GPU or `device_id=[]` to use CPU.
```

### Using the Retriever
After preparing the retriever, you can use it in the RAG application or other tasks. For example, you can use the `DenseRetriever` to retrieve the top 5 passages for a given query:

```python
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig
from flexrag.models import EncoderConfig, HFEncoderConfig

cfg = DenseRetrieverConfig(
    database_path='<path_to_database>',
    top_k=5,
    query_encoder_config=EncoderConfig(
        encoder_type='hf',
        hf_config=HFEncoderConfig(
            model_path='facebook/contriever-msmarco',
            device_id=[0]
        )
    )
)
retriever = DenseRetriever(cfg)
passages = retriever.search('What is the capital of France?')[0]
print(passages)
```

You can also evaluate your retriever using FlexRAG's predefined `ASSISTANT` in any RAG tasks. For example, to evaluate the `BM25SRetriever` on the test set of the *Natural Questions* dataset, you can run the following script:

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
    eval_config.response_preprocess.processor_type=[simplify_answer]
```

### Uploading the Retriever to the HuggingFace Hub
To share your retriever with the community, you can upload it to the HuggingFace Hub. For example, to upload the `DenseRetriever` to the HuggingFace Hub, you can run the following code:

```python
from flexrag.retrievers import DenseRetriever, DenseRetrieverConfig
from flexrag.models import EncoderConfig, HFEncoderConfig

cfg = DenseRetrieverConfig(
    database_path='<path_to_database>',
    top_k=5,
    query_encoder_config=EncoderConfig(
        encoder_type='hf',
        hf_config=HFEncoderConfig(
            model_path='facebook/contriever-msmarco',
            device_id=[0]
        )
    ),
    passage_encoder_config=EncoderConfig(
        encoder_type='hf',
        hf_config=HFEncoderConfig(
            model_path='facebook/contriever-msmarco',
            device_id=[0]
        )
    )
)
retriever = DenseRetriever(cfg)
retriever.save_to_hub(repo_id="<your-repo-id>", token="<your-hf-token>")
```

In this code, you need to specify the `repo_id` and `token` to upload the retriever to the HuggingFace Hub. You can find the `token` in your HuggingFace [account settings](https://huggingface.co/settings/tokens). After uploading the retriever, you can share the retriever with the community by sharing the link to the HuggingFace Hub.

```{important}
To make your shared `DenseRetriever` accessible to the community, you need to make sure the query encoder and the passage encoder are **configured** and **accessible** to the public. In this example, the `facebook/contriever-msmarco` model is hosted on the HuggingFace Hub, so users can access the model without any additional configuration. If you use a custom model, uploading your model to the HuggingFace Hub is recommended.
```

<!-- ## Evaluating the Retriever via `MTEB` Retrieval tasks
FlexRAG offers a set of predefined tasks designed to evaluate the retriever. Unlike the `MTEB` Benchmark, which focuses solely on evaluating the encoding part of the retrieval process, FlexRAG assesses the entire retrieval pipeline, including both the encoding and indexing stages.

You can evaluate your retriever using the `MTEB` retrieval tasks by running the following command: -->
