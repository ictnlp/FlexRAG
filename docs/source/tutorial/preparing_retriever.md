# Preparing the Retriever

FlexRAG provides several `EditableRetriever` retrievers (in FlexRAG, the `EditableRetriever` is a concept referring to a retriever that includes the `add_passages` and `clean` methods, allowing you to build the retriever using your own knowledge base), including `DenseRetriever`, `BM25SRetriever`, `ElasticRetriever` and `TypesenseRetriever`. In this tutorial, we will show you how to prepare the retriever for the RAG application.

## Downloading the Corpus
Before preparing your retriever, you need to prepare the corpus. In this example, we will use the Wikipedia corpus provided by the [DPR project](https://github.com/facebookresearch/DPR). You can download the corpus by running the following command:

```bash
# Download the corpus
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
# Unzip the corpus
gzip -d psgs_w100.tsv.gz
```

You can also use your own corpus. The corpus could be a single file or a directory containing multiple files. The allowed file formats are `.tsv`, `.csv`, `.jsonl`. The corpus should contain one *chunk* per line. Each *chunk* should have at least one field that contains the information of the chunk. In this case, the Wikipedia corpus provides three fields: `id`, `title`, and `text`, where the `text` field contains the text of the chunk, `title` contains the title of the corresponding Wikipedia page, and `id` contains the unique identifier of the chunk. You can check the first line of the corpus by running the following command:

```bash
head -n 1 psgs_w100.tsv
```

The output should be like this:

```
id      text    title
```

## Preparing the Sparse Retriever
After downloading the corpus, you need to build the index for the retriever. For example, if you want to employ the `BM25SRetriever`, you can simply run the following command to build the index:

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

In this command, we specify the retriever as `BM25S` and use the downloaded *psgs_w100.tsv* as the corpus. We designate the `title` and `text` fields from the corpus to be stored in the database and create index for the information saved in these two fields. We specify the `id` field as the unique identifier for each chunk (if you do not specify an id field or if the corpus does not include an id field, FlexRAG will automatically assign sequential numbers to each chunk as unique identifiers). Finally, the prepared BM25S retriever will be stored in the directory <path_to_database>.

## Preparing the Dense Retriever
You can also employ the `DenseRetriever` as your retriever. To build the index for the `DenseRetriever`, you can run the following command:

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
    dense_config.passage_encoder_config.hf_config.device_id=[0,1,2,3] \  # optional
    dense_config.index_type=faiss \
    dense_config.batch_size=2048 \
    reinit=True
```

Similarly, we specify the retriever as `DenseRetriever` and use the downloaded *psgs_w100.tsv* as the corpus. We designate the `title` and `text` fields from the corpus to be stored in the database and specify the `id` field as the unique identifier for each chunk.
In addition, we use the `facebook/contriever` model to encode the `text` field and store the encoded vectors in the database. Finally, the prepared `DenseRetriever` will be stored in the directory <path_to_database>.

Note that we specify the `device_id` as `[0,1,2,3]` to use 4 GPUs for encoding the text field. This configuration will speed up the encoding process. If you do not have multiple GPUs, you can simply set `device_id=[0]` to use a single GPU or `device_id=[]` to use CPU.

## Using the Retriever
After preparing the retriever, you can use it in the RAG application or other tasks. For example, you can use the `DenseRetriever` to retrieve the top 5 passages for a given query:

```python
from flexrag.retrievers import DenseRetriever, DenseRetrieverConfig
from flexrag.models import EncoderConfig, HFEncoderConfig

cfg = DenseRetrieverConfig(
    database_path='<path_to_database>',
    top_k=5,
    query_encoder_config=EncoderConfig(
        encoder_type='hf',
        hf_config=HFEncoderConfig(
            model_path='facebook/contriever',
            device_id=[0]
        )
    )
)
retriever = DenseRetriever(cfg)
passages = retriever.retrieve('What is the capital of France?')
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