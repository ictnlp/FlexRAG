# Preparing the Knowledge Base
In the real world, various types of knowledge are typically stored in documents such as PDFs, Word files, and PPTs. However, this semi-structured data cannot be parsed by large language models (LLMs) and is not suitable for building a knowledge base. Therefore, we need to convert it into structured text data beforehand. In this tutorial, we will use a simple example to demonstrate how to convert a batch of PDF files into structured data.

```{tip}
If you already have structured data, you can skip this tutorial.
```

## Parse Files using FlexRAG's Command-Line Tool
FlexRAG provides a command-line tool `prepare_corpus` to help users parse various files into structured data. In this tutorial, we will use a paper from Arxiv as an example to demonstrate how to parse a PDF file using the built-in command-line tool of FlexRAG.

Run the following command to download a paper from Arxiv:

```bash
wget https://arxiv.org/pdf/2502.18139.pdf
```

You can then run the following command to parse this paper into structured knowledge base data:

```bash
python -m flexrag.entrypoints.prepare_corpus \
    document_paths=[2502.18139.pdf] \
    output_path=knowledge.jsonl \
    document_parser_type=markitdown \
    chunker_type=sentence_chunker \
    sentence_chunker_config.max_tokens=512 \
    sentence_chunker_config.tokenizer_type=tiktoken \
    sentence_chunker_config.tiktoken_config.model_name='gpt-4o'
```

In this command, we specify the following parameters:
- `document_paths`：a list of file paths to be parsed. Here we only parse one paper;
- `output_path`：the output path of the parsed results. The path should end with `.jsonl`, `.csv`, or `.tsv`;
- `document_parser_type`：the type of document parser. Here we use `markitdown`;
- `chunker_type`：the type of text chunker. Here we use `sentence_chunker`;
- `sentence_chunker_config.max_tokens`：the maximum length of the text chunker. Here we set it to 512;
- `sentence_chunker_config.tokenizer_type`：the type of tokenizer used by the text chunker. Here we use `tiktoken`, which is provided by OpenAI;
- `sentence_chunker_config.tiktoken_config.model_name`：the model name used by the tokenizer. Here we use `gpt-4o`.

After executing the above command, you will see that the PDF file has been parsed into a JSONL file. As shown in the figure below, FlexRAG executed three steps in this process:
1. **Parsing**: parsing the file into structured data;
2. **Chunking**: chunking long text paragraphs in the structured data into short text paragraphs suitable for processing;
3. **Preprocessing**: preprocessing and filtering the chunked text paragraphs.

```{eval-rst}
.. image:: ../../../assets/parse_files.png
   :alt: Parse File
   :align: center
   :width: 80%
```

```{tip}
You can check the [FlexRAG Entrypoints](./entrypoints.md) documentation for more information about the `prepare_corpus` command.
```

```{tip}
You can check the [Preparing the Retriever](./preparing_retriever.md) documentation for how to build a retriever for your knowledge base.
```
