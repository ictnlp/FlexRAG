# Quickstart
This quickstart guide will help you to run a simple RAG assistant with FlexRAG. FlexRAG provides multiple ways to develop and evaluate your RAG assistant, including configure your own `ModularAssistant` and running a GUI application using FlexRAG' entrypoints, or developing your own RAG assistant by import FlexRAG as a library.

## Running `ModularAssistant` via FlexRAG's entrypoints
FlexRAG provides several built-in RAG assistants, including `BasicAssistant`, `ModularAssistant`, `ChatQAAssistant`, .etc. You can run these assistants with FlexRAG's entrypoints. In this guide, we will show you how to run the `ModularAssistant`,  as it offers a wide range of configuration options.

The basic structure of the `ModularAssistant` is as follows:

```{eval-rst}
.. image:: ../../../assets/ModularAssistant.png
   :alt: ModularAssistant
   :align: center
   :width: 50%
```

The `ModularAssistant` is composed of four key components: a retriever, a reranker, a context refiner, and a generator.
- The **retriever** fetches relevant passages from the knowledge base.
- The **reranker** reorders the retrieved passages for better relevance, which is optional.
- The **context refiner** optimizes the context for the generator, which is optional.
- The **generator** creates the final response based on the refined context.

Each component can be configured independently, allowing you to easily customize your RAG assistant by adjusting the settings for each one.


### Running the GUI application
The easiest way to run a RAG assistant is to use FlexRAG's entrypoints to start a GUI application. You can run the following command to start a GUI application with the `ModularAssistant` and the `BM25SRetriever` as the retriever, and the `OpenAIGenerator` as the generator:

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

Then you can visit the GUI application at `http://localhost:7860` in your browser. You will see a simple interface where you can input your question and get the response from the RAG assistant.

```{eval-rst}
.. image:: ../../../assets/gui_static.png
   :alt: GUI
   :align: center
   :width: 80%
```


### Running the RAG evaluation
You can easily evaluate your RAG assistant on a variety of knowledge-intensive tasks. The following command let you evaluate the `ModularAssistant` with the `BM25SRetriever` on the Natural Questions (NQ) dataset:
```bash
OUTPUT_PATH=<path_to_output>
DB_PATH=<path_to_database>
OPENAI_KEY=<your_openai_key>

python -m flexrag.entrypoints.run_assistant \
    name=nq \
    split=test \
    output_path=${OUTPUT_PATH} \
    assistant_type=modular \
    modular_config.retriever_type="FlexRAG/wiki2021_atlas_bm25s" \
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

For more information about the supported tasks, please refer to the {any}`RAGEvalDatasetConfig` section.

### Developing your own RAG assistant
You can also develop your own RAG assistant by inherit the `AssistantBase` class and registering it with the `ASSISTANTS` decorator. Then you are able to run your own RAG assistant using FlexRAG's entrypoints by adding the `user_module=<your_module_path>` argument to the command.
You can find more information in the [Developing your own RAG assistant](../tutorial/building_assistant.md) tutorial.

## Running RAG applications by importing FlexRAG as a library
Besides running RAG assistants with FlexRAG's entrypoints, you can also import FlexRAG as a library to develop your own RAG applications. FlexRAG provides a flexible and modular API that allows you to customize your RAG application with ease. For example, you can use the following code to build a simple RAG QA system:

```python
from flexrag.models import OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.retriever import LocalRetriever


def main():
    # load the retriever
    retriever = LocalRetriever.load_from_hub("FlexRAG/wiki2021_atlas_bm25s")

    # load the generator
    generator = OpenAIGenerator(
        OpenAIGeneratorConfig(
            model_name="Qwen2-7B-Instruct",
            base_url="http://10.28.0.148:8000/v1",
        )
    )

    # build a QA loop
    while True:
        query = input("Please input your question (type /bye to quit): ")
        if query == "/bye":
            break
        # retrieve the contexts
        contexts = retriever.search(query, top_k=3)[0]
        # construct the prompt
        user_prompt = (
            "Please answer the following question based on the given contexts.\n"
            f"Question: {query}\n"
        )
        for i, ctx in enumerate(contexts):
            user_prompt += f"Context {i+1}: {ctx.data['text']}\n"
        # generate the response
        response = generator.chat([{"role": "user", "content": user_prompt}])[0][0]
        print(response)

    return


if __name__ == "__main__":
    main()
```