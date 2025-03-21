# Quickstart: Building your own RAG application
Besides using RAG assistant, you can also import FlexRAG as a library to develop your own RAG applications. FlexRAG provides a flexible and modular API that allows you to customize your RAG application with ease. For example, you can use the following code to build a simple command line RAG QA system:

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