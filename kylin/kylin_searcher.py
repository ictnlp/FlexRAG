from argparse import ArgumentParser, Namespace
from functools import partial

import torch
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams

from kylin.retriever import add_args_for_retriever, load_retrievers
from kylin.utils import apply_template_llama3, apply_template_phi3


class KylinLLMSearcher:
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--searcher_path",
            type=str,
            required=True,
            help="The path to the model",
        )
        parser.add_argument(
            "--generator_path",
            type=str,
            default=None,
            help="The path to the generator",
        )
        parser.add_argument(
            "--rewrite_query",
            default=False,
            action="store_true",
            help="Whether to rewrite the query",
        )
        parser.add_argument(
            "--verify_context",
            default=False,
            action="store_true",
            help="Whether to verify the retrieved contexts",
        )
        parser.add_argument(
            "--summary_context",
            default=False,
            action="store_true",
            help="Whether to summarize the retrieved contexts",
        )
        # parser.add_argument(
        #     "--searcher_device_ids",
        #     type=int,
        #     nargs="+",
        #     required=True,
        #     help="The device ids for the searcher",
        # )
        # parser.add_argument(
        #     "--generator_device_ids",
        #     type=int,
        #     nargs="+",
        #     required=True,
        #     help="The device ids for the generator",
        # )
        parser.add_argument(
            "--searcher_max_length",
            type=int,
            default=512,
            help="The maximum length for the searcher",
        )
        parser.add_argument(
            "--generator_max_length",
            type=int,
            default=300,
            help="The maximum length for the generator",
        )
        parser.add_argument(
            "--gpu_memory_utilization",
            type=float,
            default=0.4,
            help="The memory limit for the GPU",
        )
        parser = add_args_for_retriever(parser)
        return parser

    def __init__(self, args: Namespace) -> None:
        # load searcher
        self.searcher = LLM(
            args.searcher_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=16384,
            trust_remote_code=True,
        )
        self.searcher_sample_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.searcher_max_length,
        )
        self.searcher_tokenizer = self.searcher.get_tokenizer()
        self.searcher_prompt_func = self.get_prompt_func(
            self.searcher,
            self.searcher_tokenizer,
        )
        self.rewrite_query = args.rewrite_query
        self.verify_context = args.verify_context
        self.summary_context = args.summary_context

        # load generator
        if args.generator_path is None:
            self.generator = self.searcher
        else:
            self.generator = LLM(
                args.generator_path,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=4096,
                trust_remote_code=True,
            )
        self.generator_sample_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.generator_max_length,
        )
        self.generator_tokenizer = self.generator.get_tokenizer()
        self.generator_prompt_func = self.get_prompt_func(
            self.generator, self.generator_tokenizer
        )

        # load retriever
        self.retrievers = load_retrievers(args)
        return

    def answer(self, question: str) -> tuple[str, dict[str, str | list[str]]]:
        # search using kylin for the contexts
        contexts, search_track = self.search([question])

        # generate the answer using the generator
        result = self.generate_with_context(question, contexts[0])
        return result, search_track

    def search(
        self, questions: list[str]
    ) -> tuple[list[list[dict[str, str]]], list[dict[str, object]]]:
        # rewrite the query
        search_track = [{"question": q} for q in questions]
        if self.rewrite_query:
            query_to_search = self.rewrite_queries(questions)
            for n, q in enumerate(query_to_search):
                search_track[n]["rewritten_query"] = q
        else:
            query_to_search = questions

        # search contexts from the retriever
        contexts = self.search_for_contexts(query_to_search)
        for n, ctxs in enumerate(contexts):
            search_track[n]["retrieved_contexts"] = ctxs

        # verify the contexts
        if self.verify_context:
            verifications = self.verify_contexts(contexts, questions)
            for n, v in enumerate(verifications):
                search_track[n]["verify"] = v

        # summarize the contexts
        if self.summary_context:
            contexts = self.summarize_contexts(contexts, questions)
            for n, ctxs in enumerate(contexts):
                search_track[n]["summarized_contexts"] = [i["summary"] for i in ctxs]
            final_contexts = []
            for ctxs in contexts:
                final_contexts.append([i for i in ctxs if i["summary"] is not None])
        else:
            final_contexts = contexts

        return final_contexts, search_track

    def search_for_contexts(self, questions: list[str]) -> list[list[dict[str, str]]]:
        # Search for the queries
        contexts = [[] for _ in questions]
        for retriever in self.retrievers:
            ctxs = self.retrievers[retriever].search(questions)
            for n, ctx in enumerate(ctxs):
                if "indices" not in ctx:
                    assert "urls" in ctx
                    ctx["indices"] = ctx["urls"]
                retrieved_docs = [
                    {
                        "query": ctx["query"],
                        "retriever": retriever,
                        "text": text,
                        "source": idx,
                    }
                    for text, idx in zip(ctx["texts"], ctx["indices"])
                ]
                contexts[n].extend(retrieved_docs)
        return contexts

    def rewrite_queries(self, queries: list[str], engine_desc: str = "") -> list[str]:
        # Rewrite the query to be more informative
        sys_prompt = (
            "Your task is to write a query for a search engine "
            "to find the answer to the following question. "
            f"{engine_desc}"
            "Please only reply your query and do not output any other words.\n\n"
        )
        basic_prompts = [f"{sys_prompt}Question: {q}" for q in queries]
        prompts = [
            self.searcher_prompt_func(
                history=[
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    }
                ]
            )
            for prompt in basic_prompts
        ]
        queries_to_search = self.searcher.generate(
            prompts,
            sampling_params=self.searcher_sample_params,
            use_tqdm=False,
        )
        queries_to_search = [i.outputs[0].text for i in queries_to_search]
        return queries_to_search

    def verify_contexts(
        self, contexts: list[list[str]], question: list[str]
    ) -> list[bool]:
        prompts = []
        sys_prompt = (
            "Your task is to verify whether the following context "
            "contains enough information to answer the following question. "
            "Please only reply 'yes' or 'no' and do not output any other words.\n\n"
        )
        user_prompts = []
        for ctxs, q in zip(contexts, question):
            user_prompt = ""
            for n, ctx in enumerate(ctxs):
                user_prompt += f"Context {n}: {ctx['text']}\n\n"
            user_prompt += f"Question: {q}"
            user_prompts.append(user_prompt)
            history = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompts.append(self.searcher_prompt_func(history=history))
        responses = self.searcher.generate(
            prompts,
            sampling_params=self.searcher_sample_params,
            use_tqdm=False,
        )
        responses = [i.outputs[0].text for i in responses]
        return ["yes" in i.lower() for i in responses]

    def summarize_contexts(
        self, contexts: list[list[dict[str, str]]], questions: list[str]
    ) -> list[list[dict[str, str]]]:
        sys_prompt = (
            "Please copy the sentences from the following context that are relevant to the given question. "
            "If the context does not contain any relevant information, respond with <none>.\n\n"
        )
        for q, ctxs in zip(questions, contexts):
            user_prompts = []
            for ctx in ctxs:
                user_prompts.append(f"Context: {ctx['text']}\n\nQuestion: {q}")
            prompts = [
                self.searcher_prompt_func(
                    history=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
                for prompt in user_prompts
            ]
            responses = self.searcher.generate(
                prompts,
                sampling_params=self.searcher_sample_params,
                use_tqdm=False,
            )
            responses = [i.outputs[0].text for i in responses]
            for summ, ctx in zip(responses, ctxs):
                if "<none>" in summ.lower():
                    ctx["summary"] = None
                else:
                    ctx["summary"] = summ
        return contexts

    def generate_with_context(self, question: str, contexts: list[str]) -> str:
        # Generate the answer
        sys_prompt = (
            "\n\nAnswer the question based on the given contexts. "
            "Note that the context might not always contain relevant information to answer the question. "
            "Only give me the answer and do not output any other words.\n\n"
        )
        for n, context in enumerate(contexts):
            if "summary" in context:
                ctx = context["summary"]
            else:
                ctx = context["text"]
            sys_prompt += f"Context {n + 1}: {ctx}\n\n"

        history = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"\n\n{question}"},
        ]
        prompt = self.generator_prompt_func(history)
        response = self.generator.generate(
            prompt,
            sampling_params=self.generator_sample_params,
            use_tqdm=False,
        )
        return response[0].outputs[0].text

    def get_prompt_func(self, model: LLM, tokenizer: PreTrainedTokenizer) -> callable:
        cfg_name = model.llm_engine.model_config.hf_config.__class__.__name__
        match cfg_name:
            case "Phi3Config":
                return partial(apply_template_phi3, tokenizer=tokenizer)
            case "Phi3SmallConfig":
                return partial(apply_template_phi3, tokenizer=tokenizer)
            case "Llama3Config":
                return partial(apply_template_llama3, tokenizer=tokenizer)
            case _:
                raise ValueError(f"Unsupported config: {cfg_name}")
        return

    def close(self) -> None:
        self.retriever.close()
        return
