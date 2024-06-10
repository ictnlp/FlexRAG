import json
import os
from argparse import ArgumentParser, Namespace
from functools import partial

import torch
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams

from retriever import add_args_for_retriever, load_retrievers
from utils import apply_template_llama3, apply_template_phi3


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
            "--rewrite",
            default=False,
            action="store_true",
            help="Whether to rewrite the query",
        )
        parser.add_argument(
            "--verify",
            default=False,
            action="store_true",
            help="Whether to verify the retrieved contexts",
        )
        parser.add_argument(
            "--summary",
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
        )
        self.searcher_sample_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.searcher_max_length,
        )
        self.searcher_tokenizer = self.searcher.get_tokenizer()
        self.searcher_prompt_func = self.get_prompt_func(
            self.searcher, self.searcher_tokenizer,
        )
        self.rewrite = args.rewrite
        self.verify = args.verify
        self.summary = args.summary

        # load generator
        if args.generator_path is None:
            self.generator = self.searcher
        else:
            self.generator = LLM(
                args.generator_path,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=torch.cuda.device_count(),
                max_model_len=4096,
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
        self.retriever = load_retrievers(args)[0]
        return

    def search(self, question: str) -> tuple[str, dict[str, str | list[str]]]:
        # Search contexts from the retriever
        contexts, query_to_search = self.gather_information([question])

        # Generate the answer based on the given information
        result = self.generate_with_context(question, contexts[0])
        search_track = {
            "question": question,
            "query_to_search": query_to_search[0],
            "contexts": contexts[0],
            "answer": result,
        }
        return result, search_track

    def gather_information(self, queries: list[str]) -> tuple[list[str], list[str]]:
        # Rewrite the query to be more informative
        if self.rewrite:
            basic_prompts = [
                (
                    "Your task is to write a query for a search engine "
                    "to find the answer to the following question. "
                    "Please reply your query only.\n\n"
                    f"Question: {q}"
                )
                for q in queries
            ]
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
        else:
            queries_to_search = queries

        # Search for the queries
        responses = self.retriever.search(queries_to_search)
        contexts = [i["texts"] for i in responses]
        return contexts, queries_to_search

    def verify_contexts(self, contexts: list[str], question: str) -> list[str]:
        ...

    def generate_with_context(self, question: str, contexts: list[str]) -> str:
        # Generate the answer
        basic_prompt = (
            "Your task is to generate an answer to the following question. "
            "Use the provided context to formulate your response. "
            "Note that the context might not always contain relevant information to answer the question. "
            "Please reply your answer only.\n\n"
        )
        for n, context in enumerate(contexts):
            basic_prompt += f"Context {n + 1}:\n{context}\n\n"
        basic_prompt += f"Question: {question}"
        prompt = self.generator_prompt_func([{"role": "user", "content": basic_prompt}])
        response = (
            self.generator.generate(
                prompt,
                sampling_params=self.generator_sample_params,
                use_tqdm=False,
            )[0]
            .outputs[0]
            .text
        )
        return response

    def get_prompt_func(self, model: LLM, tokenizer: PreTrainedTokenizer) -> callable:
        cfg_name = model.llm_engine.model_config.hf_config.__class__.__name__
        match cfg_name:
            case "Phi3Config":
                return partial(apply_template_phi3, tokenizer=tokenizer)
            case "Llama3Config":
                return partial(apply_template_llama3, tokenizer=tokenizer)
            case _:
                raise ValueError(f"Unsupported config: {cfg_name}")
        return

    def close(self) -> None:
        self.retriever.close()
        return
