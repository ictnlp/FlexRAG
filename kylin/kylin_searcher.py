import logging
import re
from argparse import ArgumentParser, Namespace

import torch

from kylin.retriever import add_args_for_retriever, load_retrievers

from .kylin_prompts import *
from .models.vllm_model import load_vllm
from .utils import SimpleProgressLogger


logger = logging.getLogger(__name__)


class KylinLLMSearcher:
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--searcher_path",
            type=str,
            required=True,
            help="The path to the kylin searcher model.",
        )
        parser.add_argument(
            "--generator_path",
            type=str,
            default=None,
            help="The path to the generator.",
        )
        parser.add_argument(
            "--extract_information",
            default=False,
            action="store_true",
            help="Whether to extract needed information from the user query.",
        )
        parser.add_argument(
            "--rewrite_query",
            default=False,
            action="store_true",
            help="Whether to rewrite the query.",
        )
        parser.add_argument(
            "--verify_context",
            default=False,
            action="store_true",
            help="Whether to verify the retrieved contexts.",
        )
        parser.add_argument(
            "--summary_context",
            default=False,
            action="store_true",
            help="Whether to summarize the retrieved contexts.",
        )
        parser.add_argument(
            "--searcher_max_length",
            type=int,
            default=512,
            help="The maximum length for the searcher.",
        )
        parser.add_argument(
            "--generator_max_length",
            type=int,
            default=300,
            help="The maximum length for the generator.",
        )
        parser.add_argument(
            "--gpu_memory_utilization",
            type=float,
            default=0.4,
            help="The memory limit for the GPU.",
        )
        parser = add_args_for_retriever(parser)
        return parser

    def __init__(self, args: Namespace) -> None:
        # load searcher
        self.extract_info = args.extract_information
        self.rewrite = args.rewrite_query
        self.verify = args.verify_context
        self.summary_context = args.summary_context
        self.log_interval = args.log_interval
        (
            self.searcher,
            self.searcher_tokenizer,
            self.searcher_sample_params,
            self.searcher_prompt_func,
        ) = load_vllm(
            model_path=args.searcher_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=torch.cuda.device_count(),
            max_gen_tokens=args.searcher_max_length,
        )

        # load generator
        if args.generator_path is None:
            self.generator = self.searcher
            self.generator_tokenizer = self.searcher_tokenizer
            self.generator_sample_params = self.searcher_sample_params
            self.generator_prompt_func = self.searcher_prompt_func
        else:
            (
                self.generator,
                self.generator_tokenizer,
                self.generator_sample_params,
                self.generator_prompt_func,
            ) = load_vllm(
                model_path=args.generator_path,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=torch.cuda.device_count(),
                max_gen_tokens=args.generator_max_length,
            )

        # load retriever
        self.retrievers = load_retrievers(args)
        return

    def answer(self, questions: list[str]) -> tuple[list[str], list[dict[str, object]]]:
        responses = []
        search_tracks = []
        progress = SimpleProgressLogger(logger, len(questions), self.log_interval)
        for question in questions:
            progress.update()
            response, search_track = self._answer(question)
            responses.append(response)
            search_tracks.append(search_track)
        return responses, search_tracks

    def _answer(self, question: str) -> tuple[str, dict[str, object]]:
        # search contexts using kylin searcher
        contexts, search_track = self.search(question)

        # generate the answer using the generator
        result, prompt = self.generate_with_context(question, contexts)
        search_track["generation_prompt"] = prompt
        return result, search_track

    def search(self, question: str) -> tuple[list[dict[str, str]], dict[str, object]]:
        # extract needed information
        search_track = {"question": question}
        if self.extract_info:
            info_needed, response = self.determine_needed_information(question)
        else:
            info_needed = [question]
        search_track["needed_information"] = info_needed

        # search contexts from the retrievers
        contexts = []
        contexts_history = []
        queries_history = []
        for q in info_needed:
            ctx, q_history, ctx_history = self.inner_search(q)
            contexts.append(ctx)
            queries_history.append(q_history)
            contexts_history.append(ctx_history)
        search_track["retrieved_contexts"] = contexts
        search_track["queries_history"] = queries_history
        search_track["contexts_history"] = contexts_history

        # summarize the contexts
        if self.summary_context:
            final_contexts = []
            for ctx, info in zip(contexts, info_needed):
                final_contexts.append(self.summarize_contexts(ctx, info))
            final_contexts = [i for j in final_contexts for i in j]
            search_track["summarized_contexts"] = [i["summary"] for i in final_contexts]
            final_contexts = [i for i in final_contexts if i["summary"] is not None]
        else:
            final_contexts = [i for j in contexts for i in j]

        return final_contexts, search_track

    def inner_search(
        self, query: str
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], list[list[dict[str, str]]]]:
        """Search from multiple retrievers

        Args:
            query (str): query to search

        Returns:
            contexts (list[dict[str, str]]):
            queries_history (list[dict[str, str]]]): query used for different retrievers.
            contexts_history (dict[str, str])
        """
        total_turn_num = 3 if self.verify else 1
        contexts_history = []
        queries_history = []
        turn_num = 0
        while turn_num < total_turn_num:
            contexts = []
            queries_history.append({})
            for retriever in self.retrievers:
                # rewrite query
                if self.rewrite:
                    failing_queries = [i[retriever] for i in queries_history[:-1]]
                    query_to_search = self.rewrite_query(
                        info=query,
                        engine_desc=self.retrievers[retriever].search_hint,
                        engine_name=retriever,
                        failing_queries=failing_queries,
                    )
                else:
                    query_to_search = query
                queries_history[turn_num][retriever] = query_to_search
                # retrieve
                ctxs = self.retrievers[retriever].search([query_to_search])[0]
                # post process
                if "indices" not in ctxs:
                    assert "urls" in ctxs
                    ctxs["indices"] = ctxs["urls"]
                for text, idx in zip(ctxs["texts"], ctxs["indices"]):
                    contexts.append(
                        {
                            "query": ctxs["query"],
                            "retriever": retriever,
                            "text": text,
                            "source": idx,
                        }
                    )
            # verify
            contexts_history.append(contexts)
            verification = self.verify_contexts(contexts, query)
            if verification:
                break
        return contexts, queries_history, contexts_history

    def determine_needed_information(
        self, query: str, given_contexts: list[str] = None
    ) -> tuple[list[str], str]:
        user_prompt = self.searcher_prompt_func(
            history=[
                {"role": "system", "content": determine_info_prompt},
                {"role": "user", "content": f"Question: {query}"},
            ]
        )
        response = self.searcher.generate(
            [user_prompt],
            sampling_params=self.searcher_sample_params,
            use_tqdm=False,
        )
        response = response[0].outputs[0].text
        info_needed = re.findall(r"\[\d+\] (.+?)(?=\[\d+\]|\Z)", response)
        return info_needed, response

    def rewrite_query(
        self,
        info: str,
        engine_name: str,
        engine_desc: str = "",
        failing_queries: list[str] = None,
    ) -> str:
        # Rewrite the query to be more informative
        sys_prompt = rewrite_prompt.format(
            engine_name=engine_name, engine_desc=engine_desc
        )
        user_prompt = f"Information: {info}"
        if (failing_queries is not None) and (len(failing_queries) > 0):
            user_prompt += (
                "\nHere are some queries that failed to retrieve the information, "
                "please avoid using them."
            )
            for q in failing_queries:
                user_prompt += f"\nFailing Query: {q}"
        prompt = self.searcher_prompt_func(
            history=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        query_to_search = self.searcher.generate(
            [prompt],
            sampling_params=self.searcher_sample_params,
            use_tqdm=False,
        )
        query_to_search = query_to_search[0].outputs[0].text
        return query_to_search

    def verify_contexts(
        self, contexts: list[dict[str, str]], question: str
    ) -> list[bool]:
        user_prompt = ""
        for n, ctx in enumerate(contexts):
            user_prompt += f"Context {n}: {ctx['text']}\n\n"
        user_prompt += f"Topic: {question}"
        history = [
            {"role": "system", "content": verify_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.searcher_prompt_func(history=history)
        response = self.searcher.generate(
            [prompt],
            sampling_params=self.searcher_sample_params,
            use_tqdm=False,
        )
        response = response[0].outputs[0].text
        return "yes" in response.lower()

    def summarize_contexts(
        self, contexts: list[dict[str, str]], question: str
    ) -> list[dict[str, str]]:
        user_prompts = []
        for ctx in contexts:
            user_prompts.append(f"Context: {ctx['text']}\n\nQuestion: {question}")
        prompts = [
            self.searcher_prompt_func(
                history=[
                    {"role": "system", "content": summary_prompt},
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
        for summ, ctx in zip(responses, contexts):
            if "<none>" in summ.lower():
                ctx["summary"] = None
            else:
                ctx["summary"] = summ
        return contexts

    def generate_with_context(
        self, question: str, contexts: list[str]
    ) -> tuple[str, str]:
        # Generate the answer
        usr_prompt = ""
        for n, context in enumerate(contexts):
            if "summary" in context:
                ctx = context["summary"]
            else:
                ctx = context["text"]
            usr_prompt += f"Context {n + 1}: {ctx}\n\n"
        usr_prompt += f"Question: {question}"

        history = [
            {"role": "system", "content": generate_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        prompt = self.generator_prompt_func(history)
        response = self.generator.generate(
            prompt,
            sampling_params=self.generator_sample_params,
            use_tqdm=False,
        )
        return response[0].outputs[0].text, prompt

    def close(self) -> None:
        for retriever in self.retrievers:
            self.retrievers[retriever].close()
        return
