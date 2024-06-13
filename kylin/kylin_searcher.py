from argparse import ArgumentParser, Namespace
import re

import torch

from kylin.retriever import add_args_for_retriever, load_retrievers
from .load_vllm import load_vllm


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
        self.rewrite = args.rewrite_query
        self.verify_context = args.verify_context
        self.summary_context = args.summary_context
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

    def answer(self, question: str) -> tuple[str, dict[str, str | list[str]]]:
        # search using kylin for the contexts
        contexts, search_track = self.search(question)

        # generate the answer using the generator
        result = self.generate_with_context(question, contexts)
        return result, search_track

    def search(self, question: str) -> tuple[list[dict[str, str]], dict[str, object]]:
        # rewrite the query
        search_track = {"question": question}
        if self.rewrite:
            info_needed, response = self.determine_needed_information(question)
            queries_to_search = []
            for n, info in enumerate(info_needed):
                queries_to_search.append(self.rewrite_query(info))
            search_track["needed_information"] = info_needed
            search_track["queries_to_search"] = queries_to_search
        else:
            queries_to_search = [question]
            info_needed = [question]
            search_track["queries_to_search"] = [question]

        # search contexts from the retriever
        contexts = []
        for q in queries_to_search:
            contexts.append(self.search_for_contexts(q))
        search_track["retrieved_contexts"] = contexts

        # iterate to refine the query
        if self.verify_context:
            for n, (info, query) in enumerate(zip(info_needed, queries_to_search)):
                ctx = contexts[n]
                verification = self.verify_contexts(ctx, info)
                turn_num = 0
                q_history = [query]
                v_history = [verification]
                while not verification:
                    q = self.rewrite_query(info, failing_queries=q_history)
                    ctx = self.search_for_contexts(q)
                    verification = self.verify_contexts(ctx, info)
                    q_history.append(query)
                    v_history.append(verification)
                    turn_num += 1
                    if turn_num > 3:
                        break
                contexts[n] = ctx
                search_track["verify_history"] = v_history
                search_track["query_history"] = q_history
        search_track["verified_contexts"] = contexts
        contexts = [i for j in contexts for i in j]

        # summarize the contexts
        if self.summary_context:
            contexts = self.summarize_contexts(contexts, question)
            search_track["summarized_contexts"] = [i["summary"] for i in contexts]
            final_contexts = [i for i in contexts if i["summary"] is not None]
        else:
            final_contexts = contexts

        return final_contexts, search_track

    def search_for_contexts(self, question: str) -> list[dict[str, str]]:
        # Search from multi retrievers
        contexts = []
        for retriever in self.retrievers:
            ctxs = self.retrievers[retriever].search([question])[0]
            if "indices" not in ctxs:
                assert "urls" in ctxs
                ctxs["indices"] = ctxs["urls"]
            retrieved_docs = [
                {
                    "query": ctxs["query"],
                    "retriever": retriever,
                    "text": text,
                    "source": idx,
                }
                for text, idx in zip(ctxs["texts"], ctxs["indices"])
            ]
            contexts.extend(retrieved_docs)
        return contexts

    def determine_needed_information(self, query: str) -> tuple[list[str], str]:
        sys_prompt = (
            "Please indicate the external knowledge needed to answer the following question. "
            "If the question can be answered without external knowledge, "
            'answer "No additional information is required".\n\n'
            "For example:\n"
            "Question: who got the first nobel prize in physics.\n"
            "Needed Information: [1] The first physics Nobel Laureate.\n\n"
            "Question: big little lies season 2 how many episodes\n"
            "Needed Information: [1] The total number of episodes in the first season.\n\n"
            "Question: who sings i can't stop this feeling anymore?\n"
            'Needed Information: [1] The singer of "i can\'t stop this feeling anymore".\n\n'
            "Question: what is 1 radian in terms of pi?\n"
            "No additional information is required.\n\n"
            "Question: Which magazine was started first Arthur's Magazine or First for Women?\n"
            "Needed Information: [1] The time when Arthur's Magazine was founded.\n"
            "[2] The time when First for Women was founded.\n\n"
        )
        user_prompt = self.searcher_prompt_func(
            history=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Question: {query}"},
            ]
        )
        response = self.searcher.generate(
            [user_prompt],
            sampling_params=self.searcher_sample_params,
            use_tqdm=False,
        )
        response = response[0].outputs[0].text
        info_needed = re.findall(r"\[\d+\] (.+)", response)
        return info_needed, response

    def rewrite_query(
        self,
        info: str,
        engine_desc: str = "",
        failing_queries: list[str] = None,
    ) -> str:
        # Rewrite the query to be more informative
        sys_prompt = (
            "Please compose a text that will be used to query the following information from a search engine. "
            f"{engine_desc} "
            "Please only reply your query and do not output any other words."
        )
        user_prompt = f"Information: {info}"
        if failing_queries is not None:
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
        queries_to_search = self.searcher.generate(
            [prompt],
            sampling_params=self.searcher_sample_params,
            use_tqdm=False,
        )
        queries_to_search = queries_to_search[0].outputs[0].text
        return queries_to_search

    def verify_contexts(
        self, contexts: list[dict[str, str]], question: str
    ) -> list[bool]:
        sys_prompt = (
            "Your task is to verify whether the following context "
            "contains enough information to understand the following topic. "
            "Please only reply 'yes' or 'no' and do not output any other words."
        )
        user_prompt = ""
        for n, ctx in enumerate(contexts):
            user_prompt += f"Context {n}: {ctx['text']}\n\n"
        user_prompt += f"Topic: {question}"
        history = [
            {"role": "system", "content": sys_prompt},
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
        sys_prompt = (
            "Please copy the sentences from the following context that are relevant to the given question. "
            "If the context does not contain any relevant information, respond with <none>.\n\n"
        )
        user_prompts = []
        for ctx in contexts:
            user_prompts.append(f"Context: {ctx['text']}\n\nQuestion: {question}")
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
        for summ, ctx in zip(responses, contexts):
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

    def close(self) -> None:
        for retriever in self.retrievers:
            retriever.close()
        return
