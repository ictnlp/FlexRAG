import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from librarian.assistant import ASSISTANTS, AssistantBase, SearchHistory
from librarian.retriever import (
    RetrievedContext,
    WikipediaRetriever,
    WikipediaRetrieverConfig,
)
from librarian.prompt import ChatPrompt
from librarian.models import GENERATORS, GenerationConfig


logger = logging.getLogger("ReAct")
GeneratorConfig = GENERATORS.make_config()


@dataclass
class ReActConfig(GeneratorConfig, GenerationConfig, WikipediaRetrieverConfig):
    use_chat: bool = False
    max_steps: int = 8


@ASSISTANTS("react", config_class=ReActConfig)
class ReActAssistant(AssistantBase):
    is_hybrid = False

    def __init__(self, cfg: ReActConfig) -> None:
        # load retriever
        self.retriever = WikipediaRetriever(cfg)

        # load generator
        self.generator = GENERATORS.load(cfg)

        # load prompts
        self.chat_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__), "prompts", "react_qa_chat_prompt.json"
            )
        )
        self.gen_prompt = open(
            os.path.join(
                os.path.dirname(__file__), "prompts", "react_qa_gen_prompt.txt"
            ),
            "r",
            encoding="utf-8",
        ).read()

        # set configs
        self.use_chat = cfg.use_chat
        self.max_steps = cfg.max_steps
        self.cfg = cfg
        return

    def answer(
        self, question: str
    ) -> tuple[str, Optional[list[RetrievedContext]], Optional[dict]]:
        # retrieve
        history: list[SearchHistory] = []

        # perform react
        step = 1
        bad_steps = 0
        while step <= self.max_steps:
            # generate thought and action
            prompt = self.gen_prompt + question + "\n" + f"Thought {step}:"
            gen_cfg = deepcopy(self.cfg)
            gen_cfg.stop_str = f"\nObservation {step}:"
            thought_action = self.generator.generate(
                [prompt], generation_config=gen_cfg
            )[0][0]
            try:
                thought, action = thought_action.strip().split(f"\nAction {step}: ")
            except:
                bad_steps += 1
                step += 1
                thought = thought_action.strip().split("\n")[0]
                gen_cfg = deepcopy(self.cfg)
                gen_cfg.stop_str = "\n"
                action = self.generator.generate(
                    [prompt + f"Thought {step}: {thought}\nAction {step}:"],
                    generation_config=gen_cfg,
                )[0][0].strip()

            # take action
            if re.match(r"[sS]earch\[[^\]]+\]", action):
                entity = action[len("search[") : -1]
                ctx = self.retriever.search(query=[entity])[0][0]
                history.append(SearchHistory(query=question, contexts=[ctx]))
                if ctx.data["summary"] is None:
                    observation = (
                        f"Could not find [{entity}]. "
                        f"Similar entities: {ctx.data['similar_entities'][:5]}"
                    )
                else:
                    observation = ctx.data["summary"]
                lookup_keyword = None
            elif re.match(r"[lL]ookup\[[^\]]+\]", action):
                keyword = action[len("lookup[") : -1]
                if lookup_keyword != keyword:  # reset lookup
                    lookup_keyword = keyword
                    lookup_list = self._construct_lookup_list(
                        keyword, ctx.data["page_content"]
                    )
                    lookup_cnt = 0
                if lookup_cnt >= len(lookup_list):
                    observation = "No more results.\n"
                else:
                    observation = (
                        f"(Result {lookup_cnt + 1} / {len(lookup_list)}) "
                        + lookup_list[lookup_cnt]
                    )
                    lookup_cnt += 1
            elif re.match(r"[fF]inish\[[^\]]+\]", action):
                answer = action[len("finish[") : -1]
                break
            else:
                observation = "Invalid action: {}".format(action)

            # update prompt
            step_str = (
                f"Thought {step}: {thought}\n"
                f"Action {step}: {action}\n"
                f"Observation {step}: {observation}\n"
            )
            prompt += step_str
            step += 1
        return answer, history, None

    def _construct_lookup_list(self, keyword: str, page: str = None) -> list[str]:
        # find all paragraphs
        if page is None:
            return []
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences: list[str] = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts
