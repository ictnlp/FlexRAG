import copy
import os
import re
from dataclasses import dataclass

import numpy as np

from flexrag.models import GENERATORS, GeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.utils import TIME_METER

from .ranker import RANKERS, RankerBase, RankerBaseConfig


@dataclass
class RankGPTRankerConfig(RankerBaseConfig, GeneratorConfig):
    """The configuration for the RankGPT ranker.

    :param step_size: the step size for the slide window ranking. Default is 10.
    :type step_size: int
    :param window_size: the window size for the slide window ranking. Default is 20.
    :type window_size: int
    :param max_chunk_size: the maximum chunk size for the slide window ranking. Default is 300.
    :type max_chunk_size: int
    """

    step_size: int = 10
    window_size: int = 20
    max_chunk_size: int = 300


@RANKERS("rank_gpt", config_class=RankGPTRankerConfig)
class RankGPTRanker(RankerBase):
    """RankGPTRanker:
    Rank the candidates based on the query using the Large Language model.
    Code was adapted from the original implementation from https://github.com/sunnweiwei/RankGPT
    """

    def __init__(self, cfg: RankGPTRankerConfig):
        super().__init__(cfg)
        self.generator = GENERATORS.load(cfg)

        # load prompt
        prompt_path = os.path.join(
            os.path.dirname(__file__), "ranker_prompts", "rankgpt_prompt.json"
        )
        self.prompt = ChatPrompt.from_json(prompt_path)

        # set basic arguments
        self.step_size = cfg.step_size
        self.window_size = cfg.window_size
        self.max_chunk_size = cfg.max_chunk_size
        return

    @TIME_METER("rankgpt_rank")
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, None]:
        # perform slide window ranking
        indices = list(range(len(candidates)))
        start_idx = max(len(candidates) - self.window_size, 0)
        end_idx = len(candidates)
        while start_idx >= 0:
            start_idx = max(start_idx, 0)
            candidates_ = [candidates[i] for i in indices[start_idx:end_idx]]
            indices_, _ = self._rank_piece(query, candidates_)
            indices[start_idx:end_idx] = indices_
            start_idx = start_idx - self.step_size
            end_idx -= self.step_size
        return np.array(indices), None

    @TIME_METER("rankgpt_rank")
    async def _async_rank(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, None]:
        # perform slide window ranking
        indices = list(range(len(candidates)))
        start_idx = max(len(candidates) - self.window_size, 0)
        end_idx = len(candidates)
        while start_idx >= 0:
            start_idx = max(start_idx, 0)
            candidates_ = [candidates[i] for i in indices[start_idx:end_idx]]
            indices_, _ = await self._async_rank_piece(query, candidates_)
            indices[start_idx:end_idx] = indices_
            start_idx = start_idx - self.step_size
            end_idx -= self.step_size
        return np.array(indices), None

    def _rank_piece(self, query: str, candidates: list[str]) -> tuple[np.ndarray, None]:
        prompt = self._get_prompt(query=query, candidates=candidates)
        response = self.generator.chat(prompts=[prompt])[0][0]

        # convert string to indices
        response = re.sub(r"\D", " ", response)
        indices_ = [int(x) - 1 for x in response.split()]

        # deduplicate indices
        indices = []
        for i in indices_:
            if i not in indices:
                indices.append(i)

        # refine indices
        ori_indices = list(range(len(candidates)))
        new_indices = [idx for idx in indices if idx in ori_indices]
        new_indices = new_indices + [
            idx for idx in ori_indices if idx not in new_indices
        ]
        return new_indices, None

    async def _async_rank_piece(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, None]:
        prompt = self._get_prompt(query=query, candidates=candidates)
        response = (await self.generator.async_chat(prompts=[prompt]))[0][0]

        # convert string to indices
        response = re.sub(r"\D", " ", response)
        indices_ = [int(x) - 1 for x in response.split()]

        # deduplicate indices
        indices = []
        for i in indices_:
            if i not in indices:
                indices.append(i)

        # refine indices
        ori_indices = list(range(len(candidates)))
        new_indices = [idx for idx in indices if idx in ori_indices]
        new_indices = new_indices + [
            idx for idx in ori_indices if idx not in new_indices
        ]
        return new_indices, None

    def _get_prompt(self, query: str, candidates: list[str]):
        max_length = 300
        prompt = copy.deepcopy(self.prompt)
        prompt.history[0].content = prompt.history[0].content.format(
            query=query, num=len(candidates)
        )
        last_turn = prompt.history.pop()
        last_turn.content = last_turn.content.format(query=query, num=len(candidates))

        rank = 0
        for cand in candidates:
            rank += 1
            content = cand.replace("Title: Content: ", "")
            content = content.strip()
            content = " ".join(content.split()[: int(max_length)])
            prompt.update(ChatTurn(role="user", content=f"[{rank}] {content}"))
            prompt.update(
                ChatTurn(role="assistant", content=f"Received passage [{rank}].")
            )
        prompt.update(last_turn)
        return prompt
