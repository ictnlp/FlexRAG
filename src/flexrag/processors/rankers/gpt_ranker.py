import copy
import re
from pathlib import Path

from flexrag.common import configure, trace
from flexrag.common.dataclasses import ChatMessages, ChatTurn, RetrievedContext
from flexrag.models import GENERATORS, GeneratorConfig

from .ranker_base import RANKERS, RankerBase, RankerBaseConfig, RankingResult


@configure
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
        prompt_path = Path(__file__).parent / "ranker_prompts" / "rankgpt_prompt.json"
        self.prompt = ChatMessages.from_json(prompt_path)

        # set basic arguments
        self.step_size = cfg.step_size
        self.window_size = cfg.window_size
        self.max_chunk_size = cfg.max_chunk_size
        return

    @trace("ranker.rankgpt")
    def rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> RankingResult:
        # perform slide window ranking
        if isinstance(candidates[0], RetrievedContext):
            assert self.ranking_field is not None, (
                "ranking_field must be specified when ranking RetrievedContext"
            )
        cands = [
            (
                cand.data[self.ranking_field]
                if isinstance(cand, RetrievedContext)
                else cand
            )
            for cand in candidates
        ]
        indices = list(range(len(cands)))
        start_idx = max(len(cands) - self.window_size, 0)
        end_idx = len(cands)
        while start_idx >= 0:
            start_idx = max(start_idx, 0)
            current_indices = indices[start_idx:end_idx]
            cands_ = [cands[i] for i in current_indices]
            local_indices = self._rank_piece(query, cands_)
            indices[start_idx:end_idx] = [current_indices[i] for i in local_indices]
            if start_idx == 0:
                break
            start_idx = start_idx - self.step_size
            end_idx -= self.step_size
        if self.reserve_num > 0:
            indices = indices[: self.reserve_num]
        return RankingResult(
            query=query,
            candidates=[candidates[i] for i in indices],
            scores=None,
        )

    def _rank_piece(self, query: str, candidates: list[str]) -> list[int]:
        prompt = self._get_prompt(query=query, candidates=candidates)
        response = self.generator.chat(messages=[prompt])[0][0].text_content

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
        return new_indices

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
            prompt.append(ChatTurn(role="user", content=f"[{rank}] {content}"))
            prompt.append(
                ChatTurn(role="assistant", content=f"Received passage [{rank}].")
            )
        prompt.append(last_turn)
        return prompt
