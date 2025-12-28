from abc import ABC, abstractmethod
from typing import Optional

from flexrag.common import LOGGER_MANAGER, Register, configure
from flexrag.common.dataclasses import RetrievedContext

logger = LOGGER_MANAGER.get_logger("flexrag.rankers")


@configure
class RankerBaseConfig:
    """The configuration for the ranker.

    :param reserve_num: the number of candidates to reserve.
        If it is less than 0, all candidates will be reserved. Default is -1.
    :type reserve_num: int
    :param ranking_field: the field name of the ranking field in the retrieved context.
        If it is None, the ranker will only accept a list of strings as candidates.
    :type ranking_field: Optional[str]
    """

    reserve_num: int = -1
    ranking_field: Optional[str] = None


@configure
class RankingResult:
    """The result of ranking.

    :param query: the query string. Required.
    :type query: str
    :param candidates: the ranked candidates.
        The results are sorted in descending order by relevance. Required.
    :type candidates: list[RetrievedContext | str]
    :param scores: the scores of the ranked candidates. Optional.
    :type scores: Optional[list[float]]
    """

    query: str
    candidates: list[RetrievedContext]
    scores: Optional[list[float]] = None


class RankerBase(ABC):
    """Base class for rankers.
    The ranker can rank candidates based on a query.
    The subclasses must implement the `rank` method.
    """

    def __init__(self, cfg: RankerBaseConfig) -> None:
        self.reserve_num = cfg.reserve_num
        self.ranking_field = cfg.ranking_field
        return

    @abstractmethod
    def rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> RankingResult:
        """Rank the candidates based on the query.

        :param query: query string.
        :type query: str
        :param candidates: list of candidate strings.
        :type candidates: list[RetrievedContext | str]
        :return: RankingResult containing ranked candidates and scores.
        :rtype: RankingResult
        """
        return

    async def async_rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        """The asynchronous version of `rank`."""
        logger.warning(
            "Current ranker does not support asynchronous rank,"
            "thus the code will be run in synchronous mode"
        )
        return self.rank(query, candidates)


RANKERS = Register[RankerBase]("ranker")
