from abc import ABC, abstractmethod

import numpy as np

from flexrag.common import LOGGER_MANAGER, Register

logger = LOGGER_MANAGER.get_logger("flexrag.models.scorers")


class PairScorerBase(ABC):
    """Base class for pair scorers.

    The pair scorer can score pairs of texts.
    The subclasses must implement the `score` method.
    """

    @abstractmethod
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score the given pairs.

        :param pairs: A batch of text pairs.
        :type pairs: list[tuple[str, str]]
        :return: A batch of scores.
        :rtype: np.ndarray
        """
        return

    async def async_score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score the given pairs asynchronously.

        :param pairs: A batch of text pairs.
        :type pairs: list[tuple[str, str]]
        :return: A batch of scores.
        :rtype: np.ndarray
        """
        logger.warning(
            "Current scorer does not support asynchronous scoring, "
            "thus the code will be run in synchronous mode"
        )
        return self.score(pairs)

    def score_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Score the given pairs in batches.

        :param pairs: A batch of text pairs.
        :type pairs: list[tuple[str, str]]
        :param batch_size: The size of each batch. Defaults to 32.
        :type batch_size: int
        :return: A batch of scores.
        :rtype: np.ndarray
        """
        all_scores: list[float] = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            batch_scores = self.score(batch_pairs)
            all_scores.append(batch_scores)
        return np.concatenate(all_scores, axis=0)


SCORERS = Register[PairScorerBase]("scorer")
