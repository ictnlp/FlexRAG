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
    def score(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        """Score the given pairs.

        :param pairs: A batch of text pairs.
        :type pairs: list[tuple[str, str]]
        :param batch_size: The request batch size. Defaults to 32.
        :type batch_size: int
        :param log_interval: The logging interval for progress updates.
            Defaults to 1000.
        :type log_interval: int
        :return: A batch of scores.
        :rtype: np.ndarray
        """
        return

    async def async_score(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        """Score the given pairs asynchronously.

        :param pairs: A batch of text pairs.
        :type pairs: list[tuple[str, str]]
        :param batch_size: The request batch size. Defaults to 32.
        :type batch_size: int
        :param log_interval: The logging interval for progress updates.
            Defaults to 1000.
        :type log_interval: int
        :return: A batch of scores.
        :rtype: np.ndarray
        """
        logger.warning(
            "Current scorer does not support asynchronous scoring, "
            "thus the code will be run in synchronous mode"
        )
        del batch_size, log_interval
        return self.score(pairs)


SCORERS = Register[PairScorerBase]("scorer")
