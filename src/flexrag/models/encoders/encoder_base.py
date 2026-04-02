from abc import ABC, abstractmethod

import numpy as np

from flexrag.common import LOGGER_MANAGER, Register, SimpleProgressLogger

logger = LOGGER_MANAGER.get_logger("flexrag.models.encoder")


class EncoderBase(ABC):
    """Base class for encoders.
    The encoder can encode texts into embeddings.
    The subclasses must implement the `encode` and `embedding_size` methods.
    """

    @abstractmethod
    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        """Encode the given texts into embeddings.

        :param texts: A batch of texts.
        :type texts: list[str] | str
        :param batch_size: The request batch size. Defaults to 32.
        :type batch_size: int
        :param log_interval: The logging interval for progress updates.
            Defaults to 1000.
        :type log_interval: int
        :return: A batch of embeddings.
        :rtype: np.ndarray
        """
        return

    async def async_encode(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        """Encode the given texts into embeddings.

        :param texts: A batch of texts.
        :type texts: list[str] | str
        :param batch_size: The request batch size. Defaults to 32.
        :type batch_size: int
        :param log_interval: The logging interval for progress updates.
            Defaults to 1000.
        :type log_interval: int
        :return: A batch of embeddings.
        :rtype: np.ndarray
        """
        logger.warning(
            "Current encoder does not support asynchronous encode,"
            "thus the code will be run in synchronous mode."
        )
        del batch_size, log_interval
        return self.encode(texts)

    @property
    @abstractmethod
    def embedding_size(self) -> int | None:
        """Get the dimension of the embeddings.
        If the dimension is dynamic or unknown, return None.
        """
        return


ENCODERS = Register[EncoderBase]("encoder")
