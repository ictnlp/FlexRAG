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
    def encode(self, texts: list[str] | str) -> np.ndarray:
        """Encode the given texts into embeddings.

        :param texts: A batch of texts.
        :type texts: list[str] | str
        :return: A batch of embeddings.
        :rtype: np.ndarray
        """
        return

    async def async_encode(self, texts: list[str] | str) -> np.ndarray:
        """Encode the given texts into embeddings.

        :param texts: A batch of texts.
        :type texts: list[str] | str
        :return: A batch of embeddings.
        :rtype: np.ndarray
        """
        logger.warning(
            "Current encoder does not support asyncronous encode,"
            "thus the code will be run in syncronous mode."
        )
        return self.encode(texts)

    def encode_batch(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        """Encode the given texts into embeddings in batches.

        :param texts: A batch of texts.
        :type texts: list[str] | str
        :param batch_size: The size of each batch. Defaults to 32.
        :type batch_size: int
        :param log_interval: The interval for logging progress. Defaults to 1000.
            If set to 0, no logs will be shown.
        :type log_interval: int
        :return: A batch of embeddings.
        :rtype: np.ndarray
        """
        if not isinstance(texts, list):
            texts = [texts]

        # prepare progress logger
        if (
            (len(texts) > log_interval)
            and (len(texts) > batch_size)
            and (log_interval > 0)
        ):
            p_logger = SimpleProgressLogger(logger, len(texts), log_interval)
        else:
            p_logger = None

        # encode
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            embeddings.append(self.encode(batch_texts))
            if p_logger is not None:
                p_logger.update(len(batch_texts), desc="Encoding")
        return np.concatenate(embeddings, axis=0)

    @property
    @abstractmethod
    def embedding_size(self) -> int | None:
        """Get the dimension of the embeddings.
        If the dimension is dynamic or unknown, return None.
        """
        return


ENCODERS = Register[EncoderBase]("encoder")
