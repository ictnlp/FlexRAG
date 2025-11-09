from abc import abstractmethod
from dataclasses import field
from typing import Annotated, Optional

from flexrag.utils import Choices, Register, configure, data
from flexrag.utils.dataclasses import Context

from .dataset import MappingDataset
from .hf_dataset import HFDataset, HFDatasetConfig


@data
class QAEvalData:
    """The dataclass for konwledge intensive QA task.

    :param question: The question for evaluation. Required.
    :type question: str
    :param golden_contexts: The contexts related to the question. Default: None.
    :type golden_contexts: Optional[list[Context]]
    :param golden_answers: The golden answers for the question. Default: None.
    :type golden_answers: Optional[list[str]]
    :param meta_data: The metadata of the evaluation data. Default: {}.
    :type meta_data: dict
    """

    question: str
    golden_contexts: Optional[list[Context]] = None
    golden_answers: Optional[list[str]] = None
    meta_data: dict = field(default_factory=dict)


class QADataset(MappingDataset[QAEvalData]):
    """Interface for knowledge intensive QA dataset."""

    @property
    @abstractmethod
    def form(self) -> str:
        """The form of the dataset, should be one of the following:
        * long: The response is a long text.
        * short: The response is a short text.
        """
        return


QA_DATASETS = Register[QADataset]("qa_dataset")


@configure
class FlashQADatasetConfig:
    """The configuration for ``FlashQADataset``.
    This dataset helps to load the QA datasets collected by
    `FlashRAG <https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets>`_.
    The ``__iter__`` method will yield `QAEvalData` objects.

    :param name: The name of the dataset to load.
    :type name: str
    :param split: The split of the dataset to load. Default: "test".
    :type split: str
    :param path: The path to the dataset. Default: "RUC-NLPIR/FlashRAG_datasets".
    :type path: str

    For example, you can load the `test` set of the `NaturalQuestions` dataset by running the following code:

    .. code-block:: python

        from flexrag.datasets import FlashQADataset, FlashQADatasetConfig

        cfg = FlashQADatasetConfig(
            name="nq",
            split="test",
        )
        dataset = FlashQADataset(cfg)

    You can also load the dataset from a local repository by specifying the path.
    For example, you can download the dataset by running the following command:

        >>> git lfs install
        >>> git clone https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets flashrag

    Then you can load the dataset by running the following code:

    .. code-block:: python

        from flexrag.datasets import FlashQADataset, FlashQADatasetConfig

        cfg = FlashQADatasetConfig(
            path="json",
            data_files=["flashrag/nq/test.jsonl"],
            split="train",
        )
        dataset = FlashQADataset(cfg)

    Available Short-Form QA datasets:

        - popqa: test
        - nq: dev, test, train
        - triviaqa: dev, test, train
        - 2wikimultihopqa: dev, train
        - hotpotqa: dev, train
        - musique: dev, train
        - bamboogle: test
        - squad: dev, train
        - web_questions: test, train
        - curatedtrec: test, train
        - fermi: dev, test, train

    Available Long-Form QA datasets:

        - ambig_qa: dev, train
        - asqa: dev, train
        - eli5: dev, train
        - piqa: dev, train
        - siqa: dev, train
        - msmarco-qa: dev, train
        - narrativeqa: dev, test, train
        - wikiqa: dev, test, train
        - wikiasp: dev, test, train

    """

    name: Annotated[
        str,
        Choices(
            "nq",
            "popqa",
            "triviaqa",
            "hotpotqa",
            "2wikimultihopqa",
            "musique",
            "bamboogle",
            "squad",
            "web_questions",
            "curatedtrec",
            "fermi",
            "ambig_qa",
            "asqa",
            "eli5",
            "piqa",
            "siqa",
            "msmarco-qa",
            "narrativeqa",
            "wikiqa",
            "wikiasp",
        ),
    ]
    split: Annotated[str, Choices("dev", "test", "train")] = "test"
    path: str = "RUC-NLPIR/FlashRAG_datasets"


@QA_DATASETS("flashrag_qa", config_name=FlashQADatasetConfig)
class FlashQADataset(QADataset):
    """The dataset for loading Knowledge Intensive QA task dataset from the
    `FlashRAG <https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets>`_ repository.
    """

    short_form_qa_names = {
        "nq",
        "popqa",
        "triviaqa",
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "bamboogle",
        "squad",
        "web_questions",
        "curatedtrec",
        "fermi",
    }
    long_form_qa_names = {
        "ambig_qa",
        "asqa",
        "eli5",
        "piqa",
        "siqa",
        "msmarco-qa",
        "narrativeqa",
        "wikiqa",
        "wikiasp",
    }

    def __init__(self, cfg: FlashQADatasetConfig) -> None:
        self.dataset = HFDataset(
            HFDatasetConfig(
                path=cfg.path,
                name=cfg.name,
                split=cfg.split,
            )
        )
        if cfg.name in self.short_form_qa_names:
            self._form = "short"
        elif cfg.name in self.long_form_qa_names:
            self._form = "long"
        else:
            raise ValueError(
                f"Unknown dataset name: {cfg.name}. "
                "Please choose from the available datasets."
            )
        return

    def __getitem__(self, index: int) -> QAEvalData:
        data = self.dataset[index]
        golden_contexts = data.pop("golden_contexts", None)
        golden_contexts = (
            [Context(**context) for context in golden_contexts]
            if golden_contexts is not None
            else None
        )
        # multiple choice data
        formatted_data = QAEvalData(
            question=data.pop("question"),
            golden_contexts=golden_contexts,
            golden_answers=data.pop("golden_answers", None),
        )
        formatted_data.meta_data = data.pop("meta_data", {})
        formatted_data.meta_data.update(data)
        return formatted_data

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def form(self) -> str:
        """The form of the dataset."""
        return self._form
