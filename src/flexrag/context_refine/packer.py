import random as rd
from dataclasses import dataclass

from flexrag.common_dataclass import RetrievedContext
from flexrag.utils import Choices, TIME_METER

from .refiner import REFINERS, RefinerBase


@dataclass
class BasicPackerConfig:
    order: Choices(["ascending", "descending", "side", "random"]) = "ascending"  # type: ignore


@REFINERS("basic_packer", config_class=BasicPackerConfig)
class BasicPacker(RefinerBase):
    def __init__(self, config: BasicPackerConfig):
        self.order = config.order
        return

    @TIME_METER("repack")
    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        match self.order:
            case "ascending":
                contexts = sorted(contexts, key=lambda x: x.score)
            case "descending":
                contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
            case "random":
                indices = list(range(len(contexts)))
                rd.shuffle(indices)
                contexts = [contexts[i] for i in indices]
            case "side":
                sort_ctxs = sorted(contexts, key=lambda x: x.score, reverse=True)
                contexts_left = []
                contexts_right = []
                for i in range(0, len(sort_ctxs), 2):
                    contexts_left.append(sort_ctxs[i])
                for i in range(1, len(sort_ctxs), 2):
                    contexts_right.append(sort_ctxs[i])
                contexts = contexts_left + contexts_right[::-1]
            case _:
                raise ValueError(f"Invalid order: {self.order}")
        return contexts
