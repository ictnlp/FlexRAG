from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from librarian.utils import Register


@dataclass
class TextUnit:
    content: str
    reserved: bool = True
    processed_by: list[str] = field(default_factory=list)


class Processor(ABC):
    def __call__(self, input_text: TextUnit) -> TextUnit:
        input_text.processed_by.append(self.name)
        return self.process(input_text)

    @abstractmethod
    def process(self, input_text: TextUnit) -> TextUnit:
        return

    @property
    def name(self):
        return self.__class__.__name__


PROCESSORS = Register[Processor]("processor")
