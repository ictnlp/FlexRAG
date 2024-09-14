from .processor import PROCESSORS, Processor, TextUnit


@PROCESSORS("exact_deduplicate")
class ExactDeduplicate(Processor):
    def __init__(self) -> None:
        self.seen = set()
        return

    def process(self, input_text: TextUnit) -> TextUnit:
        if input_text.content in self.seen:
            input_text.reserved = False
        self.seen.add(input_text.content)
        return input_text
