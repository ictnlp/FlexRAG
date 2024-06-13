from logging import Logger
from time import perf_counter

from transformers import PreTrainedTokenizer


def apply_template_llama3(
    history: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    sys_prompt: str = "You are a pirate chatbot who always responds in pirate speak!",
) -> str:
    # add system prompt
    if len(history) == 0:
        history.append({"role": "system", "content": sys_prompt})
    if history[0]["role"] != "system":
        history.insert(0, {"role": "system", "content": sys_prompt})

    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    return prompt


def apply_template_phi3(
    history: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    sys_prompt: str = None,
) -> str:
    # add system prompt
    if (len(history) == 0) and (sys_prompt is not None):
        history.append({"role": "system", "content": sys_prompt})
    if (history[0]["role"] != "system") and (sys_prompt is not None):
        history.insert(0, {"role": "system", "content": sys_prompt})

    # apply template
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    return prompt


class SimpleProgressLogger:
    def __init__(self, logger: Logger, total: int = None, interval: int = 100):
        self.total = total
        self.interval = interval
        self.logger = logger
        self.current = 0
        self.current_stage = 0
        self.desc = "Progress"
        self.start_time = perf_counter()
        return

    def update(self, step: int = 1, desc: str = None) -> None:
        if desc is not None:
            self.desc = desc
        self.current += step
        stage = self.current // self.interval
        if stage > self.current_stage:
            self.current_stage = stage
            self.log()
        return

    def log(self) -> None:
        def fmt_time(time: float) -> str:
            if time < 60:
                return f"{time:.2f}s"
            if time < 3600:
                return f"{time//60:02.0f}:{time%60:02.0f}"
            else:
                return f"{time//3600:.0f}:{(time%3600)//60:02.0f}:{time%60:02.0f}"

        if (self.total is not None) and (self.current < self.total):
            time_spend = perf_counter() - self.start_time
            time_left = time_spend * (self.total - self.current) / self.current
            speed = self.current / time_spend
            num_str = f"{self.current} / {self.total}"
            percent_str = f"({self.current/self.total:.2%})"
            time_str = f"[{fmt_time(time_spend)} / {fmt_time(time_left)}, {speed:.2f} update/s]"
            self.logger.info(f"{self.desc}: {num_str} {percent_str} {time_str}")
        else:
            time_spend = perf_counter() - self.start_time
            speed = self.current / time_spend
            num_str = f"{self.current}"
            time_str = f"[{fmt_time(time_spend)}, {speed:.2f} update/s]"
            self.logger.info(f"{self.desc}: {num_str} {time_str}")
        return
