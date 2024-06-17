from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .model_base import GeneratorBase
from .utils import get_prompt_func


class HFGenerator(GeneratorBase):
    def __init__(self, model_path: str, pipeline_parallel: bool = False) -> None:
        if pipeline_parallel:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.prompt_func = get_prompt_func(model=self.model, tokenizer=self.tokenizer)
        return

    def generate(
        self, prefixes: list[str], generation_config: GenerationConfig = None
    ) -> list[str]:
        inputs = self.tokenizer(prefixes, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        responses = self.model.generate(
            **inputs,
            generation_config=generation_config,
            eos_token_id=generation_config.forced_eos_token_id,
        )
        input_lengths = inputs["attention_mask"].sum(dim=1)
        responses = [i[l:] for i, l in zip(responses, input_lengths)]
        texts = [self.tokenizer.decode(i, skip_special_tokens=True) for i in responses]
        return texts

    def chat(
        self,
        prompts: list[list[dict[str, str]]],
        generation_config: GenerationConfig = None,
    ) -> list[str]:
        prefixes = [self.prompt_func(prompt) for prompt in prompts]
        return self.generate(prefixes, generation_config)
