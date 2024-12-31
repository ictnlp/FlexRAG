# Code adapted from https://github.com/IAAR-Shanghai/xFinder
# As xFinder official implementation requires pytorch==2.3.1,
# we adapt the code to release the dependency on specific pytorch version.

import ast
import json
import re
from typing import Any, Dict, List, Literal, Tuple, Union

from flexrag.models.hf_model import load_hf_model

PROMPT_TEMPLATE = {
    "xFinder-qwen1505": """<|System|>:{system}
<|User|>:{input}
<|Bot|>:""",
    "xFinder-llama38it": """<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
}


Instruction = """I will provide you with a question, output sentences along with an answer range. The output sentences are the response of the question provided. The answer range could either describe the type of answer expected or list all possible valid answers. Using the information provided, you must accurately and precisely determine and extract the intended key answer from the output sentences. Please don't have your subjective thoughts about the question.
First, you need to determine whether the content of the output sentences is relevant to the given question. If the entire output sentences are unrelated to the question (meaning the output sentences are not addressing the question), then output [No valid answer].
Otherwise, ignore the parts of the output sentences that have no relevance to the question and then extract the key answer that matches the answer range.
Below are some special cases you need to be aware of: 
    (1) If the output sentences present multiple different answers, carefully determine if the later provided answer is a correction or modification of a previous one. If so, extract this corrected or modified answer as the final response. Conversely, if the output sentences fluctuate between multiple answers without a clear final answer, you should output [No valid answer].
    (2) If the answer range is a list and the key answer in the output sentences is not explicitly listed among the candidate options in the answer range, also output [No valid answer].

"""

SYSTEM_PROMPT = "You are a help assistant tasked with extracting the precise key answer from given output sentences."


class Extractor:
    """
    Extractor class for extracting key answers from a given question and output sentences.

    Args:
        model_name (Literal["xFinder-qwen1505", "xFinder-llama38it"]): The model name to be used for inference.
        inference_mode (Literal["local", "api"]): The mode of inference, either 'local' or 'api'.
        model_path_or_url (str): The path or URL of the model.
        temperature (float, optional): The temperature value for sampling. Defaults to 0.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 3000.

    Raises:
        ValueError: If inference_mode is not 'local' or 'api'.
        ValueError: If temperature or max_tokens are out of the expected range.

    Attributes:
        STOP_TOKENS (List[str]): List of stop tokens to be used for inference.
    """

    STOP_TOKENS = [
        "<|endoftext|>",
        "<|im_end|>",
        "<eoa>",
        "<||>",
        "<end_of_turn>",
        "<|eot_id|>",
    ]

    def __init__(
        self,
        model_name: Literal["xFinder-qwen1505", "xFinder-llama38it"],
        model_path_or_url: str,
        temperature: float = 0,
        max_tokens: int = 3000,
        device_id: list[int] = [],
    ):
        if not (0 <= temperature <= 1):
            raise ValueError("temperature should be between 0 and 1")

        if max_tokens <= 0:
            raise ValueError("max_tokens should be greater than 0")

        self.model_name = model_name
        self.model_path_or_url = model_path_or_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = SYSTEM_PROMPT
        self.tokenizer, self.model = load_hf_model(
            model_path=model_path_or_url, device_id=device_id
        )
        return

    def create_formatted_prompt(self, query: Dict[str, Any]) -> str:
        if self.model_name not in PROMPT_TEMPLATE:
            raise ValueError(
                f"Model name '{self.model_name}' is not supported in PROMPT_TEMPLATE."
            )
        return PROMPT_TEMPLATE[self.model_name].format(
            system=self.system_prompt, input=query
        )

    def _execute_local_inference(self, query: Dict[str, Any]) -> str:
        prompt = self.create_formatted_prompt(query)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        output_ids = self.model.generate(
            input_ids, max_new_tokens=self.max_tokens, temperature=self.temperature
        )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()

    def generate_output(self, question, llm_output, standard_answer_range) -> str:
        formatted_query = f'Question: """{question}"""\n\nOutput sentences: """{llm_output}"""\n\nAnswer range: {standard_answer_range}\n\nKey extracted answer: '
        return self._execute_local_inference(formatted_query)


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer (str): The final answer to normalize.

    Returns:
        str: The normalized final answer.
    """
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
        ("\\le", "<"),
    ]
    REMOVED_EXPRESSIONS = [
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
        "\n",
        "\r",
        "\f",
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(\\text\{)\((.*?)\)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    assert "\n" not in final_answer
    assert "\r" not in final_answer
    assert "\f" not in final_answer

    if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
        final_answer = re.findall(r"finalansweris(.*)", final_answer)[-1]

    if len(re.findall(r"answer?is:?(.*)", final_answer)) > 0:
        final_answer = re.findall(r"answer?is:?(.*)", final_answer)[-1]

    if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
        final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]

    if len(re.findall(r"\$(.*?)\$", final_answer)) > 0:
        final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]
    final_answer = final_answer.strip()
    if "rac" in final_answer and "\\frac" not in final_answer:
        final_answer = final_answer.replace("rac", "\\frac")

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


class MathEvaluator:

    def __init__(self, version="v2"):
        assert version in ["v1", "v2"]
        self.version = version

    def _fix_fracs(self, string):
        """Fixes fractions in the string.

        Args:
            string (str): The string to fix.

        Returns:
            str: The fixed string.
        """
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if len(substr) > 0 and substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(self, string):
        """Fixes a/b to \frac{a}{b}.

        Args:
            string (str): The string to fix.

        Returns:
            str: The fixed string.
        """
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except AssertionError:
            return string

    def _remove_right_units(self, string):
        # "\\text{ " only ever occurs (at least in the val set) when describing
        # units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(self, string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    def _fix_sqrt_v2(self, string):
        _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
        return _string

    def _strip_string(self, string):
        """Strip a string of unnecessary characters.

        Args:
            string (str): The string to strip.

        Returns:
            str: The stripped string.
        """
        # linebreaks
        string = string.replace("\n", "")

        # remove inverse spaces
        string = string.replace("\\!", "")

        # replace \\ with \
        string = string.replace("\\\\", "\\")

        # replace tfrac and dfrac with frac
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")

        # remove \left and \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")

        # Remove circ (degrees)
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")

        # remove dollar signs
        string = string.replace("\\$", "")

        # remove units (on the right)
        string = self._remove_right_units(string)

        # remove percentage
        string = string.replace("\\%", "")
        string = string.replace("\%", "")  # noqa: W605

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        # fix sqrt3 --> sqrt{3}
        string = self._fix_sqrt(string)

        # remove spaces
        string = string.replace(" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works
        # with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix
        # in case the model output is X/Y
        string = self._fix_a_slash_b(string)

        return string

    def _strip_string_v2(self, string):
        string = str(string).strip()
        # linebreaks
        string = string.replace("\n", "")

        # right "."
        string = string.rstrip(".")

        # remove inverse spaces
        string = string.replace("\\!", "")
        string = string.replace("\\ ", "")

        # replace \\ with \
        string = string.replace("\\\\", "\\")
        string = string.replace("\\\\", "\\")

        # replace tfrac and dfrac with frac
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")

        # remove \left and \right
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")

        # Remove unit: miles, dollars if after is not none
        _string = re.sub(r"\\text{.*?}$", "", string).strip()
        if _string != "" and _string != string:
            string = _string

        # Remove circ (degrees)
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")

        # remove dollar signs
        string = string.replace("\\$", "")
        string = string.replace("$", "")

        string = string.replace("\\text", "")
        string = string.replace("x\\in", "")

        # remove percentage
        string = string.replace("\\%", "")
        string = string.replace("\%", "")  # noqa: W605
        string = string.replace("%", "")

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")

        # cdot
        string = string.replace("\\cdot", "")

        # inf
        string = string.replace("infinity", "\\infty")
        if "\\infty" not in string:
            string = string.replace("inf", "\\infty")
        string = string.replace("+\\inity", "\\infty")

        # and
        string = string.replace("and", "")
        string = string.replace("\\mathbf", "")

        # use regex to remove \mbox{...}
        string = re.sub(r"\\mbox{.*?}", "", string)

        # quote
        string.replace("'", "")
        string.replace('"', "")

        # i, j
        if "j" in string and "i" not in string:
            string = string.replace("j", "i")

        # replace a.000b where b is not number or b is end, with ab, use regex
        string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
        string = re.sub(r"(\d+)\.0+$", r"\1", string)

        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split("=")) == 2:
            if len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]

        string = self._fix_sqrt_v2(string)
        string = string.replace(" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
        # Even works with \frac1{72} (but not \frac{72}1).
        # Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple
        # cases fix in case the model output is X/Y
        string = self._fix_a_slash_b(string)

        return string

    def is_equiv(self, str1, str2, verbose=False):
        """Check if two strings are equivalent.

        Args:
            str1 (str): The first string.
            str2 (str): The second string.
            verbose (bool): If True, print the stripped strings.

        Returns:
            bool: True if the strings are equivalent, False otherwise.
        """
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False

        if self.version == "v1":
            strip_string_func = self._strip_string
        elif self.version == "v2":
            strip_string_func = self._strip_string_v2
        else:
            raise NotImplementedError

        try:
            ss1 = strip_string_func(str1)
            ss2 = strip_string_func(str2)
            if verbose:
                print(ss1, ss2)
            if ss1 == ss2:
                return True
            ss1 = normalize_final_answer(ss1)
            ss2 = normalize_final_answer(ss2)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        try:
            ss1 = normalize_final_answer(str1)
            ss2 = normalize_final_answer(str2)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        # try:
        #     if is_equiv(str1, str2):
        #         return True
        # except Exception:
        #     pass

        # try:
        #     ss1 = normalize_final_answer(str1)
        #     ss2 = normalize_final_answer(str2)
        #     if is_equiv(ss1, ss2):
        #         return True
        # except Exception:
        #     pass

        return str1 == str2


class Comparator:
    """
    Comparator class for comparing extracted answers with correct answers.

    Attributes:
        math_evaluator (MathEvaluator): An instance of MathEvaluator class.
    """

    def __init__(self):
        self.math_evaluator = MathEvaluator()

    def compare(
        self, ext_cor_pair: Tuple[str, Union[str, list], str, str]
    ) -> List[Union[str, int]]:
        """Compare the extracted answer with the correct answer. Return a list of the comparison result.

        Args:
            ext_cor_pair (Tuple[str, Union[str, list], str, str]): A tuple of the extracted answer, correct answer, and the key answer type.

        Returns:
            List[Union[str, int]]: A list of the comparison result.
        """
        right_flag = 0
        key_answer_type, standard_answer_range, extracted, correct = ext_cor_pair
        if key_answer_type == "math":
            if self.math_evaluator.is_equiv(extracted, correct) == True:
                right_flag = 1
        else:
            if (
                extracted.strip().rstrip(".").lower()
                == correct.strip().rstrip(".").lower()
            ):
                right_flag = 1

            elif key_answer_type == "alphabet_option":
                if type(standard_answer_range) == str:
                    standard_answer_range_list = ast.literal_eval(standard_answer_range)
                for option in standard_answer_range_list:
                    if (
                        option[0] == correct
                        and extracted.strip().rstrip(".").lower()
                        == option[1].strip().rstrip(".").lower()
                    ):
                        right_flag = 1
                        break

        return [*ext_cor_pair, right_flag]


class DataProcessor:

    def __init__(self):
        pass

    def read_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if (
                isinstance(item["standard_answer_range"], str)
                and item["key_answer_type"] != "math"
            ):
                try:
                    item["standard_answer_range"] = ast.literal_eval(
                        item["standard_answer_range"]
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    print(
                        "Please check if you provide the correct form of the 'standard_answer_range': ",
                        item,
                    )
                    exit(0)

            item["standard_answer_range"] = str(item["standard_answer_range"])
            item["key_answer_type"] = str(item["key_answer_type"])

        return data


class Evaluator:
    """
    Evaluator class for evaluating the performance of the xFinder model.

    Args:
        model_name (str, optional): The model name to be used for inference. Defaults to "xFinder-qwen1505".
        inference_mode (str, optional): The mode of inference, either 'local' or 'api'. Defaults to "local".
        model_path_or_url (str, optional): The path or URL of the model. Defaults to "IAAR-Shanghai/xFinder-qwen1505".
        temperature (float, optional): The temperature value for sampling. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
    """

    MATH_STANDARD_ANSWER_RANGE = "a(n) number / set / vector / matrix / interval / expression / function / equation / inequality"
    VALID_KEY_ANSWER_TYPES = {
        "math",
        "short_text",
        "categorical_label",
        "alphabet_option",
    }

    def __init__(
        self,
        model_name: str = "xFinder-qwen1505",
        model_path_or_url: str = "IAAR-Shanghai/xFinder-qwen1505",
        temperature: float = 0.7,
        max_tokens: int = 100,
        device_id: list = [],
    ):
        self.extractor = Extractor(
            model_name=model_name,
            model_path_or_url=model_path_or_url,
            temperature=temperature,
            max_tokens=max_tokens,
            device_id=device_id,
        )
        self.comparator = Comparator()
        self.data_processor = DataProcessor()
        return

    def evaluate_single_item(
        self,
        question: str,
        llm_output: str,
        answer_range: str,
        answer_type: str,
        correct_answer: str,
    ):
        if answer_type not in self.VALID_KEY_ANSWER_TYPES:
            raise ValueError(
                f"Invalid key_answer_type: {answer_type}. Must be one of {self.VALID_KEY_ANSWER_TYPES}"
            )

        if answer_type == "math":
            answer_range = self.MATH_STANDARD_ANSWER_RANGE

        extracted_answer = self.extractor.generate_output(
            question, llm_output, answer_range
        )
        evaluation_result = self.comparator.compare(
            (answer_type, answer_range, extracted_answer, correct_answer)
        )
        return evaluation_result
