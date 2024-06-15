import re
import string


def normalize_answer(text: str):
    # remove_articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # unify white space
    text = " ".join(text.split())
    # remove punctuation
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return text
