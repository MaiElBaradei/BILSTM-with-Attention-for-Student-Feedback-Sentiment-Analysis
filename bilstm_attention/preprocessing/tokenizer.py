from typing import List, Set
from .cleaning import clean_text, load_stopwords
from .vocabulary import Vocabulary

class TextPreprocessor:
    """
    Coordinates vocabulary state, data string execution, and transformation.
    """
    def __init__(
        self,
        max_seq_len: int = 256,
        min_freq: int = 2,
        remove_stopwords: bool = True,
        lang: str = "english",
        extra_stopwords: Set[str] | None = None,
    ):
        self.max_seq_len = max_seq_len
        self.vocab = Vocabulary(min_freq)
        self._fitted = False
        self._stopwords: Set[str] = load_stopwords(lang) if remove_stopwords else set()
        if extra_stopwords:
            self._stopwords |= {w.lower() for w in extra_stopwords}

    @staticmethod
    def clean(text: str) -> str:
        return clean_text(text)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.split()

    def _filter(self, tokens: List[str]) -> List[str]:
        if not self._stopwords:
            return tokens
        return [t for t in tokens if t not in self._stopwords]

    def fit(self, texts: List[str]) -> "TextPreprocessor":
        tokenized = [self._filter(self.tokenize(self.clean(t))) for t in texts]
        self.vocab.build(tokenized)
        self._fitted = True
        return self

    def get_tokens(self, text: str) -> List[str]:
        tokens = self._filter(self.tokenize(self.clean(text)))
        return tokens[: self.max_seq_len]

    def encode(self, text: str) -> List[int]:
        return self.vocab.encode(self.get_tokens(text))

    def encode_padded(self, text: str) -> tuple[List[int], int]:
        indices = self.encode(text)
        length = max(len(indices), 1)
        padded = indices + [Vocabulary.PAD_IDX] * (self.max_seq_len - len(indices))
        return padded, length