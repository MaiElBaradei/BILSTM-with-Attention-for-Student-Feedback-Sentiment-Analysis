from collections import Counter
from typing import Dict, List

class Vocabulary:
    """Maps tokens to integer indices, reserving 0=PAD and 1=UNK."""
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self._next_idx = 2

    def build(self, tokenized_texts: List[List[str]]) -> None:
        counts = Counter(tok for tokens in tokenized_texts for tok in tokens)
        for word, freq in counts.most_common():
            if freq < self.min_freq:
                break
            if word not in self.word2idx:
                self.word2idx[word] = self._next_idx
                self.idx2word[self._next_idx] = word
                self._next_idx += 1

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.word2idx.get(t, self.UNK_IDX) for t in tokens]

    def __len__(self) -> int:
        return self._next_idx

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx