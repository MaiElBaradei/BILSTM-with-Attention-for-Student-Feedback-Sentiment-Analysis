import torch
from torch.utils.data import Dataset
from typing import List
from preprocessing.tokenizer import TextPreprocessor

class FeedbackDataset(Dataset):
    """
    Scalable Dataset that tokenizes on-the-fly via __getitem__.
    This reduces the memory footprint and safely leverages DataLoader multi-processing.
    Args:
        texts (List[str]): List of text samples.
        labels (List[int]): List of corresponding labels.
        preprocessor (TextPreprocessor): Preprocessor instance for tokenization.
    """
    def __init__(self, texts: List[str], labels: List[int], preprocessor: TextPreprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        
        padded, length = self.preprocessor.encode_padded(text)
        
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )