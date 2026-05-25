from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split


from .tokenizer import TextPreprocessor


class FeedbackDataset(Dataset):
    """
    Tokenizes and encodes text at construction time so DataLoader workers
    do not hold a reference to the preprocessor's Python objects.
    """

    def __init__(
        self, texts: List[str], labels: List[int], preprocessor: TextPreprocessor
    ):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for text, label in zip(texts, labels):
            padded, length = preprocessor.encode_padded(text)
            self.samples.append(
                (
                    torch.tensor(padded, dtype=torch.long),
                    torch.tensor(length, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch):
    """Sort by descending length so pack_padded_sequence works with enforce_sorted=True."""
    inputs, lengths, labels = zip(*batch)
    inputs = torch.stack(inputs)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)

    lengths, sort_idx = lengths.sort(descending=True)
    return inputs[sort_idx], lengths, labels[sort_idx]


def download_data(
    url: str = "https://data.mendeley.com/public-files/datasets/fvtfjyvw7d/files/256a4429-4fc3-4872-9a7c-26b44a820a8c/file_downloaded",
):
    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/data.csv"):
        print("[*] Dataset already downloaded, skipping.")
        return

    print("[*] Downloading dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        total_size = int(r.headers.get("content-length", 0))

        with (
            open("data/data.csv", "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading...",
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def create_dataloaders(
    texts: List[str],
    labels: List[int],
    preprocessor: TextPreprocessor,
    batch_size: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], List[int]]:

    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    val_frac = val_size / (1.0 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=val_frac,
        random_state=seed,
        stratify=train_val_labels,
    )

    preprocessor.fit(train_texts)

    train_ds = FeedbackDataset(train_texts, train_labels, preprocessor)
    val_ds = FeedbackDataset(val_texts, val_labels, preprocessor)
    test_ds = FeedbackDataset(test_texts, test_labels, preprocessor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    # test_texts / test_labels are returned for post-training visualization
    return train_loader, val_loader, test_loader, test_texts, test_labels
