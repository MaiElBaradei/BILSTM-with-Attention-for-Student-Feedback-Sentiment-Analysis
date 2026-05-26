import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
from preprocessing.tokenizer import TextPreprocessor
from data.splits import split_dataset
from data.dataset import FeedbackDataset

def collate_fn(batch):
    """Sort by descending length so pack_padded_sequence works out-of-the-box."""
    inputs, lengths, labels = zip(*batch)
    inputs = torch.stack(inputs)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)

    lengths, sort_idx = lengths.sort(descending=True)
    return inputs[sort_idx], lengths, labels[sort_idx]

def create_dataloaders(
    texts: List[str],
    labels: List[int],
    preprocessor: TextPreprocessor,
    batch_size: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], List[int]]:
    """Coordinates splits, fits preprocessor vocab, and initializes DataLoaders."""
    
    # Partition text structures
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_dataset(
        texts, labels, test_size, val_size, seed
    )

    # State modification (Fit vocabulary strictly on Training subsets)
    preprocessor.fit(train_texts)

    # Initialize PyTorch structural collections
    train_ds = FeedbackDataset(train_texts, train_labels, preprocessor)
    val_ds = FeedbackDataset(val_texts, val_labels, preprocessor)
    test_ds = FeedbackDataset(test_texts, test_labels, preprocessor)

    # Build DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, test_texts, test_labels