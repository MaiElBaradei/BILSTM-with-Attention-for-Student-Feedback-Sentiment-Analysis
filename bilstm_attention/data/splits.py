from typing import List, Tuple
from sklearn.model_selection import train_test_split

def split_dataset(
    texts: List[str],
    labels: List[int],
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """Splits texts and labels into stratified train, validation, and test arrays."""
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
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels