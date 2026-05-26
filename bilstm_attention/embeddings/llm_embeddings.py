from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from sklearn.model_selection import train_test_split


from ..preprocessing import collate_fn as _base_collate_fn


class LLMEmbedder:
    """
    Wraps a HuggingFace AutoModel to extract token-level hidden states.

    Parameters
    ----------
    model_name : HuggingFace model identifier, e.g. "bert-base-uncased"
    layer      : Which transformer hidden layer to use. -1 = last layer.
    device     : "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        layer: int = -1,
        device: str = "cpu",
    ):
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for LLM embeddings.\n"
                "Install with:  pip install transformers"
            ) from exc

        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.layer = layer
        self.hidden_size: int = self.model.config.hidden_size

    @torch.no_grad()
    def embed(
        self,
        texts: List[str],
        max_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        texts      : list of raw strings
        max_length : maximum sub-word token length (truncates longer texts)

        Returns
        -------
        hidden : (batch, seq_len, hidden_size)  float32, on CPU
        mask   : (batch, seq_len)               1 = real token, 0 = padding
        """
        enc = self.hf_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[self.layer].cpu()  # (B, T, H)
        mask = enc["attention_mask"].cpu()  # (B, T)
        return hidden, mask

    def get_tokens(self, text: str, max_length: int = 256) -> List[str]:
        """Return sub-word tokens as strings (for attention visualisation)."""
        ids = self.hf_tokenizer.encode(
            text, truncation=True, max_length=max_length, add_special_tokens=True
        )
        return self.hf_tokenizer.convert_ids_to_tokens(ids)


class LLMEmbeddingDataset(Dataset):
    """
    Pre-computes LLM embeddings for an entire split and stores them in RAM.

    This is the recommended approach for training: pay the transformer cost
    once, then iterate over the dataset cheaply in every epoch.
    """

    def __init__(
        self,
        embedder: LLMEmbedder,
        texts: List[str],
        labels: List[int],
        max_seq_len: int = 256,
        batch_size: int = 16,
    ):
        all_hidden: List[torch.Tensor] = []
        all_lengths: List[int] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            hidden, mask = embedder.embed(batch, max_length=max_seq_len)
            all_hidden.append(hidden)
            all_lengths.extend(mask.sum(dim=1).tolist())

        self.hidden = torch.cat(all_hidden, dim=0)  # (N, T, H)
        self.lengths = torch.tensor(all_lengths, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = self.hidden.shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.hidden[idx], self.lengths[idx], self.labels[idx]


def _llm_collate_fn(batch):
    """Sort by descending length for pack_padded_sequence compatibility."""
    hiddens, lengths, labels = zip(*batch)
    hiddens = torch.stack(hiddens)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    lengths, sort_idx = lengths.sort(descending=True)
    return hiddens[sort_idx], lengths, labels[sort_idx]


def create_llm_dataloaders(
    embedder: LLMEmbedder,
    texts: List[str],
    labels: List[int],
    batch_size: int,
    max_seq_len: int = 256,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], List[int]]:
    """Split data, pre-compute embeddings per split, return
    (train, val, test DataLoaders, test_texts, test_labels).

    test_texts / test_labels are returned for post-training visualisation.
    """

    train_val_t, test_t, train_val_l, test_l = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    val_frac = val_size / (1.0 - test_size)
    train_t, val_t, train_l, val_l = train_test_split(
        train_val_t,
        train_val_l,
        test_size=val_frac,
        random_state=seed,
        stratify=train_val_l,
    )

    print("Pre-computing LLM embeddings (train)…")
    train_ds = LLMEmbeddingDataset(embedder, train_t, train_l, max_seq_len)
    print("Pre-computing LLM embeddings (val)…")
    val_ds = LLMEmbeddingDataset(embedder, val_t, val_l, max_seq_len)
    print("Pre-computing LLM embeddings (test)…")
    test_ds = LLMEmbeddingDataset(embedder, test_t, test_l, max_seq_len)

    kw = dict(collate_fn=_llm_collate_fn, num_workers=0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
        test_t,
        test_l,
    )
