from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path
import os
import requests
import zipfile
from tqdm import tqdm


from ..preprocessing.tokenizer import Vocabulary


def download_glove_default():
    os.makedirs("embeddings_files", exist_ok=True)
    output_path = "embeddings_files/glove.6B.zip"

    if os.path.exists("data/glove.6B.100d.txt"):
        print("[*] GloVe already extracted, skipping.")
        return
    if os.path.exists(output_path):
        print("[*] GloVe zip found, extracting...")
        with zipfile.ZipFile(output_path, "r") as z:
            z.extractall("data")
        return
    print("[*] Downloading GloVe 6B...")
    with requests.get("https://nlp.stanford.edu/data/glove.6B.zip", stream=True) as r:
        r.raise_for_status()

        total_size = int(r.headers.get("content-length", 0))

        with (
            open(output_path, "wb") as f,
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

    with zipfile.ZipFile(output_path, "r") as z:
        z.extractall("data")


def load_glove(path: str) -> Dict[str, np.ndarray]:
    """Parse a GloVe .txt file into a word→vector dictionary."""
    embeddings: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    print(f"Loaded {len(embeddings):,} GloVe vectors from {Path(path).name}")
    return embeddings


def build_embedding_matrix(
    vocab: Vocabulary,
    glove: Dict[str, np.ndarray],
    embedding_dim: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    words not in GloVe get a small random vector
    PAD (idx=0) stays 0s
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[Vocabulary.PAD_IDX] = 0.0

    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            vec = glove[word]
            if vec.shape[0] != embedding_dim:   # to align with model's dim
                raise ValueError(
                    f"GloVe dim ({vec.shape[0]}) ≠ embedding_dim ({embedding_dim}). "
                    "Make sure you load the correct GloVe variant."
                )
            matrix[idx] = vec
            found += 1

    coverage = 100.0 * found / len(vocab)
    print(f"GloVe coverage: {found}/{len(vocab)} tokens ({coverage:.1f}%)")
    return torch.from_numpy(matrix)


DEFAULT_GLOVE_PATH = f"data/glove.6B.100d.txt"


def get_glove_embedding_layer(
    vocab: Vocabulary,
    glove_path: Optional[str],
    embedding_dim: int,
    freeze: bool = False,
    seed: int = 42,
) -> nn.Embedding:
    """Return an nn.Embedding initialised with GloVe weights."""
    resolved = glove_path or DEFAULT_GLOVE_PATH
    glove = load_glove(resolved)
    matrix = build_embedding_matrix(vocab, glove, embedding_dim, seed)
    layer = nn.Embedding.from_pretrained(
        matrix, freeze=freeze, padding_idx=Vocabulary.PAD_IDX
    )
    return layer
