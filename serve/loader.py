from __future__ import annotations

import pickle
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Tuple

import torch

from bilstm_attention.config import Config
from bilstm_attention.model.bilstm import BiLSTMClassifier
from bilstm_attention.preprocessing.tokenizer import TextPreprocessor


def load(
    checkpoint_dir: str = "checkpoints",
) -> Tuple[BiLSTMClassifier, TextPreprocessor, Config, dict]:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_path = ckpt_dir / "best.pt"
    prep_path = ckpt_dir / "preprocessor.pkl"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run training first:  python train.py"
        )
    if not prep_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {prep_path}\n"
            "Re-run training — train.py saves preprocessor.pkl automatically."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    saved_cfg = ckpt["config"]
    known = {f.name for f in dc_fields(Config)}
    config = Config(**{k: v for k, v in saved_cfg.items() if k in known})

    with open(prep_path, "rb") as fh:
        preprocessor: TextPreprocessor = pickle.load(fh)

    if config.embedding_type == "llm":
        raise RuntimeError(
            "This checkpoint was trained with LLM embeddings. "
            "LLM inference requires a running transformer model and is not "
            "supported by the lightweight serving container. "
            "Re-train with --embedding_type random or --embedding_type glove."
        )

    model = BiLSTMClassifier(
        vocab_size=len(preprocessor.vocab),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        use_attention=config.use_attention,
        attn_dim=config.attn_dim,
        llm_input=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {
        "best_val_f1": float(ckpt.get("best_val_f1", 0.0)),
        "trained_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_path": str(ckpt_path),
    }

    return model, preprocessor, config, meta
