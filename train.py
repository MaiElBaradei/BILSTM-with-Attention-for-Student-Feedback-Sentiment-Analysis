#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import random
import sys
import unicodedata
from pathlib import Path
import pickle

import pandas as pd
import torch

from bilstm_attention.config import Config
from bilstm_attention.preprocessing import download_data
from bilstm_attention.embeddings import download_glove_default
from bilstm_attention.model.bilstm import BiLSTMClassifier
from bilstm_attention.training.trainer import Trainer
from bilstm_attention.visualization.attention_viz import (
    visualize_batch,
    visualize_llm_batch,
    plot_training_history,
)



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a BiLSTM (± Bahdanau attention) sentiment classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data_path",       default="data/data.csv")
    p.add_argument("--text_col",        default="comments")
    p.add_argument("--label_col",       default="star_rating")
    p.add_argument("--label_names",     default="Negative,Positive",
                   help="Comma-separated class names")
    p.add_argument("--max_seq_len",     type=int, default=256)
    p.add_argument("--test_size",       type=float, default=0.2)
    p.add_argument("--val_size",        type=float, default=0.1)
    p.add_argument("--seed",            type=int, default=42)

    # Embedding
    p.add_argument("--embedding_type",  default="random",
                   choices=["random", "glove", "llm"])
    p.add_argument("--embedding_dim",   type=int, default=100)
    p.add_argument("--glove_path",      default=None)
    p.add_argument("--llm_model",       default="bert-base-uncased")
    p.add_argument("--llm_layer",       type=int, default=-1)
    p.add_argument("--freeze_embeddings", action=argparse.BooleanOptionalAction, default=False)

    # Model
    p.add_argument("--hidden_size",     type=int, default=128)
    p.add_argument("--num_layers",      type=int, default=2)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--use_attention",   action=argparse.BooleanOptionalAction, default=True,
                   help="Use --no-use_attention to disable Bahdanau attention")
    p.add_argument("--attn_dim",        type=int, default=128)
    p.add_argument("--num_classes",     type=int, default=2)

    # Training
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--learning_rate",   type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=1e-5)
    p.add_argument("--epochs",          type=int, default=20)
    p.add_argument("--device",          default="auto",
                   help='"auto" picks CUDA > MPS > CPU automatically')
    p.add_argument("--clip_grad",       type=float, default=1.0)
    p.add_argument("--patience",        type=int, default=5)
    p.add_argument("--lr_patience",     type=int, default=2)
    p.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True,
                   help="Weight loss by inverse class frequency (recommended for imbalanced data)")

    # Checkpointing
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    p.add_argument("--resume_from",     default=None,
                   help="Path to a .pt checkpoint to resume training from")

    # Visualisation
    p.add_argument("--viz_output_dir",  default="visualizations")
    p.add_argument("--viz_samples",     type=int, default=5)
    
    # Weights & Biases
    p.add_argument("--use_wandb",       action=argparse.BooleanOptionalAction, default=True,
                   help="Enable W&B tracking (--no-use_wandb to disable)")
    p.add_argument("--wandb_project",   default="bilstm-sentiment")
    p.add_argument("--wandb_run_name",  default=None,
                   help="W&B run name (auto-generated when omitted)")
    p.add_argument("--wandb_entity",    default=None,
                   help="W&B username or team name")

    return p


def args_to_config(args: argparse.Namespace) -> Config:
    d = vars(args).copy()
    d["label_names"] = [n.strip() for n in d["label_names"].split(",")]
    return Config(**d)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)



_NO_COMMENT_RE = re.compile(
    r"^\s*(no\s+comments?|no\s+feedback|not\s+applicable|n\s*/\s*a|none|nothing|nil|-+|\.+)\s*$",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)
    # 2. Non-breaking / invisible unicode whitespace
    text = re.sub(r"[\xa0­​-‍⁠﻿]", " ", text)
    # 3. Repeated characters (keep at most 2)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # 4. Repeated punctuation (keep just one)
    text = re.sub(r"([!?,.]){2,}", r"\1", text)
    # 5. Whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_csv(config: Config):
    download_data(config.url)
    df = pd.read_csv(config.data_path)

    if config.text_col not in df.columns or config.label_col not in df.columns:
        sys.exit(
            f"ERROR: CSV must contain columns '{config.text_col}' and "
            f"'{config.label_col}'. Found: {list(df.columns)}"
        )

    n_raw = len(df)

    # Label cleaning
    df[config.label_col] = pd.to_numeric(df[config.label_col], errors="coerce")
    df = df[df[config.label_col].between(0, 5)]

    # Drop neutral (rating == 3) — only keep clear negative (<3) or positive (>3)
    df = df[df[config.label_col] != 3.0].copy()

    # Binarise: <3 → 0 (Negative),  >3 → 1 (Positive)
    df[config.label_col] = (df[config.label_col] > 3).astype(int)

    # Text cleaning─
    df[config.text_col] = df[config.text_col].fillna("").astype(str)

    # Drop rows whose comment is a "no comment" placeholder
    no_comment_mask = df[config.text_col].apply(
        lambda t: bool(_NO_COMMENT_RE.fullmatch(t.strip()))
    )
    n_no_comment = int(no_comment_mask.sum())
    df = df[~no_comment_mask].copy()

    # Normalize remaining text
    df[config.text_col] = df[config.text_col].apply(_normalize_text)

    # Drop rows that became empty after normalization
    df = df[df[config.text_col].str.len() > 0].copy()

    n_kept = len(df)
    print(
        f"[data]  {n_raw:,} raw rows  →  {n_kept:,} kept  "
        f"({n_raw - n_kept:,} dropped: neutral ratings + {n_no_comment:,} 'no comment' rows)"
    )
    print(
        f"[data]  class balance — "
        f"Negative: {(df[config.label_col]==0).sum():,}  "
        f"Positive: {(df[config.label_col]==1).sum():,}"
    )

    texts  = df[config.text_col].tolist()
    labels = df[config.label_col].tolist()
    return texts, labels


# Class-weight computation

def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """Balanced inverse-frequency weights: w_c = N / (num_classes * count_c)."""
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    return torch.tensor(weights, dtype=torch.float)


# Standard (vocab) path

def run_standard(config: Config, device: torch.device) -> None:
    import matplotlib.pyplot as plt
    from bilstm_attention.preprocessing import TextPreprocessor
    from bilstm_attention.preprocessing import create_dataloaders
    from bilstm_attention.embeddings.glove import get_glove_embedding_layer

    texts, labels = load_csv(config)

    preprocessor = TextPreprocessor(
        max_seq_len=config.max_seq_len,
        min_freq=2,
        extra_stopwords={"class", "classes", "prof", "professor", "teacher"},
    )
    train_loader, val_loader, test_loader, test_texts, test_labels = create_dataloaders(
        texts, labels, preprocessor,
        batch_size=config.batch_size,
        test_size=config.test_size,
        val_size=config.val_size,
        seed=config.seed,
    )
    print(f"Vocab size: {len(preprocessor.vocab):,}")
    print(f"Train / Val / Test batches: "
          f"{len(train_loader)} / {len(val_loader)} / {len(test_loader)}")

    # Save preprocessor so the inference server can tokenise at prediction time
    
    prep_path = Path(config.checkpoint_dir) / "preprocessor.pkl"
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(prep_path, "wb") as fh:
        pickle.dump(preprocessor, fh)
    print(f"Preprocessor saved → {prep_path}")

    pretrained = None
    if config.embedding_type == "glove":
        if not config.glove_path:
            download_glove_default()
        layer = get_glove_embedding_layer(
            preprocessor.vocab,
            config.glove_path,
            config.embedding_dim,
            freeze=config.freeze_embeddings,
        )
        pretrained = layer.weight.data

    model = BiLSTMClassifier(
        vocab_size=len(preprocessor.vocab),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        use_attention=config.use_attention,
        attn_dim=config.attn_dim,
        pretrained_embeddings=pretrained,
        freeze_embeddings=config.freeze_embeddings,
        llm_input=False,
    )
    _print_model_summary(model, config)

    cw = compute_class_weights(labels, config.num_classes) if config.use_class_weights else None
    if cw is not None:
        print(f"Class weights: {cw.tolist()}")
    trainer = Trainer(model, config, device, class_weights=cw)
    if config.resume_from:
        trainer.load_checkpoint(config.resume_from)

    trainer.fit(train_loader, val_loader)

    print("\n──── Test set evaluation ────")
    report = trainer.evaluate(test_loader, label_names=config.label_names)
    _print_report(report)
    _wandb_log_report(report)

    hist_path = f"{config.viz_output_dir}/training_history.png"
    plt.close(plot_training_history(trainer.history, save_path=hist_path))
    print(f"Training history saved → {hist_path}")
    _wandb_log_images([hist_path], "plots")

    if config.use_attention:
        print(f"\nGenerating {config.viz_samples} attention visualisations…")
        rng = random.Random(config.seed)
        idx = rng.sample(range(len(test_texts)), min(config.viz_samples, len(test_texts)))
        visualize_batch(
            model=model,
            samples=[test_texts[i] for i in idx],
            labels=[test_labels[i] for i in idx],
            label_names=config.label_names,
            preprocessor=preprocessor,
            device=str(device),
            output_dir=config.viz_output_dir,
            max_samples=config.viz_samples,
        )
        import glob
        _wandb_log_images(glob.glob(f"{config.viz_output_dir}/sample_*.png"), "attention")


# LLM embedding path

def run_llm(config: Config, device: torch.device) -> None:
    import matplotlib.pyplot as plt
    from bilstm_attention.embeddings.llm_embeddings import LLMEmbedder, create_llm_dataloaders

    texts, labels = load_csv(config)

    embedder = LLMEmbedder(
        model_name=config.llm_model,
        layer=config.llm_layer,
        device=str(device),
    )
    config.embedding_dim = embedder.hidden_size
    print(f"LLM hidden size → embedding_dim = {config.embedding_dim}")

    train_loader, val_loader, test_loader, test_texts, test_labels = create_llm_dataloaders(
        embedder=embedder,
        texts=texts,
        labels=labels,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        test_size=config.test_size,
        val_size=config.val_size,
        seed=config.seed,
    )

    model = BiLSTMClassifier(
        vocab_size=1,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        use_attention=config.use_attention,
        attn_dim=config.attn_dim,
        llm_input=True,
    )
    _print_model_summary(model, config)

    cw = compute_class_weights(labels, config.num_classes) if config.use_class_weights else None
    if cw is not None:
        print(f"Class weights: {cw.tolist()}")
    trainer = Trainer(model, config, device, class_weights=cw)
    if config.resume_from:
        trainer.load_checkpoint(config.resume_from)

    trainer.fit(train_loader, val_loader)

    print("\n──── Test set evaluation ────")
    report = trainer.evaluate(test_loader, label_names=config.label_names)
    _print_report(report)
    _wandb_log_report(report)

    hist_path = f"{config.viz_output_dir}/training_history.png"
    plt.close(plot_training_history(trainer.history, save_path=hist_path))
    print(f"Training history saved → {hist_path}")
    _wandb_log_images([hist_path], "plots")

    if config.use_attention:
        print(f"\nGenerating {config.viz_samples} attention visualisations…")
        rng = random.Random(config.seed)
        idx = rng.sample(range(len(test_texts)), min(config.viz_samples, len(test_texts)))
        visualize_llm_batch(
            model=model,
            samples=[test_texts[i] for i in idx],
            labels=[test_labels[i] for i in idx],
            label_names=config.label_names,
            embedder=embedder,
            device=str(device),
            output_dir=config.viz_output_dir,
            max_samples=config.viz_samples,
            max_length=config.max_seq_len,
        )
        import glob
        _wandb_log_images(glob.glob(f"{config.viz_output_dir}/sample_*_llm.png"), "attention")


# Helpers

def _print_model_summary(model, config: Config) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nModel  hidden={config.hidden_size}  layers={config.num_layers}  "
        f"attention={'ON' if config.use_attention else 'OFF'}  "
        f"embedding={config.embedding_type}\n"
        f"  Total params:     {total:,}\n"
        f"  Trainable params: {trainable:,}\n"
    )


def _print_report(report: dict) -> None:
    skip = {"accuracy", "macro avg", "weighted avg"}
    for cls, m in report.items():
        if isinstance(m, dict) and cls not in skip:
            print(f"  {cls:<14} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}")
    if "macro avg" in report:
        m = report["macro avg"]
        print(f"  {'macro avg':<14} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}")
    if "accuracy" in report:
        print(f"  Accuracy: {report['accuracy']:.4f}")


def _wandb_log_report(report: dict) -> None:
    """Push test classification report to W&B summary."""
    try:
        import wandb
        if wandb.run is None:
            return
        for cls, m in report.items():
            if isinstance(m, dict):
                for metric, val in m.items():
                    wandb.summary[f"test/{cls}/{metric}"] = val
        if "accuracy" in report:
            wandb.summary["test/accuracy"] = report["accuracy"]
    except ImportError:
        pass


def _wandb_log_images(paths: list[str], prefix: str) -> None:
    """Log PNG files to W&B as Images."""
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log({f"{prefix}/{Path(p).stem}": wandb.Image(p)
                   for p in paths if Path(p).exists()})
    except ImportError:
        pass


# Entry point

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = args_to_config(args)
    device = resolve_device(config.device)

    print(f"Device   : {device}")
    print(f"Attention: {'ON' if config.use_attention else 'OFF (ablation)'}")
    print(f"Embedding: {config.embedding_type}")

    Path(config.viz_output_dir).mkdir(parents=True, exist_ok=True)

    if config.use_wandb:
        if config.wandb_run_name is None:
            config.wandb_run_name = (
                f"{config.embedding_type}"
                f"-h{config.hidden_size}"
                f"-{'attn' if config.use_attention else 'no-attn'}"
                f"-{'cw' if config.use_class_weights else 'no-cw'}"
            )
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                entity=config.wandb_entity,
                config={k: v for k, v in config.__dict__.items()
                        if not k.startswith("wandb_")},
                tags=[config.embedding_type,
                      "attention" if config.use_attention else "no-attention"],
            )
            print(f"W&B run: {wandb.run.url}")
        except ImportError:
            print("[W&B] wandb not installed — tracking disabled")
            config.use_wandb = False

    try:
        if config.embedding_type == "llm":
            run_llm(config, device)
        else:
            run_standard(config, device)
    finally:
        if config.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass


if __name__ == "__main__":
    main()
