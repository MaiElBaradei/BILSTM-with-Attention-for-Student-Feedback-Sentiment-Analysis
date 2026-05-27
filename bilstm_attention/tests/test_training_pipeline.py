from __future__ import annotations

import os
from types import SimpleNamespace
import tempfile

import torch
from bilstm_attention.preprocessing.dataloaders import create_dataloaders
from bilstm_attention.preprocessing.tokenizer import TextPreprocessor
from bilstm_attention.model import BiLSTMClassifier
from bilstm_attention.training.trainer import Trainer


def generate_balanced_texts(samples_per_class: int = 10):
    texts = []
    labels = []
    for _ in range(samples_per_class):
        texts.append("good product fast delivery")
        labels.append(1)
        texts.append("bad product slow delivery")
        labels.append(0)
    return texts, labels


def test_training_pipeline_runs_end_to_end(tmp_path):
    texts, labels = generate_balanced_texts(10)

    preprocessor = TextPreprocessor(
        max_seq_len=12,
        min_freq=1,
        remove_stopwords=False,
    )

    train_loader, val_loader, _, _, _ = create_dataloaders(
        texts=texts,
        labels=labels,
        preprocessor=preprocessor,
        batch_size=8,
        num_workers=0,
        test_size=0.1,
        val_size=0.1,
        seed=42,
    )

    model = BiLSTMClassifier(
        vocab_size=len(preprocessor.vocab),
        embedding_dim=16,
        hidden_size=8,
        num_layers=1,
        num_classes=2,
        dropout=0.1,
        use_attention=True,
        attn_dim=16,
    )

    config = SimpleNamespace(
        learning_rate=1e-3,
        weight_decay=0.0,
        lr_patience=1,
        clip_grad=1.0,
        epochs=2,
        patience=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    trainer = Trainer(model, config, device=torch.device("cpu"))
    trainer.fit(train_loader, val_loader)

    assert len(trainer.history["train_loss"]) == trainer.config.epochs
    assert len(trainer.history["val_f1"]) == trainer.config.epochs
    assert os.path.exists(tmp_path / "checkpoints" / "epoch_000.pt")

    report = trainer.evaluate(val_loader)
    # classification_report returns class keys as strings ('0','1', ...) and an 'accuracy' entry
    assert "accuracy" in report
    assert "0" in report and "1" in report
