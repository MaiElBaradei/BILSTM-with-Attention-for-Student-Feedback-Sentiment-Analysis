from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report


from ..config import Config


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",  # monitor val_f1 (higher is better)
            patience=config.lr_patience,
            factor=0.5,
        )
        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        self.best_val_f1: float = 0.0
        self.start_epoch: int = 0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }


    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
    ) -> Dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        all_preds: list = []
        all_labels: list = []
        tag = "train" if train else "eval "

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for inputs, lengths, labels in tqdm(loader, desc=tag, leave=False):
                inputs = inputs.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                logits, _ = self.model(inputs, lengths)
                loss = self.criterion(logits, labels)

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad
                    )
                    self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        n = len(all_labels)
        return {
            "loss": total_loss / n,
            "acc": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        }


    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        patience_counter = 0

        for epoch in range(self.start_epoch, self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            train_m = self._run_epoch(train_loader, train=True)
            val_m = self._run_epoch(val_loader, train=False)

            self.scheduler.step(val_m["f1"])

            for k in ("loss", "acc", "f1"):
                self.history[f"train_{k}"].append(train_m[k])
                self.history[f"val_{k}"].append(val_m[k])

            print(
                f"  train  loss={train_m['loss']:.4f}  "
                f"acc={train_m['acc']:.4f}  f1={train_m['f1']:.4f}"
            )
            print(
                f"  val    loss={val_m['loss']:.4f}  "
                f"acc={val_m['acc']:.4f}  f1={val_m['f1']:.4f}"
            )

            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train/loss": train_m["loss"],
                            "train/acc": train_m["acc"],
                            "train/f1": train_m["f1"],
                            "val/loss": val_m["loss"],
                            "val/acc": val_m["acc"],
                            "val/f1": val_m["f1"],
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )
            except ImportError:
                pass

            is_best = val_m["f1"] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_m["f1"]
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{self.config.patience})")

            if patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

        print(f"\nTraining complete.  Best val_f1={self.best_val_f1:.4f}")


    def evaluate(
        self,
        loader: DataLoader,
        label_names: Optional[list] = None,
    ) -> Dict[str, Any]:
        self.model.eval()
        all_preds: list = []
        all_labels: list = []

        with torch.no_grad():
            for inputs, lengths, labels in loader:
                inputs = inputs.to(self.device)
                lengths = lengths.to(self.device)
                logits, _ = self.model(inputs, lengths)
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(labels.tolist())

        # Only report on label indices that actually appear in this split.
        # This prevents a ValueError when label_names has more entries than
        # the number of unique classes seen (e.g. binary data + 3-name list).
        unique_labels = sorted(set(all_labels + all_preds))
        if label_names is not None:
            effective_names = [
                label_names[i] for i in unique_labels if i < len(label_names)
            ]
            if len(effective_names) != len(unique_labels):
                effective_names = None
        else:
            effective_names = None

        report = classification_report(
            all_labels,
            all_preds,
            labels=unique_labels,
            target_names=effective_names,
            output_dict=True,
            zero_division=0,
        )
        return report


    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a forward pass in eval mode. Returns (logits, alpha)."""
        self.model.eval()
        inputs = inputs.to(self.device)
        lengths = lengths.to(self.device)
        return self.model(inputs, lengths)
