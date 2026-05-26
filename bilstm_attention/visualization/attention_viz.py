from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from .utils import _draw_explanation, _draw_heatmap, _draw_wordcloud, _build_explanation




def visualize_attention(
    tokens: List[str],
    weights: np.ndarray,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    tokens_per_row: int = 12,
    cell_h: float = 0.70,
    cell_w: float = 1.40,
    true_label: Optional[int] = None,
    pred_label: Optional[int] = None,
    label_names: Optional[List[str]] = None,
    probs: Optional[np.ndarray] = None,
    top_k_explanation: int = 6,
) -> plt.Figure:
    """
    Produce a three-panel visualisation for one sample.

    Panels
    ------
    1. Token attention heatmap  — grid of coloured token cells
    2. Attention word cloud     — tokens sized proportionally to attention weight
    3. Written explanation      — prediction, confidence, verdict, top tokens,
                                  attention focus metric

    Panels 2 and 3 appear only when the relevant arguments are supplied.
    Panel 2 is silently replaced by a placeholder if `wordcloud` is not installed.

    Parameters
    ----------
    tokens            : real (non-padded) token strings
    weights           : attention weights aligned to tokens
    title             : suptitle shown above the figure
    save_path         : if given, saves to disk as PNG
    tokens_per_row    : wrap the heatmap after this many columns
    cell_h / cell_w   : inches per heatmap cell
    true_label        : ground-truth class index
    pred_label        : predicted class index
    label_names       : e.g. ["Negative", "Positive"]
    probs             : 1-D softmax probability array (one entry per class)
    top_k_explanation : number of top tokens to list in the explanation
    """
    n = len(tokens)
    weights = np.asarray(weights[:n], dtype=np.float32)

    n_cols = min(n, tokens_per_row)
    n_rows = math.ceil(n / n_cols)

    has_enrichment = any(x is not None for x in (true_label, pred_label, probs))

    heatmap_w = max(n_cols * cell_w, 4.5)
    heatmap_h = max(n_rows * cell_h, 1.5)
    wc_w = 4.0
    expl_h = 1.8

    fig_w = heatmap_w + wc_w + 0.6
    fig_h = heatmap_h + 1.2 + (expl_h + 0.3 if has_enrichment else 0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=9, y=0.99, va="top", wrap=True)

    if has_enrichment:
        outer = gridspec.GridSpec(
            2,
            1,
            figure=fig,
            height_ratios=[heatmap_h + 1.2, expl_h],
            hspace=0.25,
            top=0.94,
            bottom=0.03,
            left=0.02,
            right=0.98,
        )
        inner = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=outer[0],
            width_ratios=[heatmap_w, wc_w],
            wspace=0.08,
        )
        ax_heat = fig.add_subplot(inner[0])
        ax_wc = fig.add_subplot(inner[1])
        ax_text = fig.add_subplot(outer[1])
    else:
        gs = gridspec.GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[heatmap_w, wc_w],
            wspace=0.08,
            top=0.93,
            bottom=0.10,
            left=0.02,
            right=0.98,
        )
        ax_heat = fig.add_subplot(gs[0])
        ax_wc = fig.add_subplot(gs[1])
        ax_text = None

    # Draw panels
    _draw_heatmap(ax_heat, tokens, weights, tokens_per_row, cell_h, cell_w)
    _draw_wordcloud(ax_wc, tokens, weights)

    if has_enrichment and ax_text is not None:
        explanation = _build_explanation(
            tokens,
            weights,
            true_label,
            pred_label,
            label_names,
            probs,
            top_k=top_k_explanation,
        )
        _draw_explanation(ax_text, explanation)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

def visualize_batch(
    model: torch.nn.Module,
    samples: List[str],
    labels: List[int],
    label_names: List[str],
    preprocessor,
    device: str,
    output_dir: str = "visualizations",
    max_samples: int = 5,
) -> None:
    """
    Run inference on raw-text samples (standard vocab path), extract
    attention weights, and save one enriched PNG per sample.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    model.eval()

    for i, (text, label) in enumerate(zip(samples[:max_samples], labels[:max_samples])):
        tokens = preprocessor.get_tokens(text)
        if not tokens:
            print(f"  Sample {i}: empty after preprocessing — skipping.")
            continue

        padded, length = preprocessor.encode_padded(text)
        x = torch.tensor([padded], dtype=torch.long).to(dev)
        l_tensor = torch.tensor([length], dtype=torch.long).to(dev)

        with torch.no_grad():
            logits, alpha = model(x, l_tensor)

        if alpha is None:
            print(f"  Sample {i}: attention disabled — skipping visualisation.")
            continue

        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        pred = int(probs.argmax())
        attn_weights = alpha[0, :length].cpu().numpy()

        snippet = text[:70] + ("…" if len(text) > 70 else "")
        title = f'"{snippet}"'

        save_path = f"{output_dir}/sample_{i:02d}.png"
        fig = visualize_attention(
            tokens,
            attn_weights,
            title=title,
            save_path=save_path,
            true_label=label,
            pred_label=pred,
            label_names=label_names,
            probs=probs,
        )
        plt.close(fig)
        print(f"  Saved {save_path}")