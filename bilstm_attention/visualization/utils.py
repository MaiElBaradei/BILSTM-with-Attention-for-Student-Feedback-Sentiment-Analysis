from typing import Dict, List, Optional
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    from wordcloud import WordCloud as _WordCloud

    _WC_AVAILABLE = True
except ImportError:
    _WC_AVAILABLE = False

def _get_label_name(idx: int, label_names: Optional[List[str]]) -> str:
    if label_names and idx < len(label_names):
        return label_names[idx]
    return str(idx)


def _build_explanation(
    tokens: List[str],
    weights: np.ndarray,
    true_label: Optional[int],
    pred_label: Optional[int],
    label_names: Optional[List[str]],
    probs: Optional[np.ndarray],
    top_k: int = 6,
) -> str:
    """
    Generate a human-readable explanation for a single prediction.

    Covers:
      • What the model predicted and its confidence
      • Whether it matched the ground truth
      • Which tokens received the most attention
      • How focused or diffuse the attention distribution was
    """
    n = len(tokens)
    w = np.asarray(weights[:n], dtype=np.float64)
    lines: List[str] = []

    if pred_label is not None:
        pred_name = _get_label_name(pred_label, label_names)
        conf_str = (
            f"  (confidence: {probs[pred_label] * 100:.1f}%)"
            if probs is not None
            else ""
        )

        if true_label is not None:
            true_name = _get_label_name(true_label, label_names)
            if true_label == pred_label:
                verdict = "  ✓  correct"
            else:
                verdict = f'  ✗  incorrect  —  true label: "{true_name}"'
            lines.append(f"Prediction:       {pred_name}{conf_str}{verdict}")
        else:
            lines.append(f"Prediction:       {pred_name}{conf_str}")

    # Show full class probability distribution if available
    if probs is not None and label_names is not None:
        dist_parts = [
            f"{_get_label_name(i, label_names)}: {p * 100:.1f}%"
            for i, p in enumerate(probs)
            if i < len(label_names)
        ]
        lines.append(f"Class probs:      {',  '.join(dist_parts)}")

    # Top-k attended tokens
    if n > 0:
        top_idx = np.argsort(w)[::-1][:top_k]
        top_parts = [
            f'"{tokens[j]}" ({w[j] * 100:.1f}%)' for j in top_idx if j < n and tokens[j]
        ]
        if top_parts:
            lines.append(f"Top attended:     {',  '.join(top_parts)}")

    # Attention focus (entropy-based)
    if n > 1:
        w_safe = w + 1e-12
        w_norm = w_safe / w_safe.sum()
        entropy = float(-np.sum(w_norm * np.log(w_norm)))
        max_entropy = math.log(n)
        focus_pct = max(0.0, (1.0 - entropy / max_entropy) * 100)

        if focus_pct >= 70:
            descriptor = "highly concentrated — model relies on a few key tokens"
        elif focus_pct >= 40:
            descriptor = "moderately focused — several tokens share influence"
        else:
            descriptor = "widely distributed — model considers many tokens equally"

        lines.append(f"Attention focus:  {focus_pct:.0f}%  ({descriptor})")

    return "\n".join(lines)


def _draw_heatmap(
    ax: plt.Axes,
    tokens: List[str],
    weights: np.ndarray,
    tokens_per_row: int,
    cell_h: float,
    cell_w: float,
) -> tuple:
    """Draw the token-grid heatmap.  Returns (cmap, norm) for a shared colorbar."""
    n = len(tokens)
    n_cols = min(n, tokens_per_row)
    n_rows = math.ceil(n / n_cols)

    pad = n_rows * n_cols - n
    tokens_padded = tokens + [""] * pad
    weights_padded = np.concatenate([weights, np.zeros(pad)])

    grid_t = np.array(tokens_padded).reshape(n_rows, n_cols)
    grid_w = weights_padded.reshape(n_rows, n_cols)

    cmap = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0.0, vmax=max(float(weights.max()), 1e-8))

    font_sz = max(6, 10 - n_cols // 4)

    for r in range(n_rows):
        for c in range(n_cols):
            tok = grid_t[r, c]
            w = grid_w[r, c]
            color = cmap(norm(w))
            rect = plt.Rectangle(
                [c, n_rows - r - 1],
                1.0,
                1.0,
                facecolor=color,
                linewidth=0.5,
                edgecolor="white",
            )
            ax.add_patch(rect)
            if tok:
                text_color = "white" if norm(w) > 0.65 else "black"
                ax.text(
                    c + 0.5,
                    n_rows - r - 0.5,
                    tok,
                    ha="center",
                    va="center",
                    fontsize=font_sz,
                    color=text_color,
                    rotation=45 if n_cols > 16 else 0,
                )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Token Attention Heatmap", fontsize=9, pad=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        pad=0.02,
        fraction=0.04,
        label="Attention weight",
    )

    return cmap, norm


def _draw_wordcloud(ax: plt.Axes, tokens: List[str], weights: np.ndarray) -> None:
    """Draw a word cloud sized by attention weight into ax."""
    if not _WC_AVAILABLE:
        ax.text(
            0.5,
            0.5,
            "wordcloud\nnot installed\n(pip install wordcloud)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="grey",
        )
        ax.set_axis_off()
        return

    # Build frequency dict — deduplicate by summing weights for repeated tokens
    freq: Dict[str, float] = {}
    for tok, w in zip(tokens, weights):
        if tok:
            freq[tok] = freq.get(tok, 0.0) + float(w)

    if not freq:
        ax.set_axis_off()
        return

    max_w = max(freq.values())
    cmap = plt.get_cmap("YlOrRd")

    def _color_func(word: str, **_kw) -> str:
        norm_w = min(max(freq.get(word, 0.0) / max(max_w, 1e-8), 0.15), 1.0)
        rgba = cmap(0.2 + 0.8 * norm_w)
        return f"rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})"

    wc = _WordCloud(
        width=380,
        height=260,
        background_color="white",
        color_func=_color_func,
        max_words=60,
        relative_scaling=0.6,
        prefer_horizontal=0.85,
        min_font_size=8,
        collocations=False,
    ).generate_from_frequencies(freq)

    ax.imshow(wc, interpolation="bilinear")
    ax.set_title("Attention Word Cloud", fontsize=9, pad=4)
    ax.set_axis_off()


def _draw_explanation(ax: plt.Axes, text: str) -> None:
    """Render the written explanation inside a light-grey box."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            facecolor="#f5f5f5",
            linewidth=1,
            edgecolor="#cccccc",
            clip_on=False,
        )
    )

    ax.text(
        0.015,
        0.95,
        "Explanation",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#333333",
        va="top",
        ha="left",
    )
    ax.text(
        0.015,
        0.78,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#222222",
        va="top",
        ha="left",
        family="monospace",
        linespacing=1.5,
    )