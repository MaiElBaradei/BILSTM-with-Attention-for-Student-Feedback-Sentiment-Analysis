"""
Bahdanau (additive) attention — implemented from scratch.

The score function is:
    e_j = v^T  tanh( W1 * h_j  +  W2 * s )

where
    h_j  : j-th encoder output  (batch, seq_len, encoder_dim)
    s    : query vector          (batch, decoder_dim)   — the final BiLSTM hidden state
    v, W1, W2 : learned parameters

Attention weights and context vector:
    alpha = softmax(e)                        (batch, seq_len)
    context = Σ_j alpha_j * h_j              (batch, encoder_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class BaseAttention(ABC, nn.Module):
    """Attention modules must implement forward() with this signature."""

    @abstractmethod
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        encoder_outputs : (batch, seq_len, encoder_dim)
        query           : (batch, query_dim)
        mask            : (batch, seq_len) BoolTensor — True marks padding positions

        Returns
        context : (batch, encoder_dim)
        alpha   : (batch, seq_len)   soft attention weights that sum to ~1
        """



class BahdanauAttention(BaseAttention):
    """
    Bahdanau additive attention.

    Parameters
    encoder_dim : dimensionality of encoder outputs  (2*hidden for BiLSTM)
    query_dim   : dimensionality of the query vector (2*hidden for BiLSTM)
    attn_dim    : size of the internal tanh projection
    """

    def __init__(self, encoder_dim: int, query_dim: int, attn_dim: int = 128):
        super().__init__()
        # Project encoder outputs: (batch, seq_len, attn_dim), from (B, T, 256) (fwd + bwd ) -> 128
        self.W1 = nn.Linear(encoder_dim, attn_dim, bias=False)
        # Project query: (batch, attn_dim)
        self.W2 = nn.Linear(query_dim, attn_dim, bias=False)
        # Score scalar: (batch, seq_len, 1) -> squeeze -> (batch, seq_len); Converts attention hidden vector into scalar score.
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # W1(h):  (batch, seq_len, attn_dim)
        # W2(s):  (batch, attn_dim)  -> unsqueeze -> (batch, 1, attn_dim)  [broadcasts]
        energy = self.v(
            torch.tanh(self.W1(encoder_outputs) + self.W2(query).unsqueeze(1))
        ).squeeze(
            -1
        )  # (batch, seq_len), Now each word has scalar attention score.

        # Mask padding positions with -inf before softmax
        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        alpha = F.softmax(energy, dim=-1)  # (batch, seq_len)

        # Replace NaN that can appear when an entire row was -inf (shouldn't
        # happen with valid data, but guards against edge cases)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Weighted sum of encoder outputs (weighted sum across sequence)
        context = torch.bmm(
            alpha.unsqueeze(1),  # (batch, 1, seq_len)
            encoder_outputs,  # (batch, seq_len, encoder_dim)
        ).squeeze(
            1
        )  # (batch, encoder_dim)

        return context, alpha
