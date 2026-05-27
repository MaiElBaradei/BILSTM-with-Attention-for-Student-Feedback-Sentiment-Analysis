"""
Bidirectional LSTM classifier with optional Bahdanau attention.

Sentence representation:
  use_attention=True  -> Bahdanau attention over all time steps
  use_attention=False -> concatenated final fwd+bwd hidden states

Embedding modes controlled by constructor flags:
  - Default (vocab)   : learns an embedding table from token indices
  - GloVe pre-trained : pass pretrained_embeddings tensor
  - LLM input         : set llm_input=True; model expects raw float tensors
                        instead of token indices (skips the embedding layer)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple

from .attention import BahdanauAttention, BaseAttention


class BiLSTMClassifier(nn.Module):
    """
    Parameters
    vocab_size            : vocabulary size (ignored when llm_input=True)
    embedding_dim         : token embedding / LLM hidden dim
    hidden_size           : LSTM hidden units per direction
    num_layers            : stacked LSTM layers
    num_classes           : output classes
    dropout               : applied after embedding, between LSTM layers, before classifier
    use_attention         : if False, uses final hidden state instead of attention context
    attn_dim              : internal projection size for BahdanauAttention
    pretrained_embeddings : (vocab_size, embedding_dim) weight tensor (GloVe)
    freeze_embeddings     : freeze the embedding table
    llm_input             : True -> input is already (batch, seq_len, dim) float tensors
    attention_module      : optional custom BaseAttention subclass (default: BahdanauAttention)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        use_attention: bool = True,
        attn_dim: int = 128,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        llm_input: bool = False,
        attention_module: Optional[BaseAttention] = None,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.llm_input = llm_input

        if llm_input:
            self.embedding = None  # handled externally
        elif pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.embed_drop = nn.Dropout(dropout)   # prevents over-reliance on specific words

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # pytorch's default (T, B, D), we use: (B, T, D) more intuitive.
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2  # concat of fwd + bwd

        if use_attention:
            print("[*] Using Attention Mechanism...")
            if attention_module is not None:
                print("[*] Using custom Base Attention")
                self.attention: BaseAttention = attention_module
            else:
                print("[*] Using Bahdanau Attention")
                self.attention = BahdanauAttention(
                    encoder_dim=lstm_out_dim,
                    query_dim=lstm_out_dim,
                    attn_dim=attn_dim,
                )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_classes)


    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        x       : (batch, seq_len)         token indices  when llm_input=False
                  (batch, seq_len, dim)    float embeddings when llm_input=True
        lengths : (batch,)                 actual (non-padded) length of each sequence

        Returns
        logits : (batch, num_classes)
        alpha  : (batch, seq_len) attention weights, or None when use_attention=False
        """
        # Embedding lookup (skipped for LLM path)
        if self.llm_input:
            embedded = self.embed_drop(x)  # (B, T, D)
        else:
            embedded = self.embed_drop(self.embedding(x))  # (B, T, D)

        # Pack -> LSTM -> unpack
        lengths_cpu = lengths.cpu().clamp(min=1) # .clamp(min=1) Avoids zero-length sequences crashing LSTM.

        # Packing API requires CPU tensor, (compresses sequences into efficient representation. LSTM processes ONLY real tokens. Huge optimization.)
        packed = pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=True
        )

        # LSTM forward
        # lstm_out: Contains hidden states for EVERY timestep. (B, T, 2H)
        # h_n: Final hidden states. (num_layers * directions, B, H)
        lstm_out, (h_n, _) = self.lstm(packed)

        # Restores padded tensor shape.
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # (B, T, 2H)

        # Padding mask: True = pad position (for masked_fill in attention)
        batch_size, seq_len, _ = lstm_out.shape
        pad_mask = torch.arange(seq_len, device=x.device).unsqueeze(
            0
        ) >= lengths.unsqueeze(  # (1, T)
            1
        )  # (B, 1)  # (B, T)

        # Build query: concat final forward + backward hidden states (last layer)
        # h_n shape: (num_layers * 2, batch, hidden_size)
        fwd_last = h_n[-2]  # (B, H)
        bwd_last = h_n[-1]  # (B, H)
        final_hidden = torch.cat([fwd_last, bwd_last], dim=-1)  # (B, 2H)

        if self.use_attention:
            context, alpha = self.attention(lstm_out, final_hidden, pad_mask)
            out = self.dropout(context)
        else:
            out = self.dropout(final_hidden)
            alpha = None

        logits = self.classifier(out)
        return logits, alpha
