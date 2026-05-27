from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Data
    url: str = (
        "https://data.mendeley.com/public-files/datasets/fvtfjyvw7d/files/256a4429-4fc3-4872-9a7c-26b44a820a8c/file_downloaded"
    )
    data_path: str = "data/feedback.csv"
    text_col: str = "text"
    label_col: str = "label"
    label_names: list = field(default_factory=lambda: ["Negative", "Positive"])
    max_seq_len: int = 256
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42

    # Embedding 
    # "random" | "glove" | "llm"
    embedding_type: str = "random"
    embedding_dim: int = 100
    glove_path: Optional[str] = None  # path to glove.6B.100d.txt (or similar)
    llm_model: str = "bert-base-uncased"  # any HuggingFace model name
    llm_layer: int = -1  # which hidden layer to extract (-1 = last)
    freeze_embeddings: bool = False

    # Model
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    use_attention: bool = True  # toggle Bahdanau attention on/off
    attn_dim: int = 128  # internal attention projection size
    num_classes: int = 2

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    use_class_weights: bool = True  # inverse-frequency weighting for imbalanced data
    epochs: int = 20
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    clip_grad: float = 1.0
    patience: int = 5  # early-stopping patience (epochs)
    lr_patience: int = 2  # ReduceLROnPlateau patience

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None  # path to a .pt checkpoint

    # Visualization
    viz_output_dir: str = "visualizations"
    viz_samples: int = 5  # number of samples to visualize
  
    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "bilstm-sentiment"
    wandb_run_name: Optional[str] = None  # auto-generated when None
    wandb_entity: Optional[str] = None  # W&B username or team (None = default)
