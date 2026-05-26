# BILSTM-with-Attention-for-Student-Feedback-Sentiment-Analysis

## Repository Structure

```
.
├── bilstm_attention/               # Core Python package
│   ├── config.py                   # Dataclass with every hyper-parameter
│   ├── embeddings/
│   │   ├── glove.py                # GloVe loader + embedding matrix builder
│   │   └── llm_embeddings.py       # HuggingFace wrapper + pre-compute cache
│   ├── model/
│   │   ├── attention.py            # Base Attention + Bahdanau Attention
│   │   └── bilstm.py               # BiLSTM Classifier
│   ├── preprocessing/
│   │   ├── tokenizer.py            # Vocabulary + TextPreprocessor
│   │   └── dataset.py              
│   ├── training/
│   │   └── trainer.py              # Trainer: fit, evaluate, predict, checkpointing
│   └── visualization/
│       └── attention_viz.py        # Heatmap, word cloud, training history plots
│
├── serve/                          # FastAPI inference server
│   ├── app.py                      # Endpoints + browser UI
│   └── loader.py                   # Checkpoint + preprocessor loading
│
├── train.py                        # Training entry point (CLI)
│
├── Dockerfile                      # Training image
├── Dockerfile.serve                # CPU-only serving image
├── docker-compose.yml              
├── .dockerignore
│
├── requirements.txt                # Requirement dependencies as .txt
│
├── pyproject.toml                  # Build config + all dependency groups
└── setup.py
```
