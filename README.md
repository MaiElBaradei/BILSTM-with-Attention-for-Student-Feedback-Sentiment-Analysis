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
│   │
│   ├── preprocessing/              # MODIFIED: Decoupled text handling
│   │   ├── __init__.py             # Exposes clean package interface
│   │   ├── cleaning.py             # Regex text cleaning & stopword fallback dict
│   │   ├── vocabulary.py           # Word-to-index tracking (<PAD>/<UNK>)
│   │   └── tokenizer.py            # TextPreprocessor core execution engine
│   │
│   ├── data/                       # NEW: Modular persistence and data structures
│   │   ├── __init__.py             
│   │   ├── dataset.py              # Scalable, lazy-loading FeedbackDataset
│   │   ├── download.py             # Chunked data stream fetcher
│   │   └── splits.py               # Stratified dataset partitioner
│   │
│   ├── training/                   # MODIFIED: Included pipeline dataloaders
│   │   ├── __init__.py             
│   │   ├── dataloaders.py          # Multiprocess create_dataloaders + collate_fn
│   │   └── trainer.py              # Trainer: fit, evaluate, predict, checkpointing
│   │
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
