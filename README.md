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
│   │   ├── dataset.py              # Scalable, lazy-loading FeedbackDataset
│   │   ├── download.py             # Chunked data stream fetcher
│   │   ├── dataloaders.py          # Multiprocess create_dataloaders + collate_fn
│   │   └── splits.py               # Stratified dataset partitioner
│   │
│   ├── training/                   # MODIFIED: Included pipeline dataloaders
│   │   ├── __init__.py             
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

---

## Installation

### Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.2 (CUDA recommended for training)

### Local install

```bash
# Clone the repository
git clone https://github.com/MaiElBaradei/BILSTM-with-Attention-for-Student-Feedback-Sentiment-Analysis.git
cd BILSTM-with-Attention-for-Student-Feedback-Sentiment-Analysis

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -e .

# Optional — LLM embeddings (BERT / RoBERTa)
pip install -e ".[llm]"

# Optional — richer visualisations + word cloud
pip install -e ".[viz]"

# Optional — inference server
pip install -e ".[serve]"
```
---

## Training

### Embedding Modes

Three embedding strategies are available, selectable via `--embedding_type`.

#### `random` (default)

A randomly-initialised `nn.Embedding` table that is learned entirely from the training data alongside the BiLSTM weights.  Fast and requires no external files.

```bash
python train.py
```

#### `glove`

The embedding table is initialised from a pre-trained **GloVe** file.  If no GloVe file is on disk, the full `glove.6B.zip` (~822 MB) is downloaded automatically from Stanford NLP.  Unknown words receive small random vectors; the PAD token is always zero.

```bash
# Use GloVe 200-dimensional vectors (recommended)
python train.py \
  --embedding_type glove \
  --glove_path data/glove.6B.200d.txt \
  --embedding_dim 200 \
  --freeze_embeddings        # freeze: fewer trainable params, less overfitting
```

Coverage is printed at startup, e.g. `GloVe coverage: 6577/6920 tokens (95.0%)`.

#### `llm`

Token-level hidden states are extracted from a **HuggingFace transformer** (default: `bert-base-uncased`) and fed directly into the BiLSTM, replacing the embedding layer.  The transformer weights are never updated (frozen feature extractor).  Embeddings for all splits are pre-computed and cached in RAM before training begins, so subsequent epochs are fast.

```bash
python train.py \
  --embedding_type llm \
  --llm_model bert-base-uncased \
  --llm_layer -1             # -1 = last hidden layer
```

> **Note**: LLM inference requires `pip install -e ".[llm]"`.  The serving container does **not** support LLM checkpoints — re-train with `random` or `glove` if you want to deploy the inference server.

---

### All CLI Flags

```
Data
  --data_path PATH          CSV file to load                    [data/data.csv]
  --text_col NAME           Column containing the text          [comments]
  --label_col NAME          Column containing the rating        [star_rating]
  --label_names A,B         Comma-separated class names         [Negative,Positive]
  --max_seq_len INT         Maximum token sequence length       [256]
  --test_size FLOAT         Fraction held out for test          [0.2]
  --val_size FLOAT          Fraction held out for validation    [0.1]
  --seed INT                Random seed for splits              [42]

Embedding
  --embedding_type STR      random | glove | llm                [random]
  --embedding_dim INT       Embedding dimensionality            [100]
  --glove_path PATH         Path to GloVe .txt file             [None → auto-download]
  --llm_model STR           HuggingFace model name              [bert-base-uncased]
  --llm_layer INT           Transformer layer to extract        [-1]
  --freeze_embeddings       Freeze the embedding table          [False]
  --no-freeze_embeddings

Model
  --hidden_size INT         LSTM hidden units per direction     [128]
  --num_layers INT          Stacked LSTM layers                 [2]
  --dropout FLOAT           Dropout rate (embed + LSTM + head)  [0.3]
  --use_attention           Enable Bahdanau attention           [True]
  --no-use_attention        Ablation: use final hidden state
  --attn_dim INT            Attention projection size           [128]
  --num_classes INT         Output classes                      [2]

Training
  --batch_size INT          Mini-batch size                     [32]
  --learning_rate FLOAT     Adam initial LR                     [1e-3]
  --weight_decay FLOAT      Adam L2 weight decay                [1e-5]
  --use_class_weights       Inverse-frequency loss weighting    [True]
  --no-use_class_weights
  --epochs INT              Maximum training epochs             [20]
  --device STR              auto | cpu | cuda | mps             [auto]
  --clip_grad FLOAT         Gradient clipping max-norm          [1.0]
  --patience INT            Early-stopping patience (epochs)    [5]
  --lr_patience INT         ReduceLROnPlateau patience          [2]

Checkpointing
  --checkpoint_dir PATH     Directory for saved checkpoints     [checkpoints]
  --resume_from PATH        Resume from a .pt checkpoint        [None]

Visualisation
  --viz_output_dir PATH     Directory for output PNGs           [visualizations]
  --viz_samples INT         Number of attention plots to save   [5]

Weights & Biases
  --use_wandb               Enable W&B tracking                 [True]
  --no-use_wandb
  --wandb_project NAME      W&B project name                    [bilstm-sentiment]
  --wandb_run_name NAME     W&B run name (auto-generated)       [None]
  --wandb_entity NAME       W&B username or team                [None]
```

---

### Recommended Runs

#### Baseline — random embeddings

```bash
python train.py \
  --dropout 0.5 \
  --weight_decay 1e-4
```

#### Best single-model — GloVe 200d, frozen

Achieves the best trade-off between validation F1 and overfitting on this dataset.

```bash
python train.py \
  --embedding_type glove \
  --glove_path data/glove.6B.200d.txt \
  --embedding_dim 200 \
  --freeze_embeddings \
  --dropout 0.5 \
  --weight_decay 1e-4
```

#### Ablation — no attention

```bash
python train.py \
  --no-use_attention \
  --embedding_type glove \
  --glove_path data/glove.6B.200d.txt \
  --embedding_dim 200 \
  --freeze_embeddings
```

#### Resume from checkpoint

```bash
python train.py \
  --resume_from checkpoints/epoch_004.pt \
  --embedding_type glove \
  --glove_path data/glove.6B.200d.txt \
  --embedding_dim 200
```
---

## Experiment Tracking — Weights & Biases

Every training run logs the following to a **locally-hosted W&B server** (no cloud account needed):

| What is logged | When |
|---|---|
| All hyper-parameters from `Config` | `wandb.init()` at run start |
| `train/loss`, `train/acc`, `train/f1` | End of every epoch |
| `val/loss`, `val/acc`, `val/f1` | End of every epoch |
| `lr` (current learning rate) | End of every epoch |
| `best-model` artifact (best.pt) | Every time val F1 improves |
| Test classification report | After final evaluation |
| Training history plot (PNG) | After training completes |
| Attention visualisation PNGs | After training completes |

W&B run names are auto-generated from key hyper-parameters, e.g. `glove-h128-attn-cw`, making it easy to compare runs in the UI.

### Starting the local W&B server

```bash
# Start the server (first time: create an account at http://localhost:8080)
docker compose up wandb-server

# Copy .env.example and paste your API key
cp .env.example .env
# edit .env → WANDB_API_KEY=<key from http://localhost:8080/settings>
```

### Training with W&B tracking

```bash
# W&B is enabled by default; set WANDB_BASE_URL to point at the local server
WANDB_BASE_URL=http://localhost:8080 python train.py \
  --embedding_type glove \
  --glove_path data/glove.6B.200d.txt \
  --embedding_dim 200
```

Or via Docker Compose (the `WANDB_BASE_URL` is injected automatically from `docker-compose.yml`):

```bash
docker compose run trainer \
  --embedding_type glove \
  --glove_path data/glove.6B.200d.txt \
  --embedding_dim 200 \
  --freeze_embeddings \
  --dropout 0.5
```

Disable W&B entirely:

```bash
python train.py --no-use_wandb ...
```


---

## Inference Server

After training, `checkpoints/best.pt` and `checkpoints/preprocessor.pkl` (the fitted vocabulary) are used to serve predictions through a **FastAPI** application.

### 9.1 Browser UI

Navigate to `http://localhost:8000` after starting the server.  The UI provides:

- A text area where you type the feedback comment
- An **Analyse** button (or `Ctrl+Enter`)
- Predicted label + confidence (%)
- Horizontal probability bars for all classes
- A **token-level attention heatmap** — each token is coloured proportionally to its attention weight (darker = more attended), with exact percentages shown on hover

### REST API

Full interactive docs are available at `http://localhost:8000/docs` (Swagger UI).

#### `GET /health`

Liveness probe used by both Docker Compose and Kubernetes.

```json
{"status": "ok", "model_loaded": true}
```

#### `GET /model-info`

Returns the training configuration and best validation F1.

```json
{
  "embedding_type": "glove",
  "hidden_size": 128,
  "num_layers": 2,
  "use_attention": true,
  "num_classes": 2,
  "label_names": ["Negative", "Positive"],
  "best_val_f1": 0.6821,
  "trained_epoch": 6,
  "checkpoint_path": "checkpoints/best.pt"
}
```

#### `POST /predict`

**Request**

```json
{"text": "The professor explained every concept very clearly and the assignments were well-designed."}
```

**Response**

```json
{
  "label": "Positive",
  "label_index": 1,
  "confidence": 0.923,
  "probabilities": {
    "Negative": 0.077,
    "Positive": 0.923
  },
  "attention": [
    {"token": "explained",    "weight": 0.1432},
    {"token": "clearly",      "weight": 0.1187},
    {"token": "well-designed","weight": 0.0984},
    ...
  ]
}
```

#### `POST /predict/batch`

Accepts a JSON array of `{"text": "..."}` objects and returns an array of `PredictResponse` objects.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"text": "Great course!"},{"text": "Very confusing lectures."}]'
```

#### Start locally (no Docker)

```bash
pip install -e ".[serve]"
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

---

## Docker

Two separate Dockerfiles keep image sizes appropriate for their purpose:

| Image | Base | PyTorch variant | Approximate size |
|---|---|---|---|
| `Dockerfile` (training) | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | CUDA | ~8 GB |
| `Dockerfile.serve` (serving) | `python:3.12-slim` | CPU-only wheel | ~400 MB |

The serving image is intentionally CPU-only.  Single-sentence inference is fast on CPU and avoids the cost and complexity of exposing a GPU to the serving container.

### 10.1 Training Container

Build and run the trainer in isolation:

```bash
docker build -t bilstm-trainer .

docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/visualizations:/app/visualizations \
  -v $(pwd)/embeddings:/app/embeddings \
  bilstm-trainer \
  --embedding_type glove \
  --glove_path data/glove.6B.200d.txt \
  --embedding_dim 200 \
  --freeze_embeddings \
  --dropout 0.5
```

### Serving Container

Build and run the server in isolation:

```bash
docker build -f Dockerfile.serve -t bilstm-serve .

docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  bilstm-serve
```

Open `http://localhost:8000`.

### Full Stack with Docker Compose

The `docker-compose.yml` defines three services:

| Service | Image | Purpose | URL |
|---|---|---|---|
| `wandb-server` | `wandb/local:latest` | Local W&B experiment tracker | `http://localhost:8080` |
| `trainer` | built from `Dockerfile` | GPU training + W&B logging | — |
| `serve` | built from `Dockerfile.serve` | FastAPI inference server | `http://localhost:8000` |

#### First-time W&B setup

```bash
# 1. Start the W&B server
docker compose up wandb-server

# 2. Open http://localhost:8080 → create an account → copy your API key
# 3. Paste the key into .env
cp .env.example .env
# WANDB_API_KEY=<your-key>
```

#### Training run

```bash
docker compose run trainer \
  --embedding_type glove \
  --glove_path data/glove.6B.200d.txt \
  --embedding_dim 200 \
  --freeze_embeddings \
  --dropout 0.5 \
  --weight_decay 1e-4
```

#### Start inference server

```bash
# Training must have completed first (produces best.pt + preprocessor.pkl)
docker compose up serve
```

#### Start everything

```bash
docker compose up
```

---

## Kubernetes — kubeadm Bare-Metal

The manifests in `k8s/` target a two-node kubeadm cluster: one control node and one compute node (this machine, named **CN01**).

### Architecture



All serving pods are pinned to the compute node via `nodeSelector: kubernetes.io/hostname: CN01` because that is where the `local` PersistentVolume lives (on-disk checkpoint files).  The `ReadOnlyMany` PV allows multiple replicas to mount the same files simultaneously.

### Manifest overview

| File | Resource | Purpose |
|---|---|---|
| `k8s/pv-kubeadm.yaml` | `PersistentVolume` | `local` volume at `/opt/bilstm-checkpoints` on CN01, bound by node affinity |
| `k8s/deployment.yaml` | `Deployment` + `PVC` | 2 replicas, nodeSelector, liveness/readiness probes, resource limits |
| `k8s/service.yaml` | `Service` | `NodePort:30800` — accessible at `http://<ANY_NODE_IP>:30800` |

### Deployment walkthrough

#### 1. Copy checkpoint files onto CN01

```bash
sudo mkdir -p /opt/bilstm-checkpoints
sudo cp checkpoints/best.pt checkpoints/preprocessor.pkl /opt/bilstm-checkpoints/
sudo chmod -R 755 /opt/bilstm-checkpoints
```

#### 2. Build the serving image and load it into containerd

```bash
docker build -f Dockerfile.serve -t bilstm-serve:latest .
docker save bilstm-serve:latest | sudo ctr -n k8s.io images import -
```

> If your cluster uses `crio` instead of `containerd`: `docker save bilstm-serve:latest | sudo podman load`

#### 3. Apply manifests

```bash
kubectl apply -f k8s/pv-kubeadm.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

#### 4. Verify

```bash
kubectl get pods -l app=bilstm-serve
kubectl get svc bilstm-serve

# Should show two Running pods and NodePort 80:30800/TCP
```

#### 5. Access the API

```bash
# From anywhere on the same network
curl http://<CN01_IP>:30800/health

# Browser UI
open http://<CN01_IP>:30800
```

#### Rolling update after retraining

```bash
# Copy new checkpoint files
sudo cp checkpoints/best.pt checkpoints/preprocessor.pkl /opt/bilstm-checkpoints/

# Rebuild and reload the image, then restart pods to pick up the new weights
docker build -f Dockerfile.serve -t bilstm-serve:latest .
docker save bilstm-serve:latest | sudo ctr -n k8s.io images import -
kubectl rollout restart deployment/bilstm-serve
kubectl rollout status  deployment/bilstm-serve
```

---

## Configuration Reference

All parameters are defined in `bilstm_attention/config.py` as a `Config` dataclass.  Every field has a corresponding `--flag` in the CLI.

### Data

| Parameter | Default | Description |
|---|---|---|
| `url` | Mendeley URL | Dataset download URL |
| `data_path` | `data/data.csv` | Local CSV path |
| `text_col` | `comments` | Column name for text |
| `label_col` | `star_rating` | Column name for label |
| `label_names` | `["Negative","Positive"]` | Display names |
| `max_seq_len` | `256` | Token truncation length |
| `test_size` | `0.2` | Test split fraction |
| `val_size` | `0.1` | Validation split fraction |
| `seed` | `42` | Random seed |

### Embedding

| Parameter | Default | Description |
|---|---|---|
| `embedding_type` | `random` | `random` / `glove` / `llm` |
| `embedding_dim` | `100` | Embedding dimensions |
| `glove_path` | `None` | Path to GloVe `.txt` (auto-download if None) |
| `llm_model` | `bert-base-uncased` | HuggingFace model ID |
| `llm_layer` | `-1` | Transformer layer to extract |
| `freeze_embeddings` | `False` | Freeze weights after init |

### Model

| Parameter | Default | Description |
|---|---|---|
| `hidden_size` | `128` | LSTM units per direction |
| `num_layers` | `2` | Stacked LSTM layers |
| `dropout` | `0.3` | Dropout probability |
| `use_attention` | `True` | Bahdanau attention on/off |
| `attn_dim` | `128` | Attention projection size |
| `num_classes` | `2` | Output classes |

### Training

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | `32` | Mini-batch size |
| `learning_rate` | `1e-3` | Adam LR |
| `weight_decay` | `1e-5` | Adam L2 regularisation |
| `use_class_weights` | `True` | Inverse-frequency loss weighting |
| `epochs` | `20` | Maximum epochs |
| `device` | `auto` | `auto` picks CUDA > MPS > CPU |
| `clip_grad` | `1.0` | Gradient clipping max-norm |
| `patience` | `5` | Early-stopping patience |
| `lr_patience` | `2` | LR scheduler patience |

### Checkpointing

| Parameter | Default | Description |
|---|---|---|
| `checkpoint_dir` | `checkpoints` | Directory for `.pt` files |
| `resume_from` | `None` | Path to resume from |

### Visualisation

| Parameter | Default | Description |
|---|---|---|
| `viz_output_dir` | `visualizations` | Output PNG directory |
| `viz_samples` | `5` | Samples to visualise (from test set) |

### Weights & Biases

| Parameter | Default | Description |
|---|---|---|
| `use_wandb` | `True` | Enable tracking |
| `wandb_project` | `bilstm-sentiment` | Project name |
| `wandb_run_name` | `None` | Auto-generated if None |
| `wandb_entity` | `None` | Username or team |

---